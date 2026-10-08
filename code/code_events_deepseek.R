#------------------------------------------------------------------------------#
### SATP: code each event with DeepSeek (is it an event, actions, actors, counts)
#
# Takes the one-incident event texts from split_multi_incident_entries.R and asks
# DeepSeek, in non-thinking mode, to code each one in the layout of
# sandbox/satp_event_schema.md:
#   is_event       the text reports a specific violent act or security operation
#                  (policy news, statements, strike calls, court cases are not events)
#   actions        any of armed_assault, abduction, bombing, infrastructure,
#                  surrender, arrest, seizure (an event can have several)
#   actor, inter1  who carried it out and its type, only when the text names them
#   inter2         who or what was targeted
#   fatalities, injuries
#   successful     FALSE if the act was foiled, defused or failed (still a conflict event)
#   n_incidents    separate acts in the text (more than 1 means the text should be split)
# The ACLED-style event_type and sub_event_type are then derived from these.
#
# Action types and the non-event rule follow the SATP coding manual (Oct 2015).
#
# Inputs:  scraped_data/satp_event_texts.rds    (code/split_multi_incident_entries.R)
#          environment variable DEEPSEEK_API_KEY
# Outputs: scraped_data/coding_cache.rds        (DeepSeek answers, so reruns cost nothing)
#          scraped_data/satp_events_coded.rds/.csv
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("dplyr", "tidyr", "purrr", "stringr", "readr", "rlang", "httr2", "jsonlite")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

events_file <- "../scraped_data/satp_event_texts.rds"
cache_file  <- "../scraped_data/coding_cache.rds"
out_rds     <- "../scraped_data/satp_events_coded.rds"
out_csv     <- "../scraped_data/satp_events_coded.csv"

#------------------------------------------------------------------------------#
####                              Settings                                  ####
#------------------------------------------------------------------------------#
# TRUE: only print how many events need a call, no API calls
DRY_RUN <- FALSE
# number of events to send to DeepSeek, drawn at random from those that need a call;
# NA sends all of them
N_LIMIT <- 2500

# version of SYSTEM_PROMPT; raise it whenever the prompt changes. Events coded under an
# older version are coded again.
PROMPT_VERSION <- 3L

API_URL    <- "https://api.deepseek.com/chat/completions"
MODEL      <- "deepseek-flash"
BATCH_SIZE <- 50     # events per batch; the cache is saved after each batch
N_PARALLEL <- 24     # requests running at the same time

# USD per million tokens, from the DeepSeek pricing page; peak hours are 01:00-04:00 and
# 06:00-10:00 UTC on weekdays and cost twice as much (Chinese public holidays are not tracked)
PRICE_OFFPEAK <- c(hit = 0.003, miss = 0.15, out = 0.6)
PRICE_PEAK    <- 2 * PRICE_OFFPEAK

ACTIONS <- c("armed_assault", "abduction", "bombing", "infrastructure",
             "surrender", "arrest", "seizure")
ACTORS  <- c("State forces", "Rebel group")
TARGETS <- c("State forces", "Rebel group", "Civilians", "Government",
             "Property or infrastructure", "None")

#------------------------------------------------------------------------------#
####                      1. Pick the events to code                        ####
#------------------------------------------------------------------------------#
# the cache is keyed by a hash of the event text, so it survives a rebuild of event_uid
events <- readRDS(events_file) %>%
  mutate(event_key = map_chr(event_text, hash))

cache <- if (file.exists(cache_file)) readRDS(cache_file) else list()

needs_call <- function(key) (cache[[key]]$prompt %||% 0L) < PROMPT_VERSION

todo <- events %>%
  distinct(event_key, event_text) %>%
  filter(map_lgl(event_key, needs_call))

if (!is.na(N_LIMIT)) {
  set.seed(42)
  todo <- slice_sample(todo, n = min(N_LIMIT, nrow(todo)))
}

message("Events in total:         ", nrow(events))
message("Distinct event texts:    ", n_distinct(events$event_key))
message("To send to DeepSeek now: ", nrow(todo),
        " (about ", round(sum(nchar(todo$event_text)) / 4 / 1000), "k text tokens)")

if (DRY_RUN) stop("DRY_RUN is TRUE: set it to FALSE to call DeepSeek.")

#------------------------------------------------------------------------------#
####                         2. Ask DeepSeek                                ####
#------------------------------------------------------------------------------#
SYSTEM_PROMPT <- paste(
  "You code one news summary about political violence in India as a single event. Answer in json.",
  "",
  "is_event: true only if the text reports a specific violent act or security operation that took place (an attack, clash or encounter, bombing or blast, killing, abduction or kidnapping, arson or sabotage of property or infrastructure, arrest, surrender, or seizure of weapons, cash or other strategic goods). false for non-events: policy changes, announcements by governments or militants, statements and speeches, calls for strikes or curfews, protests, talks, court and legal proceedings, analysis and background reports.",
  "",
  "actions: every action type that took place in the event, from this list; an empty list if is_event is false.",
  "- armed_assault: a broad category for an attack on people, including shootings, encounters and killings.",
  "- abduction: hijacking or kidnapping that is reported as happening in this event. The rescue, release or escape of a person abducted earlier is not a new abduction.",
  "- bombing: explosions, blasts, and IED or grenade attacks. It includes explosive devices planted or placed to attack (on roads, bridges, in vehicles or at targets) that were found, defused or did not go off: a foiled attempt still counts as a bombing.",
  "- infrastructure: an attack on property or infrastructure, including arson and sabotage.",
  "- surrender: militants surrendering to security forces.",
  "- arrest: a security operation that leads to arrests.",
  "- seizure: a security operation in which forces actually recover or seize weapons, ammunition, cash or other strategic goods, for example an arms cache or stored explosives. A search that finds nothing is not a seizure, and a device planted to attack is a bombing, not a seizure.",
  "",
  "actor_type: who carried out the act. \"State forces\" (police, army, paramilitary and other security forces) or \"Rebel group\" (militant, insurgent, Maoist or terrorist groups, including unidentified terrorists, militants, extremists or gunmen). null if the text does not describe the perpetrator or it is unclear who acted, for example an exchange of fire where the text does not say who started it. actor: the name of the group or force as written in the text, or null if it is not named.",
  "",
  "target_type: who or what was targeted, only if actor_type is not null. One of \"State forces\", \"Rebel group\", \"Civilians\", \"Government\" (officials and politicians), \"Property or infrastructure\", or \"None\" (for example a surrender). If State forces are the actor and no target is stated, the target is \"Rebel group\". If the people arrested or targeted are not described as militants or insurgents (for example counterfeit currency or drug cases), or it is unclear, target_type is null.",
  "",
  "fatalities and injuries: the number of people killed and injured as reported in the text, 0 if none.",
  "",
  "successful: false if the act was foiled, defused, aborted or failed (for example a bomb that was planted but did not go off), otherwise true.",
  "",
  "n_incidents: the number of separate acts the text describes, counting acts at different places or times as separate; 1 for a single act, 0 if is_event is false.",
  "",
  "Answer like this: {\"is_event\": true, \"actions\": [\"armed_assault\"], \"actor\": \"CPI-Maoist\", \"actor_type\": \"Rebel group\", \"target_type\": \"State forces\", \"fatalities\": 2, \"injuries\": 0, \"successful\": true, \"n_incidents\": 1}",
  sep = "\n")

# one chat request per event text
build_request <- function(text) {
  request(API_URL) %>%
    req_auth_bearer_token(Sys.getenv("DEEPSEEK_API_KEY")) %>%
    req_body_json(list(
      model           = MODEL,
      temperature     = 0,
      thinking        = list(type = "disabled"),
      response_format = list(type = "json_object"),
      messages        = list(list(role = "system", content = SYSTEM_PROMPT),
                             list(role = "user",   content = text)))) %>%
    req_retry(max_tries = 4)
}

# the parsed answer, or NULL if the call failed or the reply is not valid json
read_answer <- function(resp) {
  if (!inherits(resp, "httr2_response")) return(NULL)
  tryCatch(fromJSON(resp_body_json(resp)$choices[[1]]$message$content, simplifyVector = FALSE),
           error = function(e) NULL)
}

# token counts DeepSeek reports for one response: cached input, uncached input, output
read_usage <- function(resp) {
  none <- c(hit = 0, miss = 0, out = 0)
  if (!inherits(resp, "httr2_response")) return(none)
  tryCatch({
    u <- resp_body_json(resp)$usage
    c(hit = u$prompt_cache_hit_tokens %||% 0, miss = u$prompt_cache_miss_tokens %||% 0,
      out = u$completion_tokens %||% 0)
  }, error = function(e) none)
}

is_peak <- function(time) {
  t <- as.POSIXlt(time, tz = "UTC")
  t$wday %in% 1:5 && t$hour %in% c(1:3, 6:9)
}

cost_usd <- function(usage, time) {
  sum(usage * if (is_peak(time)) PRICE_PEAK else PRICE_OFFPEAK) / 1e6
}

# keeps only valid values; returns NULL if is_event is missing
read_coding <- function(answer) {
  if (!is.logical(answer$is_event) || length(answer$is_event) != 1) return(NULL)
  actions <- unlist(answer$actions)
  count   <- function(x) suppressWarnings(as.integer(x %||% 0L))
  list(is_event    = answer$is_event,
       actions     = actions[actions %in% ACTIONS],
       actor       = answer$actor %||% NA_character_,
       actor_type  = if (isTRUE(answer$actor_type %in% ACTORS)) answer$actor_type else NA_character_,
       target_type = if (isTRUE(answer$target_type %in% TARGETS)) answer$target_type else NA_character_,
       fatalities  = count(answer$fatalities),
       injuries    = count(answer$injuries),
       successful  = if (isTRUE(is.logical(answer$successful))) answer$successful else NA,
       n_incidents = count(answer$n_incidents))
}

# cache: one element per event text, named by its hash
#   status "ok"      -> coding holds the parsed answer
#   status "invalid" -> DeepSeek answered but the reply was unusable
#   prompt, model    -> PROMPT_VERSION and MODEL that gave the answer
# events whose call failed are not cached, so a rerun tries them again
message("\nCalling DeepSeek for ", nrow(todo), " events")

batches    <- split(seq_len(nrow(todo)), ceiling(seq_len(nrow(todo)) / BATCH_SIZE))
start_time  <- Sys.time()
usage_total <- c(hit = 0, miss = 0, out = 0)

for (b in seq_along(batches)) {
  rows  <- todo[batches[[b]], ]
  resps <- req_perform_parallel(map(rows$event_text, build_request),
                                on_error = "continue", max_active = N_PARALLEL, progress = FALSE)

  answers <- map(resps, read_answer)
  failed  <- map_lgl(answers, is.null)
  usage_total <- usage_total + reduce(map(resps, read_usage), `+`)

  for (j in which(!failed)) {
    coding <- read_coding(answers[[j]])
    cache[[rows$event_key[j]]] <-
      if (is.null(coding)) list(status = "invalid", prompt = PROMPT_VERSION, model = MODEL)
      else list(status = "ok", coding = coding, prompt = PROMPT_VERSION, model = MODEL)
  }
  saveRDS(cache, cache_file)

  if (all(failed)) {
    first_error <- resps[[1]]
    stop("Every call in the batch failed (check the API key, the balance and MODEL). First error: ",
         if (inherits(first_error, "condition")) conditionMessage(first_error) else "invalid reply")
  }

  n_done  <- max(batches[[b]])
  minutes <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  message("  ", n_done, " of ", nrow(todo), " done; ", sum(failed), " failed in this batch; ",
          round(minutes, 1), " min elapsed, about ", round(minutes * (nrow(todo) - n_done) / n_done, 1),
          " min left; cost so far $", round(cost_usd(usage_total, start_time), 3))
}

message("\nTokens this run: ", format(usage_total[["hit"]] + usage_total[["miss"]], big.mark = ","),
        " input (", format(usage_total[["hit"]], big.mark = ","), " cached), ",
        format(usage_total[["out"]], big.mark = ","), " output")
message("Cost this run: $", round(cost_usd(usage_total, start_time), 3),
        " ($", round(cost_usd(usage_total, start_time) / nrow(todo) * 1000, 3), " per 1,000 events)")

#------------------------------------------------------------------------------#
####                  3. Attach the coding and derive the type              ####
#------------------------------------------------------------------------------#
coded <- bind_rows(imap(cache, function(x, key) {
  if (x$status != "ok" || x$prompt < PROMPT_VERSION) return(NULL)
  with(x$coding, tibble(
    event_key = key, is_event = is_event, actor = actor, inter1 = actor_type,
    inter2 = target_type, fatalities = fatalities, injuries = injuries,
    successful = successful, n_incidents = n_incidents,
    armed_assault  = as.integer("armed_assault"  %in% actions),
    abduction      = as.integer("abduction"      %in% actions),
    bombing        = as.integer("bombing"        %in% actions),
    infrastructure = as.integer("infrastructure" %in% actions),
    surrender      = as.integer("surrender"      %in% actions),
    arrest         = as.integer("arrest"         %in% actions),
    seizure        = as.integer("seizure"        %in% actions)))
}))

# the first matching rule gives the ACLED-style type (priority order in the schema spec)
events_coded <- events %>%
  left_join(coded, by = "event_key") %>%
  mutate(
    civilian_targeting = inter2 == "Civilians",
    sub_event_type = case_when(
      !is_event               ~ NA_character_,
      abduction == 1          ~ "Abduction/forced disappearance",
      bombing == 1            ~ "Bombing",
      armed_assault == 1 & civilian_targeting %in% TRUE ~ "Attack",
      armed_assault == 1      ~ "Armed clash",
      infrastructure == 1     ~ "Looting/property destruction",
      arrest == 1             ~ "Arrests",
      seizure == 1            ~ "Disrupted weapons use",
      surrender == 1          ~ "Change to group/activity"),
    event_type = case_when(
      sub_event_type == "Abduction/forced disappearance" ~ "Violence against civilians",
      sub_event_type == "Attack"                         ~ "Violence against civilians",
      sub_event_type == "Bombing"                        ~ "Explosions/Remote violence",
      sub_event_type == "Armed clash"                    ~ "Battles",
      !is.na(sub_event_type)                             ~ "Strategic developments"))

message("\nEvents coded: ", sum(!is.na(events_coded$is_event)), " of ", nrow(events_coded))
message("Share that are events: ", round(mean(events_coded$is_event, na.rm = TRUE), 3))
message("Events with more than one incident: ", sum(events_coded$n_incidents > 1, na.rm = TRUE))
message("Events with no actor type: ", sum(events_coded$is_event & is.na(events_coded$inter1), na.rm = TRUE))
print(count(filter(events_coded, is_event), sub_event_type, sort = TRUE))

saveRDS(events_coded, out_rds)
write_csv(events_coded, out_csv)
