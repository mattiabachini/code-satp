#------------------------------------------------------------------------------#
### SATP: split entries that describe several incidents into separate events
#
# Most SATP entries describe one incident. Some describe several (a daily digest,
# "in another incident ..."). This script turns each entry into one or more event
# texts, so every later step (location, actor, action type) works on one incident.
#
# The steps:
#   1. Flag entries that might hold several incidents (cheap text rules).
#   2. Ask DeepSeek to label the sentences of each flagged entry as parts of a main
#      incident or as context, then attach the context to the nearest main incident.
#   3. Build one row per event. Unflagged entries become one event unchanged.
#
# DeepSeek never writes text. It only returns sentence numbers, so every event text
# is made of the original sentences, word for word.
#
# Inputs:  scraped_data/scraped_incidents.rds   (code/scrape_data_merge.R)
#          environment variable DEEPSEEK_API_KEY (only when DRY_RUN is FALSE)
# Outputs: scraped_data/split_cache.rds         (DeepSeek answers, so reruns cost nothing)
#          scraped_data/satp_event_texts.rds/.csv  (n_chars, n_dates, has_cue: the entry's
#          flagging variables, to compare split rates by rule)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("dplyr", "tidyr", "purrr", "stringr", "readr", "httr2", "jsonlite")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

incident_file <- "../scraped_data/scraped_incidents.rds"
cache_file    <- "../scraped_data/split_cache.rds"
out_rds       <- "../scraped_data/satp_event_texts.rds"
out_csv       <- "../scraped_data/satp_event_texts.csv"

#------------------------------------------------------------------------------#
####                              Settings                                  ####
#------------------------------------------------------------------------------#
# TRUE: only flag entries and print counts, no API calls
DRY_RUN <- FALSE
# number of entries to send to DeepSeek, drawn at random from those that need a call;
# NA sends all of them
N_LIMIT <- NA

# an entry is flagged if it matches at least one of these rules
MIN_CHARS   <- 900   # long text
MIN_DATES   <- 2     # two or more different dates written out ("March 3", "March 5")
CUE_PHRASES <- c("in another incident", "in a separate incident", "in yet another",
                 "separately", "meanwhile", "in a related development", "elsewhere",
                 "separate (incident|attack|operation|encounter|blast|explosion|firing|case)",
                 "(two|three|four|five|six|several|multiple) (separate )?(incidents|attacks|encounters|operations|blasts|explosions|killings|arrests|abductions|clashes)",
                 "in another", "another incident", "in one incident", "first incident",
                 "second incident", "a series of")

# a date returned by DeepSeek is used only if it is within this many days of the entry date
MAX_DATE_SHIFT_DAYS <- 60

# version of SYSTEM_PROMPT; raise it whenever the prompt changes. An entry answered under an
# older version is asked again if that answer split it (or was unusable); an entry that older
# versions judged to be one incident stays as it is, because a stricter prompt only splits less.
PROMPT_VERSION <- 3L

# USD per million tokens, from the DeepSeek pricing page; peak hours are 01:00-04:00 and
# 06:00-10:00 UTC on weekdays and cost twice as much (Chinese public holidays are not tracked)
PRICE_OFFPEAK <- c(hit = 0.003, miss = 0.15, out = 0.6)
PRICE_PEAK    <- 2 * PRICE_OFFPEAK

API_URL    <- "https://api.deepseek.com/chat/completions"
MODEL      <- "deepseek-flash"
BATCH_SIZE <- 50     # entries per batch; the cache is saved after each batch
N_PARALLEL <- 24      # requests running at the same time

#------------------------------------------------------------------------------#
####                      1. Flag entries to check                          ####
#------------------------------------------------------------------------------#
# a sentence ends at . ! or ? followed by a space and a capital letter or quote
SENTENCE_BREAK <- "(?<=[.!?])\\s+(?=[A-Z\"'])"
DATE_PATTERN   <- regex(paste0("(", paste(month.name, collapse = "|"), ")\\s+\\d{1,2}"))

incidents <- readRDS(incident_file) %>%
  mutate(
    sentences   = str_split(incident_summary, SENTENCE_BREAK),
    n_sentences = lengths(sentences),
    n_chars     = nchar(incident_summary),
    n_dates     = map_int(str_extract_all(incident_summary, DATE_PATTERN), n_distinct),
    has_cue     = str_detect(str_to_lower(incident_summary), paste(CUE_PHRASES, collapse = "|")),
    # a one-sentence entry cannot hold two incidents
    check_for_split = n_sentences >= 2 & (n_chars >= MIN_CHARS | n_dates >= MIN_DATES | has_cue)
  )

cache <- if (file.exists(cache_file)) readRDS(cache_file) else list()

# cache: one element per entry, named by incident_uid
#   status "ok"      -> groups holds the incidents
#   status "invalid" -> DeepSeek answered but the grouping was unusable
#   prompt           -> PROMPT_VERSION the answer was given under (absent in version 1)
#   model            -> MODEL that gave the answer (absent before the model was recorded)
# entries whose call failed are not cached, so a rerun tries them again
needs_call <- function(uid) {
  answer <- cache[[as.character(uid)]]
  if (is.null(answer)) return(TRUE)
  older_prompt <- (answer$prompt %||% 1L) < PROMPT_VERSION
  older_prompt && (answer$status != "ok" || length(answer$groups) > 1)
}

candidates <- incidents %>% filter(check_for_split)
todo       <- candidates %>% filter(map_lgl(incident_uid, needs_call))
if (!is.na(N_LIMIT)) {
  set.seed(42)
  todo <- slice_sample(todo, n = min(N_LIMIT, nrow(todo)))
}

message("Entries in total:        ", nrow(incidents))
message("Flagged for checking:    ", sum(incidents$check_for_split),
        " (", round(100 * mean(incidents$check_for_split), 1), "%)")
message("  long text:             ", sum(incidents$n_sentences >= 2 & incidents$n_chars >= MIN_CHARS))
message("  several dates:         ", sum(incidents$n_sentences >= 2 & incidents$n_dates >= MIN_DATES))
message("  cue phrase:            ", sum(incidents$n_sentences >= 2 & incidents$has_cue))
message("Flagged and need a call: ", sum(map_lgl(candidates$incident_uid, needs_call)))
message("To send to DeepSeek now: ", nrow(todo),
        " (about ", round(sum(todo$n_chars) / 4 / 1000), "k text tokens)")

if (DRY_RUN) stop("DRY_RUN is TRUE: set it to FALSE to call DeepSeek.")

#------------------------------------------------------------------------------#
####                         2. Ask DeepSeek                                ####
#------------------------------------------------------------------------------#
SYSTEM_PROMPT <- paste(
  "You group the numbered sentences of a news summary about violence in India into incidents.",
  "A main incident is one specific act (an attack, clash, blast, killing, abduction, arrest, surrender, seizure or similar) at one place and time, reported as happening at or around the publication date.",
  "Everything else is context: background on a person, group or place; earlier acts mentioned only to explain the main one; follow-up details, results and casualty counts; statements, statistics and policy announcements; calls for strikes, curfews or meetings; legal proceedings.",
  "Give each main incident its own group with role \"main\". Two or more main groups are allowed only when the summary reports distinct acts, each with its own place or time.",
  "Put context sentences in groups with role \"context\".",
  "If the summary reports no specific act, return one group with role \"main\" holding all sentences.",
  "Every sentence number must appear in exactly one group.",
  "For a main group also give the date the act happened as YYYY-MM-DD if the text states it, otherwise null; context groups have date null.",
  "Answer in json: {\"incidents\": [{\"sentences\": [1, 2], \"role\": \"main\", \"date\": \"2009-03-04\"}, {\"sentences\": [3], \"role\": \"context\", \"date\": null}]}",
  sep = "\n")

# one chat request per entry; the sentences are numbered so the answer can refer to them
build_request <- function(entry_date, sentences) {
  numbered <- paste0("[", seq_along(sentences), "] ", sentences, collapse = "\n")
  user_msg <- paste0("The summary was published on ", entry_date, ".\n\n", numbered)

  request(API_URL) %>%
    req_auth_bearer_token(Sys.getenv("DEEPSEEK_API_KEY")) %>%
    req_body_json(list(
      model           = MODEL,
      temperature     = 0,
      response_format = list(type = "json_object"),
      messages        = list(list(role = "system", content = SYSTEM_PROMPT),
                             list(role = "user",   content = user_msg)))) %>%
    req_retry(max_tries = 4)
}

# the parsed answer, or NULL if the call failed or the reply is not valid json
read_answer <- function(resp) {
  if (!inherits(resp, "httr2_response")) return(NULL)
  tryCatch(fromJSON(resp_body_json(resp)$choices[[1]]$message$content, simplifyVector = FALSE),
           error = function(e) NULL)
}

# context sentences join the main incident whose sentences are closest by position
# (ties go to the earlier incident); a role other than "context" counts as main.
# With no main group, the entry becomes one incident.
attach_context <- function(groups) {
  is_main <- tolower(map_chr(groups, "role")) != "context"
  if (!any(is_main)) {
    return(list(list(ids = sort(unlist(map(groups, "ids"))), date = groups[[1]]$date)))
  }
  main <- groups[is_main]
  main <- main[order(map_dbl(main, ~ min(.x$ids)))]
  anchors <- map(main, "ids")
  for (id in unlist(map(groups[!is_main], "ids"))) {
    k <- which.min(map_dbl(anchors, ~ min(abs(.x - id))))
    main[[k]]$ids <- c(main[[k]]$ids, id)
  }
  map(main, ~ list(ids = sort(.x$ids), date = .x$date))
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

# turns the answer into a list of incidents (sentence numbers and date);
# returns NULL unless every sentence is assigned exactly once
read_groups <- function(answer, n_sentences) {
  groups <- map(answer$incidents, ~ list(ids  = as.integer(unlist(.x$sentences)),
                                         role = .x$role %||% "main",
                                         date = .x$date))
  groups   <- Filter(function(g) length(g$ids) > 0, groups)
  assigned <- sort(unlist(map(groups, "ids")))
  if (length(groups) > 0 && identical(assigned, seq_len(n_sentences))) attach_context(groups) else NULL
}

message("\nCalling DeepSeek for ", nrow(todo), " entries")

# the entries are sent in batches; each batch runs N_PARALLEL requests at a time and is
# saved to the cache when it finishes
batches    <- split(seq_len(nrow(todo)), ceiling(seq_len(nrow(todo)) / BATCH_SIZE))
start_time  <- Sys.time()
usage_total <- c(hit = 0, miss = 0, out = 0)

for (b in seq_along(batches)) {
  rows  <- todo[batches[[b]], ]
  reqs  <- map2(as.character(rows$date), rows$sentences, build_request)
  resps <- req_perform_parallel(reqs, on_error = "continue", max_active = N_PARALLEL, progress = FALSE)

  answers <- map(resps, read_answer)
  failed  <- map_lgl(answers, is.null)
  usage_total <- usage_total + reduce(map(resps, read_usage), `+`)

  for (j in which(!failed)) {
    groups <- read_groups(answers[[j]], rows$n_sentences[j])
    cache[[as.character(rows$incident_uid[j])]] <-
      if (is.null(groups)) list(status = "invalid", prompt = PROMPT_VERSION, model = MODEL)
      else list(status = "ok", groups = groups, prompt = PROMPT_VERSION, model = MODEL)
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
        " ($", round(cost_usd(usage_total, start_time) / max(nrow(todo), 1) * 1000, 3), " per 1,000 entries)")

#------------------------------------------------------------------------------#
####                       3. Build one row per event                       ####
#------------------------------------------------------------------------------#
# the date from DeepSeek, or the entry date if it is missing, unparseable or far from the entry date
usable_date <- function(llm_date, entry_date) {
  d  <- tryCatch(as.Date(llm_date %||% NA_character_), error = function(e) as.Date(NA))
  ok <- !is.na(d) && abs(as.numeric(d - entry_date)) <= MAX_DATE_SHIFT_DAYS
  as.character(if (ok) d else entry_date)
}

# split_status:
#   not_checked -> no rule flagged the entry
#   single      -> checked, DeepSeek found one incident
#   split       -> checked, DeepSeek found several incidents
#   unresolved  -> flagged, but not sent (N_LIMIT), the call failed, or the grouping was
#                  unusable; kept as one event
events <- incidents %>%
  mutate(cached = map(as.character(incident_uid), ~ cache[[.x]] %||% list(status = "none"))) %>%
  mutate(events = pmap(list(date, sentences, incident_summary, check_for_split, cached),
    function(entry_date, sents, summary, flagged, cached) {
      version <- if (cached$status == "none") NA_integer_ else cached$prompt %||% 1L
      if (cached$status == "ok") {
        tibble(event_part  = seq_along(cached$groups),
               event_text  = map_chr(cached$groups, ~ paste(sents[.x$ids], collapse = " ")),
               event_date  = as.Date(map_chr(cached$groups, ~ usable_date(.x$date, entry_date))),
               split_status = if (length(cached$groups) > 1) "split" else "single",
               prompt_version = version)
      } else {
        tibble(event_part = 1L, event_text = summary, event_date = entry_date,
               split_status = if (flagged) "unresolved" else "not_checked",
               prompt_version = version)
      }
    })) %>%
  select(incident_uid, series, source_ids, n_chars, n_dates, has_cue, events) %>%
  unnest(events) %>%
  arrange(incident_uid, event_part) %>%
  mutate(event_uid = row_number(), .before = 1)

message("\nEntries: ", n_distinct(events$incident_uid), "; events: ", nrow(events))
print(count(events, split_status))

saveRDS(events, out_rds)
write_csv(events, out_csv)
