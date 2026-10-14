#------------------------------------------------------------------------------#
### SATP: paired comparison of deepseek-flash and deepseek-chat on the same events
#
# 500 four-action events with a unit, drawn at random from the analysis set (satp_events_conf07.rds),
# in two halves of 250. Both models code every event with the prompt and settings of
# code_events_deepseek.R. In half A deepseek-flash runs first and deepseek-chat second; in half B the
# order is reversed, so a cache that serves the second run (same prompt, same event text) helps each
# model once. Every call saves the served model name, the token usage (cached input, uncached input,
# output), the time and the reply.
#
# Cost is shown three ways at the deepseek-flash price list: as billed (cached and uncached input,
# peak and off-peak by call time), with every input token uncached, and with the system prompt cached.
# Tokens per event and the cached share of input, by model and by position (first or second run),
# show how much of any cost gap is the cache.
#
# Input:   scraped_data of the Irrigating Peace repository (satp_events_conf07.rds), code_events_deepseek.R
#          (the system prompt), environment variable DEEPSEEK_API_KEY
# Outputs: data/model_comparison_calls.rds       one row per call
#          data/model_comparison_summary.csv     tokens and cost by model and position
#          data/model_comparison_discordant.csv  events where the two models differ, with both answers
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

conf07_file <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/satp_events_conf07.rds"

N_PER_HALF <- 250L
N_PARALLEL <- 24L
BATCH_SIZE <- 50L
API_URL    <- "https://api.deepseek.com/chat/completions"

# USD per million tokens of deepseek-flash; peak hours are 01:00-04:00 and 06:00-10:00 UTC on weekdays
PRICE_OFFPEAK <- c(hit = 0.003, miss = 0.15, out = 0.6)

# the system prompt of code_events_deepseek.R, evaluated from that file
src <- readLines("code_events_deepseek.R")
i0  <- grep("^SYSTEM_PROMPT <-", src)
i1  <- i0 + which(grepl('sep = "\\n")', src[i0:length(src)], fixed = TRUE))[1] - 1
eval(parse(text = src[i0:i1]))

#------------------------------------------------------------------------------#
####                         1. The 500 events                              ####
#------------------------------------------------------------------------------#
set.seed(2026)
events <- readRDS(conf07_file) %>%
  filter(year %in% 2000:2024, any_analysis_action %in% 1, !is.na(unit_id), !is.na(notes)) %>%
  distinct(event_text = notes) %>%
  slice_sample(n = 2 * N_PER_HALF) %>%
  mutate(event_id = row_number(), half = rep(c("A", "B"), each = N_PER_HALF))

message("Events: ", nrow(events), "; mean text length ", round(mean(nchar(events$event_text))), " characters")

#------------------------------------------------------------------------------#
####                         2. Ask both models                             ####
#------------------------------------------------------------------------------#
build_request <- function(text, model) {
  request(API_URL) %>%
    req_auth_bearer_token(Sys.getenv("DEEPSEEK_API_KEY")) %>%
    req_body_json(c(
      list(model           = model,
           temperature     = 0,
           response_format = list(type = "json_object"),
           messages        = list(list(role = "system", content = SYSTEM_PROMPT),
                                  list(role = "user",   content = text))),
      if (model != "deepseek-chat") list(thinking = list(type = "disabled")))) %>%
    req_retry(max_tries = 4)
}

read_call <- function(resp, time) {
  body <- if (inherits(resp, "httr2_response")) tryCatch(resp_body_json(resp), error = function(e) NULL) else NULL
  usage <- body$usage
  tibble(model_served = body$model %||% NA_character_,
         hit   = as.numeric(usage$prompt_cache_hit_tokens %||% NA),
         miss  = as.numeric(usage$prompt_cache_miss_tokens %||% NA),
         out   = as.numeric(usage$completion_tokens %||% NA),
         reply = body$choices[[1]]$message$content %||% NA_character_,
         time  = time)
}

run_model <- function(rows, model) {
  batches <- split(seq_len(nrow(rows)), ceiling(seq_len(nrow(rows)) / BATCH_SIZE))
  map_dfr(batches, function(idx) {
    time  <- Sys.time()
    resps <- req_perform_parallel(map(rows$event_text[idx], build_request, model = model),
                                  on_error = "continue", max_active = N_PARALLEL, progress = FALSE)
    bind_cols(tibble(event_id = rows$event_id[idx]), map_dfr(resps, read_call, time = time))
  }) %>% mutate(model_asked = model)
}

runs <- tibble(half = c("A", "A", "B", "B"), position = c(1L, 2L, 1L, 2L),
               model = c("deepseek-flash", "deepseek-chat", "deepseek-chat", "deepseek-flash"))

calls <- pmap_dfr(runs, function(half, position, model) {
  message("Half ", half, ", run ", position, ": ", model)
  run_model(events[events$half == half, ], model) %>% mutate(half = half, position = position)
})

saveRDS(calls, "../data/model_comparison_calls.rds")

#------------------------------------------------------------------------------#
####                        3. Tokens and cost                              ####
#------------------------------------------------------------------------------#
is_peak <- function(time) {
  t <- as.POSIXlt(time, tz = "UTC")
  t$wday %in% 1:5 & t$hour %in% c(1:3, 6:9)
}

# tokens of the system prompt: input tokens of a call minus about four characters per token of its event text
calls <- calls %>% left_join(events %>% select(event_id, event_text), by = "event_id")
prompt_tokens <- median(calls$hit + calls$miss - nchar(calls$event_text) / 4, na.rm = TRUE)
message("System prompt: about ", round(prompt_tokens), " tokens")

calls <- calls %>%
  mutate(peak = is_peak(time),
         price_factor = if_else(peak, 2, 1),
         cost_billed   = (hit * PRICE_OFFPEAK[["hit"]] + miss * PRICE_OFFPEAK[["miss"]] + out * PRICE_OFFPEAK[["out"]]) *
                           price_factor / 1e6,
         cost_all_miss = ((hit + miss) * PRICE_OFFPEAK[["miss"]] + out * PRICE_OFFPEAK[["out"]]) * price_factor / 1e6,
         cost_prefix_cached = (prompt_tokens * PRICE_OFFPEAK[["hit"]] + (hit + miss - prompt_tokens) * PRICE_OFFPEAK[["miss"]] +
                                 out * PRICE_OFFPEAK[["out"]]) * price_factor / 1e6)

by_group <- function(d, ...) {
  d %>% group_by(...) %>%
    summarise(calls = n(), failed = sum(is.na(reply)),
              input_tokens = round(mean(hit + miss, na.rm = TRUE)),
              cached_share = round(sum(hit, na.rm = TRUE) / sum(hit + miss, na.rm = TRUE), 3),
              output_tokens = round(mean(out, na.rm = TRUE), 1),
              usd_per_1000_billed = round(1000 * mean(cost_billed, na.rm = TRUE), 4),
              usd_per_1000_all_uncached = round(1000 * mean(cost_all_miss, na.rm = TRUE), 4),
              usd_per_1000_prefix_cached = round(1000 * mean(cost_prefix_cached, na.rm = TRUE), 4),
              peak_share = round(mean(peak), 2), .groups = "drop")
}

summary_tbl <- bind_rows(by_group(calls, model_asked) %>% mutate(position = NA_integer_),
                         by_group(calls, model_asked, position))
write_csv(summary_tbl, "../data/model_comparison_summary.csv")

message("\nModel names the API reported:")
print(count(calls, model_asked, model_served))
message("\nTokens and cost per 1,000 events at the deepseek-flash price list (position NA = both runs):")
print(summary_tbl, width = Inf)

message("\nTotal billed for the 1,000 calls: $", round(sum(calls$cost_billed, na.rm = TRUE), 3))

#------------------------------------------------------------------------------#
####                     4. Do the two models agree?                        ####
#------------------------------------------------------------------------------#
answer_fields <- function(reply) {
  a <- tryCatch(fromJSON(reply, simplifyVector = FALSE), error = function(e) list())
  acts <- unlist(a$actions)
  tibble(valid         = is.logical(a$is_event),
         is_event      = a$is_event %||% NA,
         in_india      = a$in_india %||% NA,
         armed_assault = "armed_assault" %in% acts,
         abduction     = "abduction" %in% acts,
         bombing       = "bombing" %in% acts,
         infrastructure = "infrastructure" %in% acts,
         fatalities    = suppressWarnings(as.integer(a$fatalities %||% NA)),
         n_incidents   = suppressWarnings(as.integer(a$n_incidents %||% NA)))
}

coded <- bind_cols(calls %>% select(event_id, model_asked, position), map_dfr(calls$reply, answer_fields)) %>%
  mutate(real_event = is_event & coalesce(in_india, TRUE),
         four_action = armed_assault | abduction | bombing | infrastructure,
         model = sub("deepseek-", "", model_asked))

message("\nValid replies by model:")
print(coded %>% group_by(model) %>% summarise(valid = sum(valid), of = n(), .groups = "drop"))

wide <- coded %>%
  select(event_id, model, real_event, is_event, in_india, four_action, armed_assault, abduction, bombing,
         infrastructure, fatalities, n_incidents) %>%
  pivot_wider(names_from = model, values_from = c(real_event, is_event, in_india, four_action, armed_assault, abduction,
                                                  bombing, infrastructure, fatalities, n_incidents))

measures <- c("is_event", "in_india", "real_event", "four_action", "armed_assault", "abduction", "bombing",
              "infrastructure", "fatalities", "n_incidents")
agree <- map_dfr(measures, function(m) {
  a <- wide[[paste0(m, "_flash")]]; b <- wide[[paste0(m, "_chat")]]
  tibble(measure = m, identical = sum(a == b | (is.na(a) & is.na(b)), na.rm = TRUE), of = length(a))
})
message("\nEvents where flash and chat give the same answer:")
print(agree)

discordant <- wide %>%
  filter(real_event_flash != real_event_chat | four_action_flash != four_action_chat |
           fatalities_flash != fatalities_chat) %>%
  left_join(events %>% select(event_id, event_text), by = "event_id")
write_csv(discordant, "../data/model_comparison_discordant.csv")
message("\nEvents where the models differ on real event, four-action or fatalities: ", nrow(discordant),
        " (written to data/model_comparison_discordant.csv, to be hand-checked)")
