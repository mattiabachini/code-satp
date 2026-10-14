#------------------------------------------------------------------------------#
### SATP: DeepSeek's coding of the validation sample against the hand check
#
# Compares code_events_deepseek.R, run on the 300 validation events, with the hand check
# (compare_validation_sample.R): is it an event, which of the four analysis actions, and the
# counts of people killed, injured and arrested.
#
# A real event is an event in India: is_event and in_india for both. A DeepSeek output from a
# prompt without the in_india field (versions 4 and earlier) counts every event as in India.
#
# Input:   scraped_data/satp_events_coded_test.rds   (SATP_TEST_FILE=../data/validation_sample_events.rds
#                                                     Rscript code_events_deepseek.R); environment
#                                                     variable SATP_CODED_FILE names another output file
#                                                     and SATP_TAG (for example _chat) is appended to the output names
#          data/validation_sample_claude_coded.csv    (hand check with counts)
# Outputs: data/deepseek_validation_summary.csv       agreement by measure
#          data/deepseek_validation_disagreements.csv events where DeepSeek and the hand check differ
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

library(tidyverse)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

ACTIONS <- c("armed_assault", "bombing", "infrastructure", "abduction")
COUNTS  <- c("fatalities", "injuries", "arrested")

hand <- read_csv("../data/validation_sample_claude_coded.csv", show_col_types = FALSE) %>%
  rename_with(~ paste0("cl_", .x), -id) %>%
  mutate(cl_four_action = (cl_armed_assault + cl_bombing + cl_infrastructure + cl_abduction) > 0)

coded_file <- Sys.getenv("SATP_CODED_FILE", unset = "../scraped_data/satp_events_coded_test.rds")
tag        <- Sys.getenv("SATP_TAG", unset = "")

ds <- readRDS(coded_file) %>%
  select(id, event_text, stratum, ds_is_event = is_event, ds_actor_type = inter1,
         ds_armed_assault = armed_assault, ds_bombing = bombing, ds_infrastructure = infrastructure,
         ds_abduction = abduction, ds_fatalities = fatalities, ds_injuries = injuries,
         ds_arrested = arrested, any_of(c(ds_in_india = "in_india"))) %>%
  mutate(ds_four_action = (ds_armed_assault + ds_bombing + ds_infrastructure + ds_abduction) > 0)

if (!"ds_in_india" %in% names(ds)) ds$ds_in_india <- NA
d <- inner_join(hand, ds, by = "id")
message("Events compared: ", nrow(d), " of 300; DeepSeek returned no answer for ", sum(is.na(d$ds_is_event)))
d <- filter(d, !is.na(ds_is_event)) %>%
  mutate(cl_real = cl_is_event == 1 & cl_in_india == 1,
         ds_real = ds_is_event & coalesce(ds_in_india, TRUE))

#------------------------------------------------------------------------------#
####                              Is it an event                            ####
#------------------------------------------------------------------------------#
message("\nIs it an event: hand check (rows) against DeepSeek (columns)")
print(table(hand_is_event = d$cl_is_event, deepseek_is_event = d$ds_is_event))

message("\nReal event (an event in India): hand check (rows) against DeepSeek (columns)")
print(table(hand_real = d$cl_real, deepseek_real = d$ds_real))

message("\nOf the events DeepSeek calls real, the hand-checked type:")
print(d %>% filter(ds_real) %>%
        mutate(hand_type = case_when(cl_in_india == 0 ~ "outside India", cl_is_event == 0 ~ "not an event",
                                     TRUE ~ "real event in India")) %>%
        count(stratum, hand_type) %>% pivot_wider(names_from = hand_type, values_from = n, values_fill = 0))

#------------------------------------------------------------------------------#
####                         Actions, events both call real                 ####
#------------------------------------------------------------------------------#
both <- filter(d, cl_real, ds_real)
message("\nEvents both call real: ", nrow(both))

action_rows <- map_dfr(c(ACTIONS, "four_action"), function(a) {
  hand_flag <- if (a == "four_action") both$cl_four_action else both[[paste0("cl_", a)]] == 1
  ds_flag   <- both[[paste0("ds_", a)]] %in% c(TRUE, 1)
  tibble(measure = a, hand_n = sum(hand_flag), deepseek_n = sum(ds_flag),
         agree = mean(hand_flag == ds_flag),
         precision = sum(hand_flag & ds_flag) / max(sum(ds_flag), 1),   # deepseek flags that the hand check confirms
         recall    = sum(hand_flag & ds_flag) / max(sum(hand_flag), 1))
})
print(action_rows)

#------------------------------------------------------------------------------#
####                                  Counts                                ####
#------------------------------------------------------------------------------#
count_rows <- map_dfr(COUNTS, function(v) {
  h <- both[[paste0("cl_", v)]]
  s <- both[[paste0("ds_", v)]]
  pos <- h > 0
  tibble(measure = v, hand_total = sum(h), deepseek_total = sum(s),
         exact = mean(h == s), exact_when_positive = mean(h[pos] == s[pos]),
         n_positive = sum(pos), mean_abs_error = mean(abs(h - s)))
})
message("\nCounts, events both call real")
print(count_rows)

write_csv(bind_rows(action_rows, count_rows), paste0("../data/deepseek_validation_summary", tag, ".csv"))

#------------------------------------------------------------------------------#
####                            Disagreements                               ####
#------------------------------------------------------------------------------#
d %>%
  mutate(kind = case_when(
    cl_real & !ds_real ~ "hand: real event, DeepSeek: not",
    !cl_real & ds_real ~ "hand: not a real event, DeepSeek: real",
    cl_real & ds_real & cl_four_action != ds_four_action ~ "four-action differs",
    cl_real & ds_real & (cl_fatalities != ds_fatalities | cl_injuries != ds_injuries |
                                        cl_arrested != ds_arrested) ~ "counts differ",
    TRUE ~ NA_character_)) %>%
  filter(!is.na(kind)) %>%
  arrange(kind, id) %>%
  select(id, stratum, kind, event_text, cl_is_event, cl_in_india, ds_is_event, ds_in_india, cl_four_action, ds_four_action,
         cl_fatalities, ds_fatalities, cl_injuries, ds_injuries, cl_arrested, ds_arrested,
         cl_note, cl_count_note) %>%
  write_csv(paste0("../data/deepseek_validation_disagreements", tag, ".csv"))

message("\nDisagreements by kind:")
print(count(read_csv(paste0("../data/deepseek_validation_disagreements", tag, ".csv"), show_col_types = FALSE), kind))
