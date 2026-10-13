#------------------------------------------------------------------------------#
### SATP: check DeepSeek's casualty and arrest counts against the hand-coded counts
#
# The variables of Fetzer (2020) beyond the event count are casualties by victim group
# (civilians, security forces, militants), arrests and events without casualties.
# code_events_deepseek.R codes them; this script tests that coding on hand-coded incidents.
#
# Two steps, by which file exists:
#   1. No coded test file: writes a sample of hand-coded incidents (200 with at least one
#      count above zero, 100 with none) to scraped_data/count_test_events.rds. Then run
#        SATP_TEST_FILE=../scraped_data/count_test_events.rds Rscript code_events_deepseek.R
#      from the code folder.
#   2. Coded test file present: compares DeepSeek's counts with the hand-coded ones.
#
# Hand-coded groups map to Fetzer's three: civilians = civilians + government officials,
# security = security forces, militants = Maoists + other armed groups.
#
# Inputs:  data/deaths.csv, data/injuries.csv, data/arrests.csv   (hand-coded, Maoist region 2005-2016)
#          scraped_data/satp_events_coded_test.rds                (code/code_events_deepseek.R, step 2)
# Outputs: scraped_data/count_test_events.rds                     (step 1)
#          data/fetzer_counts_disagreements.csv                   (step 2: events where a group count differs)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

library(tidyverse)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

test_events_file <- "../scraped_data/count_test_events.rds"
coded_file       <- "../scraped_data/satp_events_coded_test.rds"

#------------------------------------------------------------------------------#
####                        Hand-coded counts                               ####
#------------------------------------------------------------------------------#
first_per_text <- function(file) read_csv(file, show_col_types = FALSE) %>%
  filter(!is.na(incident_summary)) %>% distinct(incident_summary, .keep_all = TRUE)

hand <- first_per_text("../data/deaths.csv") %>%
  inner_join(first_per_text("../data/injuries.csv"), by = "incident_summary") %>%
  inner_join(first_per_text("../data/arrests.csv"), by = "incident_summary") %>%
  transmute(
    event_text = incident_summary,
    fatalities_hand        = total_fatalities,
    injuries_hand          = total_injuries,
    killed_civilians_hand  = civilian_fatalities + govt_official_fatalities,
    killed_security_hand   = security_fatalities,
    killed_militants_hand  = maoist_fatalities + other_armed_grp_fatalities,
    injured_civilians_hand = civilian_injuries + govt_official_injuries,
    injured_security_hand  = security_injuries,
    injured_militants_hand = maoist_injuries + non_maoist_armed_group_injuries,
    arrested_hand          = total_arrests) %>%
  drop_na()

#------------------------------------------------------------------------------#
####                      Step 1: write the test events                     ####
#------------------------------------------------------------------------------#
if (!file.exists(coded_file)) {
  has_count <- hand$fatalities_hand + hand$injuries_hand + hand$arrested_hand > 0
  set.seed(7)
  bind_rows(slice_sample(filter(hand, has_count), n = 200),
            slice_sample(filter(hand, !has_count), n = 100)) %>%
    saveRDS(test_events_file)
  message("Wrote 300 test events to ", test_events_file,
          ". Now run code_events_deepseek.R with SATP_TEST_FILE set to that file.")
} else {

#------------------------------------------------------------------------------#
####                         Step 2: compare the counts                     ####
#------------------------------------------------------------------------------#
vars <- c("fatalities", "injuries", "killed_civilians", "killed_security", "killed_militants",
          "injured_civilians", "injured_security", "injured_militants", "arrested")

coded <- readRDS(coded_file) %>% filter(!is.na(is_event))
message("Test events coded: ", nrow(coded))

metrics <- map_dfr(vars, function(v) {
  truth <- coded[[paste0(v, "_hand")]]
  pred  <- coded[[v]]
  tibble(variable       = v,
         hand_total     = sum(truth),
         deepseek_total = sum(pred),
         exact_match    = mean(pred == truth),            # same count
         same_zero      = mean((pred > 0) == (truth > 0)), # both zero or both positive
         mean_abs_error = mean(abs(pred - truth)))
})
print(metrics, width = Inf)

coded %>%
  filter(killed_civilians != killed_civilians_hand | killed_security != killed_security_hand |
           killed_militants != killed_militants_hand | arrested != arrested_hand) %>%
  select(event_text, ends_with("_hand"), all_of(vars)) %>%
  write_csv("../data/fetzer_counts_disagreements.csv")
}
