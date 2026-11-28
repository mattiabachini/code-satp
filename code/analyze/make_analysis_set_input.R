#------------------------------------------------------------------------------#
### SATP: input file for the DeepSeek pass on the analysis set
#
# The analysis set is the events of 2000 to 2024 with a DistilBERT confidence above 0.7, at least one of
# the four analysis actions (armed assault, bombing, infrastructure, abduction) and a unit. Their texts
# are the event texts of split_multi_incident_entries.R, matched to the Irrigating Peace events by
# incident_uid and event_part.
#
# Run the pass with
#   SATP_TEST_FILE=../data/final/analysis_set_events.rds SATP_RUN_NAME=analysis SATP_N_LIMIT=50000 \
#   SATP_MODEL=deepseek-chat Rscript code_events_deepseek.R
# (from the code folder; SATP_MODEL defaults to deepseek-flash). The answers go to
# data/intermediate/coded/coding_cache_analysis[_model].rds and satp_events_coded_analysis[_model].rds/.csv.
#
# Input:   data/intermediate/scraped/satp_event_texts.rds
#          Irrigating Peace data/intermediate/restricted/satp_events_conf07.rds
# Output:  data/final/analysis_set_events.rds   columns: incident_uid, event_part, event_text, year, unit_id
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("dplyr")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

conf07_file <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/satp_events_conf07.rds"

texts <- readRDS("../data/intermediate/scraped/satp_event_texts.rds") %>%
  select(incident_uid, event_part, event_text)

analysis_set <- readRDS(conf07_file) %>%
  filter(year %in% 2000:2024, any_analysis_action %in% 1, !is.na(unit_id)) %>%
  select(incident_uid, event_part, year, unit_id) %>%
  inner_join(texts, by = c("incident_uid", "event_part")) %>%
  select(incident_uid, event_part, event_text, year, unit_id)

saveRDS(analysis_set, "../data/final/analysis_set_events.rds")

message("Analysis set events: ", nrow(analysis_set), "; distinct texts: ", n_distinct(analysis_set$event_text),
        "; mean text length ", round(mean(nchar(analysis_set$event_text))), " characters")
