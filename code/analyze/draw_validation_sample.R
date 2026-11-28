#------------------------------------------------------------------------------#
### SATP: random validation sample of the events used in the analysis
#
# Draws 300 events from the analysis set (DistilBERT confidence above 0.7 and at least one of
# the four analysis actions: armed assault, bombing, infrastructure, abduction), stratified by
# SATP series. Events without a district stay in the draw, so the sample also measures how
# often the district is missing.
#
# Strata, by the regional series tag of the entry (first match in this order):
#   Jammu and Kashmir 75, Northeast 75, Maoist 60, Punjab 30, other (india series only) 60.
#
# Inputs:  satp_events_acled_style.rds       (Irrigating Peace restricted folder; sandbox/export_acled_style.R)
#          data/intermediate/scraped/satp_event_texts.rds (series tags)
# Outputs: data/intermediate/validation/validation_sample_blind.csv      id, stratum, year, event_text (for coding without the pipeline's answers)
#          data/intermediate/validation/validation_sample_pipeline.rds   id and the pipeline's coding of the same events
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

library(tidyverse)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

ip_dir <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted"

series <- readRDS("../data/intermediate/scraped/satp_event_texts.rds") %>%
  distinct(incident_uid, event_part, series)

events <- readRDS(file.path(ip_dir, "satp_events_acled_style.rds")) %>%
  filter(event_confidence > 0.7, any_analysis_action %in% TRUE) %>%
  left_join(series, by = c("incident_uid", "event_part")) %>%
  mutate(stratum = case_when(
    str_detect(series, "jammukashmir")           ~ "Jammu and Kashmir",
    str_detect(series, "insurgencynortheast")    ~ "Northeast",
    str_detect(series, "maoistinsurgency")       ~ "Maoist",
    str_detect(series, "punjab")                 ~ "Punjab",
    TRUE                                         ~ "Other"))

quota <- c("Jammu and Kashmir" = 75, "Northeast" = 75, "Maoist" = 60, "Punjab" = 30, "Other" = 60)

message("Analysis-set events by stratum:")
print(count(events, stratum))

set.seed(2026)
sample_events <- events %>%
  group_by(stratum) %>%
  group_modify(~ slice_sample(.x, n = min(quota[[.y$stratum]], nrow(.x)))) %>%
  ungroup() %>%
  slice_sample(prop = 1) %>%
  mutate(id = row_number(), year = year(as.Date(event_date)))

sample_events %>%
  select(id, stratum, year, event_text = notes) %>%
  write_csv("../data/intermediate/validation/validation_sample_blind.csv")

sample_events %>%
  select(id, stratum, year, unit_id, admin1, admin2, event_type, sub_event_type, event_confidence,
         armed_assault, bombing, infrastructure, abduction, arrest, seizure, surrender, notes) %>%
  saveRDS("../data/intermediate/validation/validation_sample_pipeline.rds")

message("Wrote ", nrow(sample_events), " events")
print(count(sample_events, stratum))
