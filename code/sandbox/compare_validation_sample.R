#------------------------------------------------------------------------------#
### SATP: compare the hand check of the validation sample with the pipeline
#
# The 300 events drawn by draw_validation_sample.R were coded from the text alone, without the
# pipeline's answers (data/validation_sample_claude_batch1-6.csv): is it an event in India,
# which actions, which district. This script joins that coding with the pipeline's and reports,
# by series stratum:
#   - real events: the event is a specific act in India (not a statement, threat, investigation
#     news, accident or an act abroad)
#   - four-action events: a real event with armed assault, bombing, infrastructure or abduction
#   - district: the pipeline's unit against the unit of the district named or inferred from the text
#
# Inputs:  data/validation_sample_claude_batch1.csv ... batch6.csv   (hand check)
#          data/validation_sample_claude_counts.csv                  (people killed, injured and arrested per event)
#          data/validation_sample_blind.csv, validation_sample_pipeline.rds   (draw_validation_sample.R)
#          india_unit_aliases.rds (Irrigating Peace restricted folder), sandbox/district_aliases.csv
# Outputs: data/validation_sample_claude_coded.csv   the combined hand check
#          data/validation_sample_comparison.csv     one row per event: text, both codings, disagreements first
#          data/validation_summary.csv               accuracy by stratum
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

library(tidyverse)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

ip_dir <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted"

name_key <- function(x) str_replace_all(str_to_lower(x), "[^a-z]", "")

#------------------------------------------------------------------------------#
####                          Combine the hand check                        ####
#------------------------------------------------------------------------------#
mine <- map_dfr(1:6, ~ read_csv(sprintf("../data/validation_sample_claude_batch%d.csv", .x),
                                col_types = cols(.default = "c"))) %>%
  mutate(across(c(id, is_event, in_india, armed_assault, bombing, infrastructure, abduction,
                  arrest, seizure, surrender, n_incidents), as.integer))
stopifnot(identical(sort(mine$id), 1:300))
counts <- read_csv("../data/validation_sample_claude_counts.csv", show_col_types = FALSE)
mine <- left_join(mine, counts, by = "id")
write_csv(arrange(mine, id), "../data/validation_sample_claude_coded.csv")

#------------------------------------------------------------------------------#
####                         Districts to units                             ####
#------------------------------------------------------------------------------#
aliases <- readRDS(file.path(ip_dir, "india_unit_aliases.rds")) %>% select(alias_n, unit_id, unit_state)
extra <- read_csv("sandbox/district_aliases.csv", show_col_types = FALSE) %>%
  transmute(alias_n = name_key(alias), state_key = name_key(state), target = name_key(known_name)) %>%
  inner_join(aliases, by = c("target" = "alias_n"), relationship = "many-to-many") %>%
  filter(name_key(unit_state) == state_key) %>%
  select(alias_n, unit_id, unit_state)
lookup <- bind_rows(aliases, extra) %>% distinct()

# districts created after 2001 map to their 2001 parent; the alias tables do not hold these
parent_2001 <- tribble(
  ~district, ~state, ~parent,
  "Bandipora", "Jammu and Kashmir", "Baramulla", "Ganderbal", "Jammu and Kashmir", "Srinagar",
  "Samba", "Jammu and Kashmir", "Jammu", "Reasi", "Jammu and Kashmir", "Udhampur",
  "Shopian", "Jammu and Kashmir", "Pulwama", "Kulgam", "Jammu and Kashmir", "Anantnag",
  "Pathankot", "Punjab", "Gurdaspur", "Mohali", "Punjab", "Rupnagar", "Fazilka", "Punjab", "Firozpur",
  "Dima Hasao", "Assam", "North Cachar Hills", "Chirang", "Assam", "Kokrajhar", "Udalguri", "Assam", "Darrang",
  "Narayanpur", "Chhattisgarh", "Bastar", "Sukma", "Chhattisgarh", "Dantewada", "Bijapur", "Chhattisgarh", "Dantewada",
  "Mohla-Manpur", "Chhattisgarh", "Rajnandgaon", "Bhadradri Kothagudem", "Telangana", "Khammam",
  "Jayashankar Bhupalpally", "Telangana", "Warangal", "Tengnoupal", "Manipur", "Chandel",
  "Kangpokpi", "Manipur", "Senapati")

state_in_alias <- c("Telangana" = "Andhra Pradesh", "Odisha" = "Orissa")

to_unit <- function(district, state) {
  pmap_chr(list(district, state), function(d, s) {
    if (is.na(d) || d == "") return(NA_character_)
    p <- parent_2001$parent[parent_2001$district == d & parent_2001$state == s]
    key <- name_key(if (length(p)) p[1] else d)
    cand <- lookup %>% filter(alias_n == key)
    if (n_distinct(cand$unit_id) > 1) {
      st <- name_key(coalesce(unname(state_in_alias[s]), s))
      cand <- filter(cand, name_key(unit_state) == st)
    }
    if (n_distinct(cand$unit_id) == 1) cand$unit_id[1] else NA_character_
  })
}

#------------------------------------------------------------------------------#
####                         Join and compare                               ####
#------------------------------------------------------------------------------#
pipeline <- readRDS("../data/validation_sample_pipeline.rds") %>%
  rename(pipeline_unit = unit_id, pipeline_state = admin1, pipeline_district = admin2)
blind <- read_csv("../data/validation_sample_blind.csv", show_col_types = FALSE)

d <- mine %>%
  rename(claude_state = state, claude_district = district) %>%
  left_join(select(pipeline, id, stratum, year, pipeline_unit, pipeline_state, pipeline_district,
                   pipeline_type = sub_event_type, event_confidence), by = "id") %>%
  left_join(select(blind, id, event_text), by = "id") %>%
  mutate(real_event  = is_event == 1 & in_india == 1,
         four_action = real_event & (armed_assault + bombing + infrastructure + abduction) > 0,
         claude_unit = to_unit(claude_district, claude_state),
         district_given = !is.na(claude_district) & claude_district != "" & district_basis %in% c("named", "inferred"),
         unit_known = district_given & !is.na(claude_unit),
         unit_match = if_else(unit_known & !is.na(pipeline_unit), pipeline_unit == claude_unit, NA))

message("District names I could not map to a unit (check these by hand):")
print(d %>% filter(district_given, is.na(claude_unit)) %>% distinct(claude_state, claude_district), n = Inf)

#------------------------------------------------------------------------------#
####                              Summary                                   ####
#------------------------------------------------------------------------------#
by_stratum <- function(df) df %>% summarise(
  n                   = n(),
  share_real_event    = mean(real_event),
  share_not_event     = mean(is_event == 0),
  share_outside_india = mean(in_india == 0),
  share_real_not_four = mean(real_event & !four_action),   # real event whose actions are only arrest, seizure, surrender or none
  share_four_action   = mean(four_action),                 # the share of the analysis set that is a real four-action event
  share_pipeline_unit = mean(!is.na(pipeline_unit)),
  n_unit_checked      = sum(!is.na(unit_match) & real_event),
  share_unit_correct  = mean(unit_match[real_event], na.rm = TRUE))

summary_tab <- bind_rows(d %>% group_by(stratum) %>% by_stratum(),
                         d %>% by_stratum() %>% mutate(stratum = "All (unweighted)"))
print(summary_tab, width = Inf)
write_csv(summary_tab, "../data/validation_summary.csv")

message("\nWhy events in the analysis set are not real four-action events:")
print(d %>% filter(!four_action) %>%
        mutate(reason = case_when(in_india == 0 ~ "outside India", is_event == 0 ~ "not an event",
                                  TRUE ~ "real event, no four-action act")) %>%
        count(stratum, reason) %>% pivot_wider(names_from = reason, values_from = n, values_fill = 0), width = Inf)

#------------------------------------------------------------------------------#
####                      Event-level file for checking                     ####
#------------------------------------------------------------------------------#
d %>%
  mutate(disagree = !four_action | (!is.na(unit_match) & !unit_match) |
                    (real_event & is.na(pipeline_unit) & unit_known),
         user_check = "") %>%
  arrange(desc(disagree), stratum, id) %>%
  select(id, stratum, year, disagree, real_event, four_action, is_event, in_india,
         armed_assault, bombing, infrastructure, abduction, arrest, seizure, surrender,
         fatalities, injuries, arrested, count_note,
         claude_state, claude_district, district_basis, district_certainty, claude_note = note,
         pipeline_state, pipeline_district, pipeline_unit, claude_unit, unit_match, pipeline_type,
         event_confidence, event_text, user_check) %>%
  write_csv("../data/validation_sample_comparison.csv")
