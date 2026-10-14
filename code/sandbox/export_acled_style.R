#------------------------------------------------------------------------------#
### SATP: export the events in the column layout of ACLED
#
# One row per event, with ACLED's columns in ACLED's order (event_id_cnty, event_date,
# year, disorder_type, event_type, sub_event_type, actor1 ... notes, fatalities, tags)
# followed by SATP columns (unit_id, the seven action flags, the DistilBERT confidence).
#
# Where each column comes from:
#   event_type, sub_event_type   ACLED's names. The action flags of the DistilBERT model
#                                (probability above 0.5) give the first match in this order:
#                                abduction, bombing, armed assault, infrastructure, arrest,
#                                seizure, surrender. Keyword rules on the event text then split
#                                bombing into grenade, suicide bomb, shelling, air or drone
#                                strike and remote explosive/landmine/IED, and armed assault
#                                into armed clash (security forces or an exchange of fire, no
#                                civilians) and attack (civilians). The armed assault split
#                                agrees with DeepSeek's coding on 83% of assaults.
#   admin1, admin2, unit_id      state and 2001 district unit from match_event_districts.R
#   actor1 ... civilian_targeting, fatalities
#                                DeepSeek coding (code_events_deepseek.R); empty for events
#                                DeepSeek has not coded
#   is_event_deepseek, in_india_deepseek
#                                DeepSeek's answers on whether the text reports a specific act and
#                                whether it took place in India; empty for events DeepSeek has not coded
#   event_confidence            the highest of the seven DistilBERT probabilities; low values
#                                mark texts that are probably not violent events
#   latitude, longitude, admin3, location   empty: SATP events are located at district level
#
# Inputs:  scraped_data/satp_event_texts.rds, satp_events_actions_local.csv,
#          satp_events_units.rds, satp_events_coded.rds
#          Irrigating Peace india_unit_aliases.rds (unit names)
# Outputs: satp_events_acled_style.csv and .rds in the Irrigating Peace restricted
#          intermediate folder (out_dir below)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("dplyr", "tidyr", "stringr", "readr", "lubridate")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

ip_dir  <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted"
out_dir <- ip_dir

THRESHOLD <- 0.5

#------------------------------------------------------------------------------#
####                                Inputs                                  ####
#------------------------------------------------------------------------------#
events <- readRDS("../scraped_data/satp_event_texts.rds") %>%
  select(event_uid, incident_uid, event_part, event_date, event_text, split_status, series)

actions <- read_csv("../scraped_data/satp_events_actions_local.csv", show_col_types = FALSE) %>%
  transmute(event_uid,
            armed_assault  = as.integer(p_armed_assault > THRESHOLD),
            abduction      = as.integer(p_abduction > THRESHOLD),
            bombing        = as.integer(p_bombing > THRESHOLD),
            infrastructure = as.integer(p_infrastructure > THRESHOLD),
            surrender      = as.integer(p_surrender > THRESHOLD),
            arrest         = as.integer(p_arrest > THRESHOLD),
            seizure        = as.integer(p_seizure > THRESHOLD),
            event_confidence = pmax(p_armed_assault, p_abduction, p_bombing, p_infrastructure,
                                    p_surrender, p_arrest, p_seizure))

units <- readRDS("../scraped_data/satp_events_units.rds") %>%
  select(event_uid, unit_id, unit_state_name, state_mentioned)

unit_names <- readRDS(file.path(ip_dir, "india_unit_aliases.rds")) %>%
  filter(source == "unit name") %>%
  distinct(unit_id, unit_name = alias)

# DeepSeek coding, matched to events by text. The analysis-set pass (sandbox/make_analysis_set_input.R,
# satp_events_coded_analysis*.rds) comes before the earlier run of satp_events_coded.rds; with passes of
# both models the file of deepseek-flash is read first
deepseek_files <- c(sort(Sys.glob("../scraped_data/satp_events_coded_analysis*.rds")),
                    "../scraped_data/satp_events_coded.rds")
deepseek_all <- bind_rows(lapply(deepseek_files, readRDS))
if (!"in_india" %in% names(deepseek_all)) deepseek_all$in_india <- NA
deepseek <- deepseek_all %>%
  filter(!is.na(is_event)) %>%
  distinct(event_text, .keep_all = TRUE) %>%
  select(event_text, is_event_deepseek = is_event, in_india_deepseek = in_india, actor1 = actor, inter1, inter2,
         civilian_targeting, fatalities)

#------------------------------------------------------------------------------#
####                               Assemble                                 ####
#------------------------------------------------------------------------------#
armed_words    <- paste0("encounter|gun ?battle|gunfight|exchange of fire|exchanged fire|clashed|ambush|",
                         "security force|security personnel|\\bsfs?\\b|jawan|crpf|\\bbsf\\b|army|",
                         "police (party|personnel|station|post|camp|team|patrol)|policem[ae]n|trooper|constable|",
                         "(militant|terrorist|maoist|naxalite|extremist|cadre)s? (was|were|is|are) (killed|shot dead)")
civilian_words <- "civilian|villager|\\bwoman\\b|women|teacher|farmer|labourer|activist|sarpanch|councillor|informer"

acled_style <- events %>%
  left_join(actions, by = "event_uid") %>%
  left_join(units, by = "event_uid") %>%
  left_join(unit_names, by = "unit_id") %>%
  left_join(deepseek, by = "event_text") %>%
  mutate(
    text = str_to_lower(event_text),
    sub_event_type = case_when(
      abduction == 1                                         ~ "Abduction/forced disappearance",
      bombing == 1 & str_detect(text, "suicide|fidayeen|human bomb") ~ "Suicide bomb",
      bombing == 1 & str_detect(text, "grenade")             ~ "Grenade",
      bombing == 1 & str_detect(text, "shell(ing|ed)|mortar|artillery|rocket|missile|cease-?fire violation") ~
        "Shelling/artillery/missile attack",
      bombing == 1 & str_detect(text, "air ?strike|drone")   ~ "Air/drone strike",
      bombing == 1                                           ~ "Remote explosive/landmine/IED",
      armed_assault == 1 & str_detect(text, armed_words) & !str_detect(text, "civilian|villager") ~ "Armed clash",
      armed_assault == 1 & str_detect(text, civilian_words)  ~ "Attack",
      armed_assault == 1                                     ~ "Armed clash",
      infrastructure == 1                                    ~ "Looting/property destruction",
      arrest == 1                                            ~ "Arrests",
      seizure == 1                                           ~ "Disrupted weapons use",
      surrender == 1                                         ~ "Change to group/activity"),
    event_type = case_when(
      sub_event_type == "Armed clash"                                        ~ "Battles",
      sub_event_type %in% c("Attack", "Abduction/forced disappearance")      ~ "Violence against civilians",
      sub_event_type %in% c("Suicide bomb", "Grenade", "Shelling/artillery/missile attack",
                            "Air/drone strike", "Remote explosive/landmine/IED") ~ "Explosions/Remote violence",
      !is.na(sub_event_type)                                                 ~ "Strategic developments"),
    disorder_type = case_when(
      event_type == "Strategic developments" ~ "Strategic developments",
      !is.na(event_type)                     ~ "Political violence"),
    admin1 = coalesce(unit_state_name, state_mentioned),
    any_analysis_action = as.integer(armed_assault + bombing + infrastructure + abduction > 0)
  ) %>%
  transmute(
    event_id_cnty = sprintf("SATP%06d", event_uid),
    event_date, year = year(event_date), time_precision = NA_integer_,
    disorder_type, event_type, sub_event_type,
    actor1, assoc_actor_1 = NA_character_, inter1,
    actor2 = NA_character_, assoc_actor_2 = NA_character_, inter2,
    interaction = if_else(!is.na(inter1) & !is.na(inter2), paste(inter1, inter2, sep = "-"), NA_character_),
    civilian_targeting = if_else(civilian_targeting %in% TRUE, "Civilian targeting", NA_character_),
    iso = 356L, region = "Southern Asia", country = "India",
    admin1, admin2 = unit_name, admin3 = NA_character_, location = NA_character_,
    latitude = NA_real_, longitude = NA_real_,
    geo_precision = case_when(!is.na(unit_id) ~ 2L, !is.na(admin1) ~ 3L),
    source = "SATP", source_scale = NA_character_,
    notes = event_text, fatalities, tags = series, timestamp = NA_integer_,
    unit_id, incident_uid, event_part, split_status,
    armed_assault, abduction, bombing, infrastructure, surrender, arrest, seizure,
    any_analysis_action, event_confidence, is_event_deepseek, in_india_deepseek)

message("Events: ", nrow(acled_style))
message("With a district unit: ", round(mean(!is.na(acled_style$unit_id)), 3),
        "; with a state: ", round(mean(!is.na(acled_style$admin1)), 3))
message("With an event type: ", round(mean(!is.na(acled_style$event_type)), 3),
        "; coded by DeepSeek: ", sum(!is.na(acled_style$is_event_deepseek)))
print(count(acled_style, event_type, sub_event_type, sort = TRUE))

saveRDS(acled_style, file.path(out_dir, "satp_events_acled_style.rds"))
write_csv(acled_style, file.path(out_dir, "satp_events_acled_style.csv"))
