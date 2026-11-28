#------------------------------------------------------------------------------#
### SATP: assign each event to a 2001 district unit by matching place names
#
# Looks for district names in the text of every event, using the Irrigating Peace
# alias table (any spelling of a district name -> analysis unit_id). The text is split
# into words; every run of 1 to MAX_WORDS words, joined without spaces, is looked up as a
# name. Longer names win over shorter ones ("Imphal West" over "Imphal").
#
# A name shared by districts in several states (Aurangabad, Bijapur) is resolved when the
# text mentions exactly one of those states. Otherwise the event's SATP series decides: a
# Maoist-series event prefers Chhattisgarh over Karnataka, and so on (SERIES_STATES, in order
# of preference). A name that is still shared stays ambiguous. Names of up to SHORT_NAME_LEN
# letters ("Una", "Mau") only count when followed by "district".
#
# Each event gets a primary unit (the first "X district" mention, else the first resolved
# name), all units mentioned, and the state mentioned. Events that name only a state keep
# unit_id empty.
#
# The same matching runs on the hand-coded incidents (data/intermediate/datasets/satp_classification.csv) and is
# scored against two references: the unit of the hand-coded district name, and the unit
# containing the hand-coded coordinates.
#
# Inputs:  data/intermediate/scraped/satp_event_texts.rds         (code/process/split_multi_incident_entries.R)
#          data/intermediate/datasets/satp_classification.csv              (hand-coded, Maoist region 2005-2016)
#          code/sandbox/district_aliases.csv         (extra spellings and districts created after 2001)
#          Irrigating Peace intermediate data (paths below):
#            india_unit_aliases.rds, india_units_2001_gadm.rds
# Outputs: data/intermediate/scraped/satp_events_units.rds/.csv
#          data/intermediate/validation/district_match_errors.csv    (hand-coded incidents the matching gets wrong)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("dplyr", "tidyr", "purrr", "stringr", "readr", "sf")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

sf_use_s2(FALSE)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

ip_dir     <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted"
alias_file <- file.path(ip_dir, "india_unit_aliases.rds")
units_file <- file.path(ip_dir, "india_units_2001_gadm.rds")

extra_alias_file <- "sandbox/district_aliases.csv"
events_file      <- "../data/intermediate/scraped/satp_event_texts.rds"
labeled_file     <- "../data/intermediate/datasets/satp_classification.csv"
out_rds          <- "../data/intermediate/scraped/satp_events_units.rds"
out_csv          <- "../data/intermediate/scraped/satp_events_units.csv"
errors_file      <- "../data/intermediate/validation/district_match_errors.csv"

MAX_WORDS      <- 5                              # longest district name, in words
SHORT_NAME_LEN <- 3
SUFFIX_WORDS   <- c("district", "districts", "dist")

# states each SATP series covers, most conflict-affected first
SERIES_STATES <- list(
  "india-maoistinsurgency"    = c("Chhattisgarh", "Jharkhand", "Bihar", "Odisha", "Andhra Pradesh",
                                  "West Bengal", "Maharashtra", "Madhya Pradesh", "Uttar Pradesh"),
  "india-jammukashmir"        = "Jammu and Kashmir",
  "india-insurgencynortheast" = c("Assam", "Manipur", "Nagaland", "Meghalaya", "Tripura",
                                  "Arunachal Pradesh", "Mizoram"),
  "india-punjab"              = "Punjab")

# preferred states of an event, from its pipe-separated series tags
preferred_states_of <- function(series) {
  tags <- strsplit(series, "|", fixed = TRUE)[[1]]
  unique(unlist(SERIES_STATES[tags], use.names = FALSE))
}

#------------------------------------------------------------------------------#
####                              Name tables                               ####
#------------------------------------------------------------------------------#
# lower case ASCII letters only: "Y.S.R." and "ysr" match, as in the alias table
ascii <- function(x) coalesce(iconv(as.character(x), to = "ASCII//TRANSLIT"), "")
name_key <- function(x) str_replace_all(str_to_lower(ascii(x)), "[^a-z]", "")

# the text as lower case words separated by single spaces
words_of <- function(x) str_squish(str_replace_all(str_to_lower(ascii(x)), "[^a-z]+", " "))

aliases <- readRDS(alias_file)
unit_state <- distinct(aliases, unit_id, unit_state)

# state names as the alias table spells them, plus other spellings seen in SATP text
states <- sort(unique(aliases$unit_state))
state_of <- c(setNames(states, name_key(states)),
              orissa = "Odisha", uttaranchal = "Uttarakhand", pondicherry = "Puducherry",
              chattisgarh = "Chhattisgarh", kerela = "Kerala", delhi = "NCT Of Delhi",
              newdelhi = "NCT Of Delhi", telangana = "Andhra Pradesh",
              ladakh = "Jammu and Kashmir", jk = "Jammu and Kashmir",
              andamanandnicobar = "Andaman and Nicobar Islands")

# extra names from district_aliases.csv (alias, state, known_name): an alias means the unit
# of known_name within that state. It covers spellings the alias table lacks and districts
# created after 2001, which belong to the unit of their 2001 parent.
extra_aliases <- read_csv(extra_alias_file, show_col_types = FALSE) %>%
  transmute(alias_n = name_key(alias), unit_state = unname(state_of[name_key(state)]),
            target = name_key(known_name)) %>%
  inner_join(select(aliases, target = alias_n, unit_id, unit_state),
             by = c("target", "unit_state"), relationship = "many-to-many") %>%
  select(alias_n, unit_id, unit_state)

aliases <- bind_rows(select(aliases, alias_n, unit_id, unit_state), extra_aliases) %>%
  distinct()

# for each name: the units it can mean and the states of those units
alias_units  <- split(aliases$unit_id, aliases$alias_n)
alias_states <- split(aliases$unit_state, aliases$alias_n)

place_keys <- union(names(alias_units), names(state_of))

#------------------------------------------------------------------------------#
####                               Matching                                 ####
#------------------------------------------------------------------------------#
# the unit a name means, given the states mentioned in the text and the event's preferred
# states; NA if it stays ambiguous
resolve_name <- function(key, states_mentioned, preferred_states = character(0)) {
  units <- unique(alias_units[[key]])
  if (length(units) == 1) return(units)
  in_state <- unique(alias_units[[key]][alias_states[[key]] %in% states_mentioned])
  if (length(in_state) == 1) return(in_state)
  rank <- match(alias_states[[key]], preferred_states)
  if (all(is.na(rank))) return(NA_character_)
  best <- unique(alias_units[[key]][which(rank == min(rank, na.rm = TRUE))])
  if (length(best) == 1) best else NA_character_
}

# one row per text: primary unit, all units mentioned, first state mentioned
find_places <- function(text, series) {
  preferred_states <- preferred_states_of(series)
  words <- strsplit(words_of(text), " ", fixed = TRUE)[[1]]
  taken <- rep(FALSE, length(words))   # words already claimed by a longer name
  names_found <- list()
  states_found <- character(0)

  for (n in MAX_WORDS:1) {
    if (length(words) < n) next
    starts  <- seq_len(length(words) - n + 1)
    phrases <- vapply(starts, function(i) paste(words[i:(i + n - 1)], collapse = ""), "")
    for (i in starts[phrases %in% place_keys]) {
      if (any(taken[i:(i + n - 1)])) next
      key    <- phrases[i]
      suffix <- isTRUE(words[i + n] %in% SUFFIX_WORDS)
      is_name <- key %in% names(alias_units)
      if (is_name && nchar(key) <= SHORT_NAME_LEN && !suffix) next
      taken[i:(i + n - 1)] <- TRUE
      if (key %in% names(state_of)) states_found <- c(states_found, state_of[[key]])
      if (is_name) names_found[[length(names_found) + 1]] <- list(pos = i, key = key, suffix = suffix)
    }
  }

  states_found <- unique(states_found)
  unit  <- map_chr(names_found, ~ resolve_name(.x$key, states_found, preferred_states))
  pos   <- map_int(names_found, "pos")
  suffix <- map_lgl(names_found, "suffix")

  resolved <- which(!is.na(unit))
  resolved <- resolved[order(!suffix[resolved], pos[resolved])]   # "X district" mentions first
  primary  <- resolved[1]

  list(unit_id     = if (length(resolved)) unit[primary] else NA_character_,
       match_type  = if (!length(resolved)) NA_character_ else if (suffix[primary]) "district suffix" else "bare name",
       units_all   = if (length(resolved)) paste(unique(unit[resolved]), collapse = "|") else NA_character_,
       n_units     = length(unique(unit[resolved])),
       n_ambiguous = sum(is.na(unit)),
       state_mentioned = if (length(states_found)) states_found[1] else NA_character_)
}

match_texts <- function(texts, series) bind_rows(map2(texts, series, find_places))

#------------------------------------------------------------------------------#
####                  Evaluation on the hand-coded incidents                ####
#------------------------------------------------------------------------------#
# unit of a hand-coded (state, district) pair
unit_of_name <- function(state, district) {
  key <- name_key(district)
  if (!key %in% names(alias_units)) return(NA_character_)
  resolve_name(key, unname(state_of[name_key(state)]))
}

# the longitude and latitude columns of satp_classification.csv are swapped
labeled <- read_csv(labeled_file, show_col_types = FALSE) %>%
  filter(!is.na(incident_summary)) %>%
  mutate(row_id = row_number(), lat = longitude, lon = latitude)

units_gadm <- readRDS(units_file)

points <- labeled %>%
  filter(between(lat, 6, 38), between(lon, 68, 98)) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326) %>%
  st_join(select(units_gadm, truth_coords = unit_id)) %>%
  st_drop_geometry() %>%
  distinct(row_id, .keep_all = TRUE) %>%
  select(row_id, truth_coords)

message("Matching the hand-coded incidents ...")
eval_df <- labeled %>%
  mutate(truth_name = map2_chr(state, district, unit_of_name)) %>%
  left_join(points, by = "row_id") %>%
  bind_cols(match_texts(.$incident_summary, "india-maoistinsurgency"))

# coverage: a unit was found; precision: correct among those found; accuracy: correct among all
score <- function(truth) {
  d       <- eval_df[!is.na(truth), ]
  found   <- !is.na(d$unit_id)
  correct <- found & d$unit_id == truth[!is.na(truth)]
  tibble(n = nrow(d), coverage = mean(found), precision = mean(correct[found]), accuracy = mean(correct))
}

message("\nHand-coded incidents: ", nrow(eval_df))
message("Unit of the hand-coded district name found for ", sum(!is.na(eval_df$truth_name)),
        "; unit of the coordinates found for ", sum(!is.na(eval_df$truth_coords)))
message("The two references agree on ",
        round(mean(eval_df$truth_name == eval_df$truth_coords, na.rm = TRUE), 3),
        " of incidents that have both")
print(bind_rows(`vs district name` = score(eval_df$truth_name),
                `vs coordinates`   = score(eval_df$truth_coords), .id = "reference"))

message("\nHand-coded districts with no unit in the alias table (add to data/intermediate/places/district_aliases.csv):")
print(eval_df %>% filter(is.na(truth_name)) %>% count(state, district, sort = TRUE) %>% head(25))

eval_df %>%
  filter(!is.na(truth_name), is.na(unit_id) | unit_id != truth_name) %>%
  select(row_id, state, district, truth_name, truth_coords, pred_unit = unit_id,
         match_type, n_units, n_ambiguous, incident_summary) %>%
  write_csv(errors_file)

#------------------------------------------------------------------------------#
####                          Apply to the events                           ####
#------------------------------------------------------------------------------#
events <- readRDS(events_file)

message("\nMatching ", nrow(events), " events ...")
events_units <- events %>%
  select(event_uid, incident_uid, event_date) %>%
  bind_cols(match_texts(events$event_text, events$series)) %>%
  left_join(rename(unit_state, unit_state_name = unit_state), by = "unit_id")

message("Events with a unit: ", round(mean(!is.na(events_units$unit_id)), 3))
message("Events with only a state: ",
        round(mean(is.na(events_units$unit_id) & !is.na(events_units$state_mentioned)), 3))
message("Events with an ambiguous name: ", round(mean(events_units$n_ambiguous > 0), 3))

saveRDS(events_units, out_rds)
write_csv(events_units, out_csv)
