#------------------------------------------------------------------------------#
### SATP: pilot of DeepSeek place extraction and SHRID geocoding
#
# Asks DeepSeek to pull out, for each event text, the places where the act took
# place (village, town, tehsil/block, police station, forest, camp ...), then
# resolves each name to a SHRID census place inside the event's 2001 district unit
# (exact, then Jaro-Winkler) and gives it the centroid of that place. A simple
# rule-based exact match (SHRID names found in the text) is computed on the same
# events for comparison.
#
# Sample: 500 analysis-set events (confidence > 0.7, an analysis action, a unit),
#         150 Maoist, 150 J&K, 150 Northeast, 50 Punjab/other.
#
# Inputs:  Irrigating Peace satp_events_conf07.rds and india_shrid_unit_key.rds
#          data/shrid_loc_names.csv, data/shrid_centroids.rds (SHRUG shrid2 polygons)
#          environment variable DEEPSEEK_API_KEY
#          environment variable SATP_PILOT_N (optional, default 500)
# Outputs: scraped_data/place_cache_pilot.rds      (DeepSeek answers, reruns are free)
#          scraped_data/place_pilot_events.csv     (one row per event)
#          scraped_data/place_pilot_places.csv     (one row per extracted place)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("dplyr", "tidyr", "purrr", "stringr", "readr", "rlang", "httr2", "jsonlite", "stringdist")
loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

ip_dir      <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/"
conf07_file <- paste0(ip_dir, "satp_events_conf07.rds")
key_file    <- paste0(ip_dir, "india_shrid_unit_key.rds")
cache_file  <- "../scraped_data/place_cache_pilot.rds"

MODEL          <- "deepseek-flash"
PROMPT_VERSION <- 1L
API_URL        <- "https://api.deepseek.com/chat/completions"
N_PARALLEL     <- 24
N_PILOT        <- as.integer(Sys.getenv("SATP_PILOT_N", unset = "500"))
JW_MAX         <- 0.10   # Jaro-Winkler distance for a fuzzy match
PRICE_OFFPEAK  <- c(hit = 0.003, miss = 0.15, out = 0.6)

#------------------------------------------------------------------------------#
####                        1. Sample and gazetteer                         ####
#------------------------------------------------------------------------------#
series_of <- function(tags) case_when(
  str_detect(tags, "maoist")       ~ "Maoist",
  str_detect(tags, "jammukashmir") ~ "J&K",
  str_detect(tags, "northeast")    ~ "Northeast",
  TRUE                             ~ "Punjab/other")

analysis <- readRDS(conf07_file) %>%
  filter(event_confidence > 0.7, any_analysis_action == 1, !is.na(unit_id)) %>%
  mutate(series = series_of(tags))

quota <- round(N_PILOT * c(Maoist = 0.3, `J&K` = 0.3, Northeast = 0.3, `Punjab/other` = 0.1))
set.seed(20261119)
sample_ev <- analysis %>%
  group_by(series) %>%
  group_modify(~ slice_sample(.x, n = min(quota[[.y$series]], nrow(.x)))) %>%
  ungroup() %>%
  mutate(event_key = map_chr(notes, hash))

norm <- function(x) str_squish(str_replace_all(str_to_lower(x), "[^a-z ]", " "))
nospace <- function(x) str_remove_all(norm(x), " ")

gaz <- read_csv("../data/shrid_loc_names.csv", show_col_types = FALSE) %>%
  inner_join(distinct(readRDS(key_file), shrid2, unit_id), by = "shrid2") %>%
  left_join(readRDS("../data/shrid_centroids.rds"), by = "shrid2")

# place level: village or town names; sub-district level: sub-district names (centroid = mean of its places)
gaz_place <- gaz %>%
  filter(!is.na(place_name), nchar(nospace(place_name)) >= 3) %>%
  transmute(unit_id, shrid2, gaz_name = place_name, key = nospace(place_name),
            kind = if_else(!is.na(town_name), "town", "village"), lon, lat)
gaz_sub <- gaz %>%
  filter(!is.na(subdistrict_name)) %>%
  group_by(unit_id, gaz_name = subdistrict_name) %>%
  summarise(lon = mean(lon, na.rm = TRUE), lat = mean(lat, na.rm = TRUE), .groups = "drop") %>%
  mutate(key = nospace(gaz_name), kind = "subdistrict", shrid2 = NA_character_)
# district names inside each unit, so the model's district names never count as places
gaz_dist <- gaz %>% distinct(unit_id, key = nospace(district_name))

message("Pilot events: ", nrow(sample_ev)); print(count(sample_ev, series))

#------------------------------------------------------------------------------#
####                        2. Ask DeepSeek                                 ####
#------------------------------------------------------------------------------#
SYSTEM_PROMPT <- paste(
  "You read one news summary about political violence in India and list the places it names. Answer in json.",
  "",
  "places: every named place in the text, in the order they appear, each as an object with",
  "- name: the place name exactly as written, without words like village, district, police station or area (\"Faraspal village\" -> \"Faraspal\"; \"Kalimela Police Station\" -> \"Kalimela\").",
  "- type: one of \"village\", \"town\" (a town or city), \"locality\" (a neighbourhood, mohalla, para or tola), \"subdistrict\" (tehsil, taluk, block, mandal, sub-division or circle), \"police_station\" (a police station or its limits), \"district\", \"state\", \"forest\" (forest, jungle, hill or valley), \"camp\" (a security camp or post), \"road\" (a road, highway or railway section), \"water\" (river or lake), \"other\".",
  "- role: \"site\" if the violent act or operation took place there; \"near\" if the text says it took place near it, between it and another place, or in its area or limits; \"other\" if it is named for another reason (where victims came from or were taken, a headquarters or office, a court, an earlier event, a statement).",
  "- within: the name of the larger place the text says it belongs to (the village's police station, block or district), or null.",
  "",
  "precision: the most specific level at which the text locates the act itself: \"village\" (a village, town, locality, camp or a point such as a road section), \"subdistrict\" (only a tehsil, block or police station area), \"district\", \"state\", or \"none\".",
  "",
  "Do not invent places that are not written in the text. People's names, group names and names of operations are not places.",
  "",
  "Answer like this: {\"places\": [{\"name\": \"Faraspal\", \"type\": \"village\", \"role\": \"site\", \"within\": \"Kirandul\"}, {\"name\": \"Kirandul\", \"type\": \"police_station\", \"role\": \"near\", \"within\": \"Dantewada\"}, {\"name\": \"Dantewada\", \"type\": \"district\", \"role\": \"near\", \"within\": null}], \"precision\": \"village\"}",
  sep = "\n")

build_request <- function(text) {
  request(API_URL) %>%
    req_auth_bearer_token(Sys.getenv("DEEPSEEK_API_KEY")) %>%
    req_body_json(list(model = MODEL, temperature = 0,
                       response_format = list(type = "json_object"),
                       thinking = list(type = "disabled"),
                       messages = list(list(role = "system", content = SYSTEM_PROMPT),
                                       list(role = "user",   content = text)))) %>%
    req_retry(max_tries = 4)
}

read_answer <- function(resp) {
  if (!inherits(resp, "httr2_response")) return(NULL)
  tryCatch(fromJSON(resp_body_json(resp)$choices[[1]]$message$content, simplifyVector = FALSE),
           error = function(e) NULL)
}
read_usage <- function(resp) {
  none <- c(hit = 0, miss = 0, out = 0)
  if (!inherits(resp, "httr2_response")) return(none)
  tryCatch({ u <- resp_body_json(resp)$usage
    c(hit = u$prompt_cache_hit_tokens %||% 0, miss = u$prompt_cache_miss_tokens %||% 0,
      out = u$completion_tokens %||% 0) }, error = function(e) none)
}

cache <- if (file.exists(cache_file)) readRDS(cache_file) else list()
todo  <- sample_ev %>% distinct(event_key, notes) %>%
  filter(map_lgl(event_key, ~ (cache[[.x]]$prompt %||% 0L) < PROMPT_VERSION))
message("To send to DeepSeek: ", nrow(todo))

usage_total <- c(hit = 0, miss = 0, out = 0)
for (rows in split(seq_len(nrow(todo)), ceiling(seq_len(nrow(todo)) / 50))) {
  resps   <- req_perform_parallel(map(todo$notes[rows], build_request),
                                  on_error = "continue", max_active = N_PARALLEL, progress = FALSE)
  answers <- map(resps, read_answer)
  usage_total <- usage_total + reduce(map(resps, read_usage), `+`)
  for (j in seq_along(rows)) if (!is.null(answers[[j]]))
    cache[[todo$event_key[rows[j]]]] <- list(answer = answers[[j]], prompt = PROMPT_VERSION, model = MODEL)
  saveRDS(cache, cache_file)
  if (all(map_lgl(answers, is.null))) stop("Every call in the batch failed (key, balance, model?)")
}
n_calls <- max(nrow(todo), 1)
message("Tokens: ", sum(usage_total[c("hit", "miss")]), " in, ", usage_total[["out"]], " out; ",
        "cost (off-peak prices) $", round(sum(usage_total * PRICE_OFFPEAK) / 1e6, 4),
        "; $", round(sum(usage_total * PRICE_OFFPEAK) / 1e6 / n_calls * 1000, 3), " per 1,000 events")

#------------------------------------------------------------------------------#
####                  3. Resolve the extracted names in SHRID               ####
#------------------------------------------------------------------------------#
PLACE_TYPES <- c("village", "town", "locality", "camp", "police_station", "other")
SUB_TYPES   <- c("subdistrict", "police_station")

places <- sample_ev %>%
  select(event_key, event_id_cnty, unit_id) %>%
  mutate(pl = map(event_key, function(k) {
    a <- cache[[k]]$answer
    if (is.null(a$places) || length(a$places) == 0) return(tibble())
    map_dfr(a$places, ~ tibble(name = .x$name %||% NA_character_, type = .x$type %||% NA_character_,
                               role = .x$role %||% NA_character_, within = .x$within %||% NA_character_))
  })) %>%
  unnest(pl) %>%
  filter(!is.na(name)) %>%
  mutate(key = nospace(name)) %>%
  anti_join(gaz_dist, by = c("unit_id", "key"))

# best SHRID candidate for one name inside one unit; table is gaz_place or gaz_sub
resolve <- function(k, u, table) {
  cand <- table[table$unit_id == u, ]
  if (nrow(cand) == 0 || nchar(k) < 3) return(NULL)
  hit <- cand[cand$key == k, ]
  how <- "exact"
  if (nrow(hit) == 0 && nchar(k) >= 5) {
    d   <- stringdist(k, cand$key, method = "jw", p = 0.1)
    ok  <- d <= JW_MAX & substr(cand$key, 1, 1) == substr(k, 1, 1)
    if (any(ok)) { hit <- cand[ok & d == min(d[ok]), ]; how <- "fuzzy" }
  }
  if (nrow(hit) == 0) return(NULL)
  tibble(gaz_name = hit$gaz_name[1], kind = hit$kind[1], shrid2 = hit$shrid2[1],
         match = how, n_cand = nrow(hit),
         lon = mean(hit$lon, na.rm = TRUE), lat = mean(hit$lat, na.rm = TRUE))
}

# places named as the site (or near it) are tried as villages/towns first, then as sub-districts
places_res <- places %>%
  filter(role %in% c("site", "near")) %>%
  mutate(res = pmap(list(key, unit_id, type), function(k, u, t) {
    r <- if (t %in% PLACE_TYPES) resolve(k, u, gaz_place)
    if (is.null(r) && t %in% c(SUB_TYPES, PLACE_TYPES)) r <- resolve(k, u, gaz_sub)
    r
  })) %>%
  unnest(res, keep_empty = TRUE)

# one row per event: the most specific resolved site, preferring role site over near and exact over fuzzy
level_rank <- c(village = 1, town = 1, subdistrict = 2)
best <- places_res %>%
  filter(!is.na(kind)) %>%
  mutate(rank = level_rank[kind] * 10 + (role == "near") * 2 + (match == "fuzzy")) %>%
  group_by(event_key) %>% slice_min(rank, n = 1, with_ties = FALSE) %>% ungroup() %>%
  select(event_key, llm_name = name, llm_type = type, llm_role = role, gaz_name, kind, match, n_cand, lon, lat)

#------------------------------------------------------------------------------#
####                   4. Rule-based exact match on the same events          ####
#------------------------------------------------------------------------------#
# a SHRID name of the event's unit that appears capitalised as a whole word in the text
rule_match <- function(text, u) {
  words <- unique(str_extract_all(text, "\\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)?\\b")[[1]])
  keys  <- unique(c(nospace(words), nospace(str_extract(words, "^[A-Z][a-z]+"))))
  sub <- gaz_sub[gaz_sub$unit_id == u & gaz_sub$key %in% keys, ]
  plc <- gaz_place[gaz_place$unit_id == u & gaz_place$key %in% keys & nchar(gaz_place$key) >= 4, ]
  dk  <- gaz_dist$key[gaz_dist$unit_id == u]
  plc <- plc[!plc$key %in% dk, ]; sub <- sub[!sub$key %in% dk, ]
  tibble(rule_place = if (nrow(plc)) paste(unique(plc$gaz_name), collapse = "; ") else NA_character_,
         rule_sub   = if (nrow(sub)) paste(unique(sub$gaz_name), collapse = "; ") else NA_character_)
}

events_out <- sample_ev %>%
  mutate(llm_precision = map_chr(event_key, ~ cache[[.x]]$answer$precision %||% NA_character_),
         rule = map2(notes, unit_id, rule_match)) %>%
  unnest(rule) %>%
  left_join(best, by = "event_key") %>%
  mutate(llm_level = case_when(kind %in% c("village", "town") ~ "place",
                               kind == "subdistrict" ~ "subdistrict", TRUE ~ "district"),
         rule_level = case_when(!is.na(rule_place) ~ "place", !is.na(rule_sub) ~ "subdistrict",
                                TRUE ~ "district"),
         agree = !is.na(gaz_name) & (str_detect(coalesce(rule_place, ""), fixed(gaz_name)) |
                                     str_detect(coalesce(rule_sub, ""), fixed(gaz_name))))

#------------------------------------------------------------------------------#
####                              5. Report                                 ####
#------------------------------------------------------------------------------#
named_site <- places %>% filter(role %in% c("site", "near"),
                                !type %in% c("district", "state")) %>% distinct(event_key)
message("\nEvents where the model names a sub-district site: ",
        round(100 * mean(events_out$event_key %in% named_site$event_key), 1), "%")
message("Model's own precision label:"); print(count(events_out, llm_precision))
message("Resolved level, LLM + SHRID vs rule-based, by series (share of events):")
print(events_out %>% group_by(series) %>%
        summarise(n = n(),
                  llm_place = mean(llm_level == "place"), llm_below = mean(llm_level != "district"),
                  rule_place = mean(rule_level == "place"), rule_below = mean(rule_level != "district"),
                  either_below = mean(llm_level != "district" | rule_level != "district")) %>%
        mutate(across(where(is.double), ~ round(100 * .x, 1))))
message("LLM match type:"); print(count(filter(events_out, !is.na(kind)), kind, match))
message("Cross-tab LLM level x rule level:"); print(table(llm = events_out$llm_level, rule = events_out$rule_level))
message("Named site places the gazetteer cannot resolve, by type:")
print(places_res %>% filter(is.na(kind)) %>% count(type, sort = TRUE))

write_csv(events_out %>% select(event_id_cnty, series, admin1, admin2, unit_id, year, llm_precision,
                                llm_name, llm_type, llm_role, gaz_name, kind, match, n_cand, lon, lat,
                                llm_level, rule_place, rule_sub, rule_level, agree, notes),
          "../scraped_data/place_pilot_events.csv")
write_csv(places_res %>% select(-key), "../scraped_data/place_pilot_places.csv")
