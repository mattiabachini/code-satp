#------------------------------------------------------------------------------#
### SATP district extraction: gazetteer baseline
# Match GADM district names (plus aliases) against incident summaries.
# Inputs:  scraped_data/scraped_incidents.rds   (code/scrape_data_merge.R)
#          data/satp_classification.csv         (hand-coded, Maoist region 2005-2016)
#          data/district_aliases.csv            (spelling variants -> GADM names)
#          GADM India districts (Irrigating Peace intermediate data, path below)
# Output:  scraped_data/scraped_incidents_districts.rds/.csv
#          data/district_baseline_errors.csv    (labeled incidents the baseline gets wrong)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

library(tidyverse)
library(data.table)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

gadm_file     <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/india_adm2.rds"
alias_file    <- "../data/district_aliases.csv"
incident_file <- "../scraped_data/scraped_incidents.rds"
labeled_file  <- "../data/satp_classification.csv"

SHORT_NAME_LEN <- 4      # district names this short only count when followed by "district"
SUFFIX_WORDS   <- c("district", "districts", "dist")
CHUNK          <- 20000  # incidents per matching batch

# lower case, letters only, single spaces
norm <- function(x) x %>% str_to_lower() %>% str_replace_all("[^a-z]+", " ") %>% str_squish()

#------------------------------------------------------------------------------#
####                              Gazetteer                                 ####
#------------------------------------------------------------------------------#
districts <- readRDS(gadm_file) %>%
  sf::st_drop_geometry() %>%
  transmute(gid_2 = GID_2, state = NAME_1, name_2 = NAME_2, varname = VARNAME_2)

aliases <- read_csv(alias_file, show_col_types = FALSE)

district_keys <- bind_rows(
  districts %>% transmute(key = norm(name_2), gid_2),
  districts %>%
    filter(!is.na(varname), varname != "NA") %>%
    mutate(varname = str_replace_all(varname, ",", "|")) %>%
    separate_longer_delim(varname, delim = "|") %>%
    transmute(key = norm(varname), gid_2),
  aliases %>%
    inner_join(districts, by = c("state", "gadm_name_2" = "name_2")) %>%
    transmute(key = norm(alias), gid_2)
) %>%
  filter(key != "") %>%
  distinct() %>%
  left_join(select(districts, gid_2, state, name_2), by = "gid_2") %>%
  mutate(type = "district")

# state mentions disambiguate district names shared across states (e.g. Aurangabad)
state_keys <- tibble(
  key   = norm(c(unique(districts$state), "Orissa", "Delhi")),
  state = c(unique(districts$state), "Odisha", "NCT of Delhi")
) %>% mutate(gid_2 = NA_character_, name_2 = NA_character_, type = "state")

keys <- as.data.table(bind_rows(district_keys, state_keys))
max_n <- max(str_count(keys$key, " ") + 1)
message(n_distinct(district_keys$gid_2), " districts, ", nrow(district_keys),
        " name variants, longest name ", max_n, " words")

#------------------------------------------------------------------------------#
####                               Matching                                 ####
#------------------------------------------------------------------------------#
# One row per incident: primary district (first "X district" mention, else first
# unambiguous name), all districts mentioned, and how many names were ambiguous.
match_incidents <- function(ids, txt) {
  w  <- strsplit(norm(txt), " ", fixed = TRUE)
  tk <- data.table(id = rep(ids, lengths(w)), tok = unlist(w))
  tk[, `:=`(row = .I, pos = seq_len(.N)), by = id]
  used  <- rep(FALSE, nrow(tk))   # tokens already claimed by a longer name
  found <- list()

  for (n in max_n:1) {
    key  <- do.call(paste, c(lapply(0:(n - 1), function(k) shift(tk$tok, -k)), sep = " "))
    same <- shift(tk$id, -(n - 1)) == tk$id
    nxt  <- ifelse(shift(tk$id, -n) == tk$id, shift(tk$tok, -n), NA_character_)
    cand <- which(same & key %in% keys$key)
    if (length(cand) == 0) next

    m <- merge(data.table(r = cand, key = key[cand], nxt = nxt[cand]),
               keys, by = "key", allow.cartesian = TRUE)
    m[, suffix := nxt %in% SUFFIX_WORDS]
    m <- m[!(type == "district" & nchar(key) <= SHORT_NAME_LEN & !suffix)]
    if (nrow(m) == 0) next

    rr   <- unique(m$r)
    idx  <- outer(rr, 0:(n - 1), "+")
    free <- rowSums(matrix(used[idx], nrow = length(rr))) == 0
    used[idx[free, ]] <- TRUE
    m <- m[r %in% rr[free]]
    found[[n]] <- m[, .(id = tk$id[r], pos = tk$pos[r], type, gid_2, state, suffix)]
  }
  if (length(found) == 0) return(data.table())
  m <- rbindlist(found)

  st <- unique(m[type == "state", .(id, st = state)])
  dm <- m[type == "district"]
  dm[, `:=`(N = .N, ment = FALSE), by = .(id, pos)]
  dm[unique(dm[st, on = .(id, state = st), nomatch = NULL, which = TRUE]), ment := TRUE]
  dm[, n_ment := sum(ment), by = .(id, pos)]
  resolved <- dm$N == 1 | (dm$ment & dm$n_ment == 1)

  ok  <- dm[resolved][order(id, pos)]
  amb <- dm[!resolved, .(n_ambiguous = uniqueN(pos)), by = id]

  out <- ok[, .(districts_all = paste(unique(gid_2), collapse = "|"),
                n_districts   = uniqueN(gid_2)), by = id]
  prim <- ok[order(id, -suffix, pos)][, .SD[1], by = id][, .(id, gid_2, match_type = fifelse(suffix, "district suffix", "bare name"))]
  out <- merge(out, prim, by = "id", all = TRUE)
  merge(out, amb, by = "id", all = TRUE)
}

run_matching <- function(ids, txt) {
  groups <- split(seq_along(ids), ceiling(seq_along(ids) / CHUNK))
  imap(groups, ~ { message("matching chunk ", .y, " of ", length(groups)); match_incidents(ids[.x], txt[.x]) }) %>%
    rbindlist(fill = TRUE)
}

#------------------------------------------------------------------------------#
####                  Evaluation on the hand-coded incidents                ####
#------------------------------------------------------------------------------#
alias_map <- aliases %>%
  transmute(truth_raw = norm(alias), alias_key = norm(gadm_name_2)) %>%
  distinct(truth_raw, .keep_all = TRUE)

labeled <- read_csv(labeled_file, show_col_types = FALSE) %>%
  filter(!is.na(incident_summary)) %>%
  mutate(row_id = row_number(), truth_raw = norm(district)) %>%
  left_join(alias_map, by = "truth_raw") %>%
  mutate(truth_key = coalesce(alias_key, truth_raw))

lab_match <- run_matching(labeled$row_id, labeled$incident_summary)
gazetteer_names <- norm(districts$name_2)

eval_df <- labeled %>%
  left_join(as_tibble(lab_match), by = c("row_id" = "id")) %>%
  left_join(select(districts, gid_2, pred_name = name_2), by = "gid_2") %>%
  mutate(has_match = !is.na(gid_2),
         correct   = has_match & norm(pred_name) == truth_key,
         in_gaz    = truth_key %in% gazetteer_names)

message("\nBaseline on ", nrow(eval_df), " hand-coded incidents")
print(eval_df %>% summarise(
  coverage          = mean(has_match),                 # share with any district found
  precision         = mean(correct[has_match]),        # correct, among those found
  accuracy          = mean(correct),                   # correct, among all
  ceiling           = mean(in_gaz),                    # share whose true district is in the gazetteer
  accuracy_in_gaz   = mean(correct[in_gaz]),
  share_ambiguous   = mean(!is.na(n_ambiguous))))
print(eval_df %>% group_by(state) %>%
        summarise(n = n(), coverage = mean(has_match), accuracy = mean(correct)) %>%
        arrange(desc(n)) %>% head(12))
message("\nTrue districts missing from the gazetteer (candidates for district_aliases.csv):")
print(eval_df %>% filter(!in_gaz) %>% count(state, district, sort = TRUE) %>% head(25))

eval_df %>%
  filter(in_gaz, !correct) %>%
  select(row_id, state, district, pred_name, match_type, n_districts, incident_summary) %>%
  write_csv("../data/district_baseline_errors.csv")

#------------------------------------------------------------------------------#
####                        Apply to scraped incidents                      ####
#------------------------------------------------------------------------------#
incidents <- readRDS(incident_file)
inc_match <- run_matching(incidents$incident_uid, incidents$incident_summary)

incidents_districts <- incidents %>%
  select(incident_uid, date, series) %>%
  left_join(as_tibble(inc_match) %>% rename(incident_uid = id), by = "incident_uid") %>%
  left_join(select(districts, gid_2, state, district = name_2), by = "gid_2")

message("\nShare of incidents with a district, by series tag")
for (tag in c("india-maoistinsurgency", "india-jammukashmir", "india-insurgencynortheast", "india-punjab")) {
  sel <- str_detect(incidents_districts$series, fixed(tag))
  message("  ", tag, ": ", round(mean(!is.na(incidents_districts$gid_2[sel])), 3), " (n = ", sum(sel), ")")
}
message("  all: ", round(mean(!is.na(incidents_districts$gid_2)), 3))

saveRDS(incidents_districts, "../scraped_data/scraped_incidents_districts.rds")
write_csv(incidents_districts, "../scraped_data/scraped_incidents_districts.csv")
