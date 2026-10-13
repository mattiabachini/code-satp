#------------------------------------------------------------------------------#
### SATP: district assignment with the SENNA procedure of Fetzer (2020)
#
# Fetzer, "Can Workfare Programs Moderate Conflict? Evidence from India" (JEEA 2020) builds
# events from SATP text with SENNA (Collobert et al. 2011): a sentence is an event candidate
# if it has a conflict verb, that verb has an object (SRL role A1), and a location entity
# matches a district in a gazetteer. This script applies that rule to the hand-coded
# incidents and measures how often it finds the hand-coded district.
#
# The steps:
#   1. Split every incident summary into sentences and tag them with SENNA.
#   2. Per sentence: SENNA verb phrases (verb plus particle) starting with an entry of
#      sandbox/fetzer_verbs.txt (Fetzer's list of 99 stems and phrases, appendix B.2),
#      whether such a verb has an A1 argument, and the location entities (NER tag LOC).
#   3. Match the location entities to GADM districts (names, variants, aliases). A name
#      shared by several districts counts only if a state named in the incident resolves it.
#   4. An incident's district is the first candidate sentence with a resolved district.
#
# Inputs:  data/satp_classification.csv         (hand-coded, Maoist region 2005-2016)
#          data/district_aliases.csv            (spelling variants -> GADM names)
#          sandbox/fetzer_verbs.txt             (conflict verb stems and phrases, one per line)
#          GADM India districts (Irrigating Peace intermediate data, path below)
#          SENNA binary compiled in the Irrigating Peace sandbox (path below)
# Outputs: scraped_data/senna_labeled_sentences.rds   (one row per sentence; delete to rerun SENNA)
#          data/senna_district_errors.csv             (candidate incidents with a wrong district)
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

library(tidyverse)
library(data.table)
library(stringi)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

senna_dir     <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/sandbox/senna"
gadm_file     <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/india_adm2.rds"
alias_file    <- "../data/district_aliases.csv"
labeled_file  <- "../data/satp_classification.csv"
verb_file     <- "sandbox/fetzer_verbs.txt"
sentence_file <- "../scraped_data/senna_labeled_sentences.rds"

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
  left_join(select(districts, gid_2, state, name_2), by = "gid_2")

# state mentions resolve district names shared across states (e.g. Aurangabad)
state_keys <- tibble(
  key   = norm(c(unique(districts$state), "Orissa", "Delhi")),
  state = c(unique(districts$state), "Odisha", "NCT of Delhi")
)

#------------------------------------------------------------------------------#
####                        Hand-coded incidents                            ####
#------------------------------------------------------------------------------#
alias_map <- aliases %>%
  transmute(truth_raw = norm(alias), alias_key = norm(gadm_name_2)) %>%
  distinct(truth_raw, .keep_all = TRUE)

labeled <- read_csv(labeled_file, show_col_types = FALSE) %>%
  filter(!is.na(incident_summary)) %>%
  mutate(row_id = row_number(), truth_raw = norm(district)) %>%
  left_join(alias_map, by = "truth_raw") %>%
  mutate(truth_key = coalesce(alias_key, truth_raw))

#------------------------------------------------------------------------------#
####                           SENNA per sentence                           ####
#------------------------------------------------------------------------------#
verb_pattern <- paste0("^(", paste(readLines(verb_file), collapse = "|"), ")")

# one row per sentence: conflict verbs with an object, and the LOC entities
summarise_sentence <- function(d) {
  verb_rows <- which(d$V5 != "-")
  has_object <- vapply(seq_along(verb_rows),
                       function(j) any(str_detect(d[[5 + j]], "A1$")), logical(1))
  # the verb with its particle, e.g. "blew up"
  verbs <- vapply(seq_along(verb_rows),
                  function(j) str_to_lower(paste(d$V1[str_detect(d[[5 + j]], "-V$")], collapse = " ")),
                  character(1))
  loc <- d[str_detect(d$V4, "-LOC$"), ]
  entity <- cumsum(str_detect(loc$V4, "^[BS]-"))
  list(conflict_verb = any(str_detect(verbs, verb_pattern) & has_object),
       loc_entities  = paste(vapply(split(loc$V1, entity), paste, character(1), collapse = " "), collapse = "|"))
}

if (!file.exists(sentence_file)) {
  sentences <- labeled %>%
    transmute(row_id, text = stri_split_boundaries(incident_summary, type = "sentence")) %>%
    unnest(text) %>%
    mutate(text = str_squish(stri_trans_general(text, "Latin-ASCII"))) %>%   # SENNA reads ASCII only
    filter(text != "") %>%
    mutate(sent_id = row_number())

  in_file  <- tempfile()
  out_file <- tempfile()
  writeLines(sentences$text, in_file)
  system2(file.path(senna_dir, "senna"), c("-path", paste0(senna_dir, "/")),
          stdin = in_file, stdout = out_file)

  # SENNA prints one block of tab-separated token rows per input line, then a blank line;
  # columns: word, POS, chunk, NER, verb, one SRL column per verb, parse tree
  lines <- readLines(out_file)
  blank <- lines == ""
  stopifnot(sum(blank) == nrow(sentences))
  tokens <- as.data.table(tstrsplit(lines[!blank], "\t", fixed = TRUE, fill = ""))
  tokens[, names(tokens) := lapply(.SD, str_trim)]
  tokens[, sent_id := (cumsum(blank) + 1)[!blank]]

  sentences <- sentences %>%
    left_join(tokens[, summarise_sentence(.SD), by = sent_id] %>% as_tibble(), by = "sent_id")
  saveRDS(sentences, sentence_file)
}
sentences <- readRDS(sentence_file)

#------------------------------------------------------------------------------#
####                  Candidate sentences and district match                ####
#------------------------------------------------------------------------------#
entities <- sentences %>%
  filter(loc_entities != "") %>%
  separate_longer_delim(loc_entities, delim = "|") %>%
  mutate(key = str_remove(norm(loc_entities), " (district|districts|dist)$")) %>%   # SENNA tags "Gaya district" as one entity
  filter(key != "")

incident_states <- entities %>%
  inner_join(state_keys, by = "key") %>%
  distinct(row_id, state)

# district names that appear in a candidate sentence, resolved where the name is unique
# or a state named in the incident picks one of its districts
matched <- entities %>%
  filter(conflict_verb) %>%
  inner_join(district_keys, by = "key", relationship = "many-to-many") %>%
  group_by(row_id, sent_id, key) %>%
  mutate(n_candidates = n_distinct(gid_2)) %>%
  ungroup() %>%
  left_join(incident_states %>% mutate(state_named = TRUE), by = c("row_id", "state")) %>%
  group_by(row_id, sent_id, key) %>%
  mutate(n_named = sum(coalesce(state_named, FALSE))) %>%
  ungroup() %>%
  filter(n_candidates == 1 | (coalesce(state_named, FALSE) & n_named == 1)) %>%
  arrange(row_id, sent_id) %>%
  group_by(row_id) %>%
  summarise(n_districts = n_distinct(gid_2), gid_2 = first(gid_2), .groups = "drop")

gazetteer_names <- norm(districts$name_2)

eval_df <- labeled %>%
  left_join(sentences %>% group_by(row_id) %>%
              summarise(has_candidate = any(conflict_verb), .groups = "drop"), by = "row_id") %>%
  left_join(matched, by = "row_id") %>%
  left_join(select(districts, gid_2, pred_name = name_2), by = "gid_2") %>%
  mutate(has_match = !is.na(gid_2),
         correct   = has_match & norm(pred_name) == truth_key,
         in_gaz    = truth_key %in% gazetteer_names)

#------------------------------------------------------------------------------#
####                                Results                                 ####
#------------------------------------------------------------------------------#
message("\nSENNA procedure on ", nrow(eval_df), " hand-coded incidents")
print(eval_df %>% summarise(
  candidate       = mean(has_candidate, na.rm = TRUE),   # share with a conflict verb and object
  coverage        = mean(has_match),                     # share with any district found
  precision       = mean(correct[has_match]),            # correct, among those found
  accuracy        = mean(correct),                       # correct, among all
  ceiling         = mean(in_gaz),                        # share whose true district is in the gazetteer
  accuracy_in_gaz = mean(correct[in_gaz]),
  multi_district  = mean(n_districts > 1, na.rm = TRUE)))

message("\nShare of incidents kept by the verb-and-object rule, by hand-coded action")
print(eval_df %>%
        pivot_longer(c(armed_assault, arrest, bombing, infrastructure, surrender, seizure, abduction),
                     names_to = "action") %>%
        filter(value == 1) %>%
        group_by(action) %>%
        summarise(n = n(), candidate = mean(has_candidate, na.rm = TRUE)))

eval_df %>%
  filter(in_gaz, has_match, !correct) %>%
  select(row_id, state, district, pred_name, n_districts, incident_summary) %>%
  write_csv("../data/senna_district_errors.csv")
