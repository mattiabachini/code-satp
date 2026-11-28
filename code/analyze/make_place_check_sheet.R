#------------------------------------------------------------------------------#
### SATP: blind place-check sheet from the revised pilot output
#
# Draws 100 of the 500 pilot events by simple random sampling (fixed seed, no
# stratification), writes a blind hand-check workbook and the all-event sheet,
# and keeps the key (tier, sources, match details, flags, weight) separate.
#
# Weight: the series' share of the analysis set divided by its share of the 500
# pilot events, so the 100 can be reweighted back to the analysis set.
#
# Inputs:  data/intermediate/places/place_pilot_multisource.csv
#          irrigating_peace/.../satp_events_conf07.rds
#          irrigating_peace/.../india_units_2001.gpkg
# Outputs: data/intermediate/validation/place_check_hand.xlsx
#          data/intermediate/validation/place_check_all.csv
#          data/intermediate/validation/place_check_key.csv
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(data.table)
  library(sf)
  library(openxlsx)
})

root <- Sys.getenv("SATP_ROOT", unset = "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
ip_dir <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/"
out_dir <- file.path(root, "data/intermediate/validation")
dir.create(out_dir, showWarnings = FALSE)

ms <- fread(file.path(root, "data/intermediate/places/place_pilot_multisource.csv"), na.strings = "NA")
events <- as.data.table(readRDS(paste0(ip_dir, "satp_events_conf07.rds")))
units <- as.data.table(st_read(paste0(ip_dir, "india_units_2001.gpkg"), quiet = TRUE))[
  , .(unit_id, unit_name)]

series_of <- function(tags) {
  fcase(grepl("maoist", tags, ignore.case = TRUE), "Maoist",
        grepl("jammukashmir", tags, ignore.case = TRUE), "J&K",
        grepl("northeast", tags, ignore.case = TRUE), "Northeast",
        default = "Punjab/other")
}
analysis <- events[event_confidence > 0.7 & any_analysis_action == 1 & !is.na(unit_id)]
analysis[, series := series_of(tags)]
analysis_counts <- analysis[, .N, by = series]
analysis_counts[, analysis_share := N / sum(N)]
pilot_counts <- ms[, .N, by = series]
pilot_counts[, pilot_share := N / sum(N)]
w <- merge(analysis_counts[, .(series, analysis_n = N, analysis_share)],
           pilot_counts[, .(series, pilot_n = N, pilot_share)], by = "series", all = TRUE)
w[, weight := analysis_share / pilot_share]
ms[, weight := w$weight[match(series, w$series)]]

# Fixed-seed simple random draw of 100 events and a random row order for all 500.
set.seed(20261120)
hand_keys <- sample(ms$event_key, 100)
ord <- sample(ms$event_key)
ms <- ms[match(ord, event_key)]
ms[, `:=`(check_id = .I, in_hand = event_key %in% hand_keys)]

# Event text and district.
notes <- unique(events[, .(event_id_cnty, notes)], by = "event_id_cnty")
ms[, event_text := notes$notes[match(event_id_cnty, notes$event_id_cnty)]]
ms[, district := units$unit_name[match(unit_id, units$unit_id)]]
ms[, state := state_name]

# Proposed candidate: the source candidate carrying shrid2_final when possible,
# otherwise the best exact/fuzzy candidate. Source order: SHRID, GeoNames, OSM.
srcs <- c("shrid", "geonames", "osm")
long <- rbindlist(lapply(srcs, function(s) data.table(
  event_key = ms$event_key, source = s,
  extracted = ms[[paste0(s, "_extracted")]],
  gazetteer = ms[[paste0(s, "_gazetteer")]],
  match = ms[[paste0(s, "_match")]],
  lon = ms[[paste0(s, "_lon")]],
  lat = ms[[paste0(s, "_lat")]],
  tehsil = ms[[paste0(s, "_tehsil_name")]],
  shrid2 = ms[[paste0(s, "_shrid2")]])))
setorder(long, event_key)

choose_cand <- function(k, sf) {
  d <- long[event_key == k & !is.na(lon)]
  if (nrow(d) == 0) return(NULL)
  if (!is.na(sf) && nzchar(sf)) {
    inside <- vapply(d$shrid2, function(x) {
      if (is.na(x) || !nzchar(x)) FALSE else sf %in% strsplit(x, ";", fixed = TRUE)[[1]]
    }, logical(1))
    if (any(inside)) d <- d[inside]
  }
  d[, ord := base::match(source, srcs)]
  d[, mrank := fifelse(match == "exact", 0L, 1L)]
  setorder(d, mrank, ord)
  d[1]
}
cand <- lapply(seq_len(nrow(ms)), function(i) choose_cand(ms$event_key[i], ms$shrid2_final[i]))
cand_at <- function(field) vapply(cand, function(z) {
  if (is.null(z)) return(if (field %in% c("lon", "lat")) NA_real_ else NA_character_)
  z[[field]]
}, FUN.VALUE = if (field %in% c("lon", "lat")) numeric(1) else character(1))

proposed_place <- cand_at("gazetteer")
proposed_subdistrict <- cand_at("tehsil")
proposed_lon <- cand_at("lon")
proposed_lat <- cand_at("lat")
osm_link <- ifelse(is.na(proposed_lat), "",
                   sprintf("https://www.openstreetmap.org/?mlat=%.6f&mlon=%.6f#map=14/%.6f/%.6f",
                           proposed_lat, proposed_lon, proposed_lat, proposed_lon))
pc_first <- function(x) {
  if (is.na(x) || !nzchar(x)) return("")
  strsplit(x, ";", fixed = TRUE)[[1]][1]
}
proposed_pc <- vapply(ms$pc01_village_code, pc_first, character(1))

check <- data.table(
  check_id = ms$check_id,
  event_id_cnty = ms$event_id_cnty,
  district = ms$district,
  state = ms$state_name,
  event_text = ms$event_text,
  proposed_place = ifelse(is.na(proposed_place), "", proposed_place),
  proposed_subdistrict = ifelse(is.na(proposed_subdistrict), "", proposed_subdistrict),
  proposed_pc01_code = proposed_pc,
  osm_link = osm_link,
  site_name_in_text = "",
  proposed_correct = "",
  right_subdistrict = "",
  note = ""
)

fwrite(check, file.path(out_dir, "place_check_all.csv"), na = "")
hand <- check[ms$in_hand == TRUE]

# Excel sheet with only the requested columns.
wb <- createWorkbook()
addWorksheet(wb, "check")
writeData(wb, "check", as.data.frame(hand), withFilter = TRUE, keepNA = FALSE)
header <- createStyle(textDecoration = "bold", fgFill = "#D9E1F2", border = "bottom")
addStyle(wb, "check", header, rows = 1, cols = seq_len(ncol(hand)), gridExpand = TRUE)
setColWidths(wb, "check",
             cols = seq_len(ncol(hand)),
             widths = c(8, 13, 20, 15, 70, 22, 22, 18, 45, 20, 16, 16, 30))
addStyle(wb, "check", createStyle(wrapText = TRUE),
         rows = 2:(nrow(hand) + 1), cols = c(5, 9, 13), gridExpand = TRUE, stack = TRUE)
freezePane(wb, "check", firstRow = TRUE)
# Allowed answers for the two judgement columns.
dataValidation(wb, "check", cols = 11:12, rows = 2:(nrow(hand) + 1),
               type = "list", value = '"yes,no,cant_tell"', allowBlank = TRUE)
saveWorkbook(wb, file.path(out_dir, "place_check_hand.xlsx"), overwrite = TRUE)

# Key: tier, sources, match details, flags and weight, linked by check_id.
for (s in srcs) ms[, (paste0(s, "_present")) := !is.na(get(paste0(s, "_lon")))]
present_mat <- as.matrix(ms[, paste0(srcs, "_present"), with = FALSE])
ms[, sources := apply(present_mat, 1, function(z) paste(srcs[z], collapse = ";"))]

key <- ms[, .(check_id, event_id_cnty, event_key, series, unit_id, district, state,
              in_hand, weight, sources, tier, tier_before, tier_pre_a5,
              a5_changed, a5_shrid_lost,
              n_sources, n_sources_agree, same_tehsil, same_shrid,
              shrid2_final, pc01_village_code, n_pc01, shrid_n_pc01, shrid_urban,
              fuzzy_support, district_hq_name, exact_override, conflict_resolved_exact,
              shrid_present, shrid_extracted, shrid_gazetteer, shrid_match,
              shrid_jw_dist, shrid_len_ratio, shrid_lon, shrid_lat,
              shrid_tehsil_name, shrid_shrid2,
              geonames_present, geonames_extracted, geonames_gazetteer, geonames_match,
              geonames_jw_dist, geonames_len_ratio, geonames_lon, geonames_lat,
              geonames_tehsil_name, geonames_shrid2,
              osm_present, osm_extracted, osm_gazetteer, osm_match,
              osm_jw_dist, osm_len_ratio, osm_lon, osm_lat,
              osm_tehsil_name, osm_shrid2)]
fwrite(key, file.path(out_dir, "place_check_key.csv"), na = "")

# Report: weights, then the 100 by series x tier.
cat("=== analysis-set and pilot shares, event weight ===\n")
print(w[, .(series, analysis_n, analysis_share = round(analysis_share, 4),
            pilot_n, pilot_share = round(pilot_share, 4),
            weight = round(weight, 4))])

cat("\n=== draw of 100: series x tier ===\n")
hand_key <- key[in_hand == TRUE]
draw_tab <- dcast(hand_key[, .N, by = .(series, tier)], series ~ tier,
                  value.var = "N", fill = 0)
for (tt in c("shrid", "high", "tehsil", "single", "single_fuzzy", "conflict", "none")) {
  if (!tt %in% names(draw_tab)) draw_tab[, (tt) := 0L]
}
setcolorder(draw_tab, c("series", intersect(c("shrid", "high", "tehsil", "single",
                                               "single_fuzzy", "conflict", "none"),
                                             names(draw_tab))))
print(draw_tab)
cat("\nby series:\n")
print(hand_key[, .N, by = series][order(-N)])
cat("\nwrote", file.path(out_dir, "place_check_hand.xlsx"), ",",
    file.path(out_dir, "place_check_all.csv"), "and",
    file.path(out_dir, "place_check_key.csv"), "\n")
