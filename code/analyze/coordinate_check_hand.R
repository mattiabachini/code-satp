#------------------------------------------------------------------------------#
### SATP: coordinate check on the hand-checked place rows (notes/todo.md, item 1)
#
# Reads a COPY of data/intermediate/validation/place_check_hand.xlsx (the original is never
# opened for writing; Excel may have it open) and the check key, then:
#   1. in_district / km_outside_district: is the pipeline pin inside the event's
#      2001 district unit (unit_id from the key)?
#   2. shrid_village_km: for pins that did not come from SHRID, distance from the
#      pin to the SHRID village of the extracted name inside the same unit
#      (polygon distance, centroid fallback). Hyphenated names are split.
# Reports mean / median / sd of the distances, split by the hand-check answer.
# No API calls. Writes data/intermediate/validation/coordinate_check_hand.csv only.
#
# Note on aliases: unit_id comes with the event, so the district alias list is
# not needed here; it matters when a unit is looked up from a district NAME.
#------------------------------------------------------------------------------#
suppressPackageStartupMessages({
  library(data.table); library(sf); library(readxl); library(stringi)
})
root <- Sys.getenv("SATP_ROOT", unset = "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
ip_dir <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/"
use_polygons <- !identical(Sys.getenv("NO_POLYGONS"), "1")

# 1. Sheet (read from a copy) ------------------------------------------------
tmp <- tempfile(fileext = ".xlsx")
file.copy(file.path(root, "data/intermediate/validation/place_check_hand.xlsx"), tmp)
h <- as.data.table(read_excel(tmp, col_types = "text"))
h[, check_id := as.integer(check_id)]
h[, row_in_entry := seq_len(.N), by = check_id]
num <- function(x) as.numeric(x)
lat <- fifelse(grepl("mlat=", h$osm_link), sub(".*mlat=([-0-9.]+)&.*", "\\1", h$osm_link),
               fifelse(grepl("#map=\\d+/", h$osm_link), sub(".*#map=\\d+/([-0-9.]+)/.*", "\\1", h$osm_link), NA_character_))
lon <- fifelse(grepl("mlon=", h$osm_link), sub(".*mlon=([-0-9.]+)#.*", "\\1", h$osm_link),
               fifelse(grepl("#map=\\d+/", h$osm_link), sub(".*#map=\\d+/[-0-9.]+/([-0-9.]+).*", "\\1", h$osm_link), NA_character_))
h[, `:=`(lat = num(lat), lon = num(lon))]

# 2. Key: unit and source of the pin -------------------------------------------
k <- fread(file.path(root, "data/intermediate/validation/place_check_key.csv"), na.strings = "")
k <- k[check_id %in% h$check_id]
pick_source <- function(cid, la, lo) {
  if (is.na(la)) return(list(src = NA_character_, nm = NA_character_))
  r <- k[check_id == cid]
  for (s in c("shrid", "geonames", "osm")) {
    if (isTRUE(abs(r[[paste0(s, "_lat")]] - la) < 2e-5 && abs(r[[paste0(s, "_lon")]] - lo) < 2e-5))
      return(list(src = s, nm = r[[paste0(s, "_extracted")]]))
  }
  list(src = "manual_or_other", nm = NA_character_)
}
ps <- Map(pick_source, h$check_id, h$lat, h$lon)
h[, `:=`(source = vapply(ps, `[[`, "", "src"), extracted = vapply(ps, `[[`, "", "nm"))]
h[, unit_id := k$unit_id[match(check_id, k$check_id)]]
h[, tier := k$tier[match(check_id, k$check_id)]]

# 3. District check --------------------------------------------------------------
units <- st_make_valid(st_read(paste0(ip_dir, "india_units_2001.gpkg"), quiet = TRUE))
units <- st_transform(units, 7755)
h[, `:=`(in_district = NA, km_outside_district = NA_real_)]
for (i in which(!is.na(h$lat) & !is.na(h$unit_id))) {
  u <- units[units$unit_id == h$unit_id[i], ]
  if (!nrow(u)) next
  p <- st_transform(st_sfc(st_point(c(h$lon[i], h$lat[i])), crs = 4326), 7755)
  inside <- lengths(st_intersects(p, u)) > 0
  set(h, i, "in_district", inside)
  if (!inside) set(h, i, "km_outside_district", round(as.numeric(st_distance(p, st_union(st_geometry(u)))) / 1000, 2))
}

# 4. SHRID village distance --------------------------------------------------------
strip_words <- c("gram panchayat", "police station", "sub-division", "subdivision", "cd block",
                 "taluka", "tahsil", "tehsil", "taluk", "block", "mandal", "circle", "village",
                 "tola", "para", "basti", "hills", "hill", "valley", "forest", "jungle", "camp", "thana", "ps")
strip_words <- strip_words[order(-nchar(strip_words))]
tp <- paste0("\\b(", paste(strip_words, collapse = "|"), ")\\b")
norm <- function(x) {
  x <- stri_trans_tolower(stri_trans_general(x, "Latin-ASCII"))
  x <- stri_replace_all_regex(x, tp, " ")
  stri_replace_all_regex(x, "[^\\p{L}]", "")
}
nm <- fread(file.path(root, "data/intermediate/gazetteers/shrid_loc_names.csv"))
sk <- unique(as.data.table(readRDS(paste0(ip_dir, "india_shrid_unit_key.rds")))[, .(shrid2, unit_id)])
cen <- as.data.table(readRDS(file.path(root, "data/intermediate/gazetteers/shrid_centroids.rds")))
g <- merge(merge(nm, sk, by = "shrid2"), cen, by = "shrid2", all.x = TRUE)
g[, key := norm(place_name)]

# Candidate names per row: extracted name (pipeline) else the site name from the sheet.
h[, lookup_name := fifelse(!is.na(extracted) & nzchar(extracted), extracted,
                           fifelse(!is.na(site_name_in_text), site_name_in_text, proposed_place))]
h[, to_check := !is.na(lat) & !is.na(lookup_name) & source != "shrid"]
cands <- rbindlist(lapply(which(h$to_check), function(i) {
  parts <- unique(c(lookup_name <- h$lookup_name[i], strsplit(h$lookup_name[i], "[-/,]")[[1]]))
  keys <- unique(norm(parts)); keys <- keys[nchar(keys) >= 3]
  m <- g[unit_id == h$unit_id[i] & key %in% keys]
  if (!nrow(m)) return(NULL)
  m[, row := i]
  m[, .(row, shrid2, lon, lat)]
}))
h[, `:=`(n_shrid_matches = 0L, shrid_village_km = NA_real_, shrid_dist_basis = NA_character_)]
if (nrow(cands)) {
  ids <- unique(cands$shrid2)
  poly <- NULL
  if (use_polygons) {
    q <- sprintf("SELECT shrid2 FROM shrid2_open WHERE shrid2 IN (%s)",
                 paste0("'", ids, "'", collapse = ","))
    poly <- tryCatch(st_read(file.path(root, "data/input/external/shrug/shrid2_open.shp"), query = q, quiet = TRUE),
                     error = function(e) { message("polygon read failed: ", conditionMessage(e)); NULL })
    if (!is.null(poly)) { poly$shrid2 <- trimws(poly$shrid2); poly <- st_transform(st_make_valid(poly), 7755) }
  }
  for (i in unique(cands$row)) {
    cc <- cands[row == i]
    p <- st_transform(st_sfc(st_point(c(h$lon[i], h$lat[i])), crs = 4326), 7755)
    d <- vapply(seq_len(nrow(cc)), function(j) {
      pg <- if (!is.null(poly)) poly[poly$shrid2 == cc$shrid2[j], ] else NULL
      if (!is.null(pg) && nrow(pg)) c(as.numeric(st_distance(p, st_union(pg))) / 1000, 1)
      else if (!is.na(cc$lon[j])) c(as.numeric(st_distance(p, st_transform(st_sfc(st_point(c(cc$lon[j], cc$lat[j])), crs = 4326), 7755))) / 1000, 0)
      else c(NA_real_, NA_real_)
    }, numeric(2))
    b <- which.min(d[1, ])
    if (length(b)) set(h, i, c("n_shrid_matches", "shrid_village_km", "shrid_dist_basis"),
                       list(nrow(cc), round(d[1, b], 2), if (d[2, b] == 1) "polygon" else "centroid"))
  }
}

# 5. Output and report --------------------------------------------------------------
out <- h[, .(check_id, row_in_entry, district, proposed_place, site_name_in_text, source, tier, unit_id,
             lat, lon, in_district, km_outside_district, lookup_name, n_shrid_matches,
             shrid_village_km, shrid_dist_basis, proposed_correct, right_subdistrict, note)]
fwrite(out, file.path(root, "data/intermediate/validation/coordinate_check_hand.csv"), na = "")
cat("rows:", nrow(h), " with a pin:", sum(!is.na(h$lat)), " no pin:", sum(is.na(h$lat)), "\n")
cat("\nin_district (rows with a pin):\n"); print(table(h$in_district, useNA = "ifany"))
cat("\nsource of the pin:\n"); print(table(h$source, useNA = "ifany"))
cat("\nout-of-district rows:\n")
print(out[in_district == FALSE, .(check_id, proposed_place, district, km_outside_district, proposed_correct, note = substr(note, 1, 60))])
cat("\nhand answer x in_district (answered rows):\n")
print(table(answer = out$proposed_correct, in_district = out$in_district, useNA = "ifany"))
st <- function(x) data.table(n = length(x), mean = round(mean(x), 2), median = round(median(x), 2), sd = round(sd(x), 2))
cat("\nSHRID-village distance (km), all looked-up rows:\n"); d <- out[!is.na(shrid_village_km)]
print(st(d$shrid_village_km))
cat("by hand answer:\n"); print(d[, c(list(answer = proposed_correct[1]), st(shrid_village_km)), by = .(a = fifelse(is.na(proposed_correct), "not_yet_entered", proposed_correct))][, a := NULL][])
cat("rows to look up but no SHRID name match:", sum(h$to_check & h$n_shrid_matches == 0), "of", sum(h$to_check), "\n")
