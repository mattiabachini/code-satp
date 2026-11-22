#------------------------------------------------------------------------------#
### SATP: pilot place resolution across SHRID, GeoNames and OSM
#
# Second revision of the multisource pilot resolver. DeepSeek answers come from
# scraped_data/place_cache_pilot.rds only (no API calls).
#
# New in this revision:
# - district filtering adds the alias lists in code/sandbox/district_aliases.csv
#   and data/district_aliases.csv, plus JW <= 0.10 variants; a name typed "town"
#   is kept and flagged district_hq_name;
# - exact matches override fuzzy matches for the same place when the exact
#   matches are consistent (one exact source, within 5 km, or shared SHRID);
# - len_ratio is stored for every fuzzy match;
# - shrid_n_pc01 and shrid_urban are added to the event file.
#
# The before/after tier tables hold the candidate set at the old district filter
# and old aggregation (before) versus the new filter and exact-over-fuzzy rule
# (after).
#
# Inputs:  irrigating_peace/.../satp_events_conf07.rds
#          irrigating_peace/.../india_shrid_unit_key.rds
#          irrigating_peace/.../india_units_2001.gpkg
#          irrigating_peace/data/input/public/SHRUG/shrug-shrid-poly-shp.zip
#          irrigating_peace/data/input/public/SHRUG/shrug-pc-keys-dta.zip
#          scraped_data/place_cache_pilot.rds
#          data/shrid_loc_names.csv, data/shrid_centroids.rds
#          data/geonames_gazetteer_india.rds, data/osm_gazetteer_india.rds
#          data/osm_admin_boundaries_india.gpkg (admin_level_6)
#          code/sandbox/district_aliases.csv, data/district_aliases.csv
# Outputs: scraped_data/place_pilot_candidates.csv
#          scraped_data/place_pilot_multisource.csv
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(data.table)
  library(sf)
  library(stringi)
  library(stringdist)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(rlang)
  library(haven)
})

root <- Sys.getenv("SATP_ROOT", unset = "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
ip_dir <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/"
shrug_pub <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/data/input/public/SHRUG"
shrug_dir <- file.path(root, "data/raw/shrug")
out_dir <- file.path(root, "scraped_data")
dir.create(out_dir, showWarnings = FALSE)
dir.create(shrug_dir, showWarnings = FALSE)

#------------------------------------------------------------------------------#
####                        1. Pilot events and places                     ####
#------------------------------------------------------------------------------#
series_of <- function(tags) case_when(
  str_detect(tags, "maoist")       ~ "Maoist",
  str_detect(tags, "jammukashmir") ~ "J&K",
  str_detect(tags, "northeast")    ~ "Northeast",
  TRUE                             ~ "Punjab/other")

analysis <- readRDS(paste0(ip_dir, "satp_events_conf07.rds")) %>%
  filter(event_confidence > 0.7, any_analysis_action == 1, !is.na(unit_id)) %>%
  mutate(series = series_of(tags))

quota <- round(500 * c(Maoist = 0.3, `J&K` = 0.3, Northeast = 0.3, `Punjab/other` = 0.1))
set.seed(20261119)
sample_ev <- analysis %>%
  group_by(series) %>%
  group_modify(~ slice_sample(.x, n = min(quota[[.y$series]], nrow(.x)))) %>%
  ungroup() %>%
  mutate(event_key = map_chr(notes, rlang::hash))

cache <- readRDS(file.path(out_dir, "place_cache_pilot.rds"))
stopifnot(all(sample_ev$event_key %in% names(cache)))
cat("pilot events:", nrow(sample_ev), "\n"); print(count(sample_ev, series))

ev <- as.data.table(sample_ev)[, .(event_key, event_id_cnty, unit_id, series, notes)]
stopifnot(!anyDuplicated(ev$event_key))

keep_types <- c("village", "town", "locality", "camp", "subdistrict",
                "police_station", "forest", "water", "road")

places <- rbindlist(lapply(seq_len(nrow(ev)), function(i) {
  a <- cache[[ev$event_key[i]]]$answer
  if (is.null(a$places) || length(a$places) == 0) return(NULL)
  rbindlist(lapply(seq_along(a$places), function(j) {
    p <- a$places[[j]]
    data.table(event_key = ev$event_key[i], event_id_cnty = ev$event_id_cnty[i],
               unit_id = ev$unit_id[i], series = ev$series[i], place_order = j,
               name = as.character(p$name %||% NA_character_),
               type = as.character(p$type %||% NA_character_),
               role = as.character(p$role %||% NA_character_),
               within = as.character(p$within %||% NA_character_))
  }))
}), fill = TRUE)

places <- places[role %in% c("site", "near") & type %in% keep_types &
                   !is.na(name) & nzchar(trimws(name))]
cat("extracted places (site/near):", nrow(places), "\n")

#------------------------------------------------------------------------------#
####                            2. Normaliser                               ####
#------------------------------------------------------------------------------#
strip_words <- c("gram panchayat", "police station", "sub-division", "subdivision",
                 "cd block", "taluka", "tahsil", "tehsil", "taluk", "block",
                 "mandal", "circle", "village", "tola", "para", "basti", "hills",
                 "hill", "valley", "forest", "jungle", "camp", "thana", "ps")
strip_words <- strip_words[order(-nchar(strip_words))]
type_pattern <- paste0("\\b(", paste(strip_words, collapse = "|"), ")\\b")

norm <- function(x) {
  x <- stri_trans_general(x, "Latin-ASCII")
  x <- stri_trans_tolower(x)
  x <- stri_replace_all_regex(x, type_pattern, " ")
  x <- stri_replace_all_regex(x, "[^\\p{L}]", "")
  x
}

norm_state <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("&", "and", x)
  x <- gsub("[^a-z ]", " ", x)
  x <- gsub("\\band\\b|\\bof\\b", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

haversine <- function(lon1, lat1, lon2, lat2) {
  rad <- pi / 180
  dlat <- (lat2 - lat1) * rad
  dlon <- (lon2 - lon1) * rad
  a <- sin(dlat / 2)^2 + cos(lat1 * rad) * cos(lat2 * rad) * sin(dlon / 2)^2
  2 * 6371.0088 * asin(pmin(1, sqrt(a)))
}

#------------------------------------------------------------------------------#
####                    3. Units, districts and gazetteers                  ####
#------------------------------------------------------------------------------#
units_sf <- st_make_valid(st_read(paste0(ip_dir, "india_units_2001.gpkg"), quiet = TRUE))
cent <- st_coordinates(st_centroid(st_geometry(units_sf)))
units_dt <- data.table(unit_id = units_sf$unit_id, unit_name = units_sf$unit_name,
                       state_name = units_sf$state_name,
                       unit_lon = cent[, 1], unit_lat = cent[, 2])
units_dt[, state_key := norm_state(state_name)]
ev <- merge(ev, units_dt[, .(unit_id, unit_name, state_name, unit_lon, unit_lat, state_key)],
            by = "unit_id", all.x = TRUE)

shrid_names <- fread(file.path(root, "data/shrid_loc_names.csv"))
shrid_key <- as.data.table(readRDS(paste0(ip_dir, "india_shrid_unit_key.rds")))[, .(shrid2, unit_id)]
shrid_cent <- as.data.table(readRDS(file.path(root, "data/shrid_centroids.rds")))
g <- merge(shrid_names, unique(shrid_key), by = "shrid2", all = FALSE)
g <- merge(g, shrid_cent, by = "shrid2", all.x = TRUE)

# District names: 2001 units + 2011 SHRID districts (old set), and the alias
# lists (expanded set).
old_dist <- unique(rbind(
  units_dt[, .(state_key, key = norm(unit_name))],
  shrid_names[, .(state_key = norm_state(state_name), key = norm(district_name))]
))
alias_files <- c(file.path(root, "code/sandbox/district_aliases.csv"),
                 file.path(root, "data/district_aliases.csv"))
aliases <- rbindlist(lapply(alias_files, function(f) {
  a <- fread(f)
  out <- data.table(state = a$state, name = a$alias)
  if ("known_name" %in% names(a)) out <- rbind(out, data.table(state = a$state, name = a$known_name))
  if ("gadm_name_2" %in% names(a)) out <- rbind(out, data.table(state = a$state, name = a$gadm_name_2))
  out
}), fill = TRUE)
aliases <- aliases[!is.na(name) & nzchar(trimws(name))]
aliases[, `:=`(state_key = norm_state(state), key = norm(name))]
aliases <- aliases[nzchar(key)]
old_dist <- old_dist[nzchar(key)]
expanded_dist <- unique(rbind(old_dist, aliases[, .(state_key, key)]))
expanded_dist <- expanded_dist[nzchar(key)]
old_list <- split(old_dist$key, old_dist$state_key)
exp_list <- split(expanded_dist$key, expanded_dist$state_key)

sp <- g[!is.na(place_name) & !is.na(lon) & !is.na(lat),
        .(unit_id, name = place_name, subdistrict_name, lon, lat, feature_id = shrid2)]
sp[, key := norm(name)]
sp <- sp[nzchar(key) & nchar(key) >= 3]
sp[, `:=`(level = "place", shrid2_own = feature_id, osm_tag = NA_character_)]

ss <- g[!is.na(subdistrict_name),
        .(lon = mean(lon, na.rm = TRUE), lat = mean(lat, na.rm = TRUE)),
        by = .(unit_id, name = subdistrict_name)]
ss <- ss[!is.nan(lon) & !is.nan(lat)]
ss[, key := norm(name)]
ss <- ss[nzchar(key) & nchar(key) >= 3]
ss[, `:=`(feature_id = paste(unit_id, name, sep = "_"), level = "subdistrict",
          subdistrict_name = name, shrid2_own = NA_character_, osm_tag = NA_character_)]

geo <- as.data.table(readRDS(file.path(root, "data/geonames_gazetteer_india.rds")))
geo <- geo[!is.na(unit_id) & !is.na(lon) & !is.na(lat)]
geo_names <- unique(geo$name)
geo_key <- data.table(name = geo_names)[, key := norm(name)]
geo[, key := geo_key$key[match(name, geo_names)]]
geo[, `:=`(feature_id = as.character(geonameid), subdistrict_name = NA_character_,
           shrid2_own = NA_character_, osm_tag = NA_character_)]

gp <- geo[feature_class == "P" & nzchar(key) & nchar(key) >= 3,
          .(unit_id, name, key, lon, lat, feature_id, subdistrict_name, shrid2_own, osm_tag)]
gp <- unique(gp, by = c("feature_id", "key"))[, level := "place"]
gs <- geo[feature_class == "A" & feature_code %in% c("ADM3", "ADM4", "ADM5") &
            nzchar(key) & nchar(key) >= 3,
          .(unit_id, name, key, lon, lat, feature_id, subdistrict_name, shrid2_own, osm_tag)]
gs <- unique(gs, by = c("feature_id", "key"))[, level := "subdistrict"]
gf <- geo[feature_class %in% c("T", "L", "H", "V", "R") & nzchar(key) & nchar(key) >= 3,
          .(unit_id, name, key, lon, lat, feature_id, subdistrict_name, shrid2_own, osm_tag)]
gf <- unique(gf, by = c("feature_id", "key"))[, level := "feature"]

osm <- as.data.table(readRDS(file.path(root, "data/osm_gazetteer_india.rds")))
osm <- osm[(class == "admin" | is.na(size_km) | size_km <= 10) &
             !is.na(unit_id) & !is.na(lon) & !is.na(lat)]
osm_names <- unique(osm$name)
osm_key <- data.table(name = osm_names)[, key := norm(name)]
osm[, key := osm_key$key[match(name, osm_names)]]
osm[, `:=`(feature_id = paste(osm_type, osm_id, sep = ":"),
           subdistrict_name = NA_character_, shrid2_own = NA_character_)]

op <- osm[class %in% c("settlement", "security") & nzchar(key) & nchar(key) >= 3,
          .(unit_id, name, key, lon, lat, feature_id, subdistrict_name, shrid2_own, osm_tag)]
op <- unique(op, by = c("feature_id", "key"))[, level := "place"]
oa <- osm[class == "admin" & admin_level %in% 6:8 & nzchar(key) & nchar(key) >= 3,
          .(unit_id, name, key, lon, lat, feature_id, subdistrict_name, shrid2_own, osm_tag)]
oa <- unique(oa, by = c("feature_id", "key"))[, level := "subdistrict"]
of <- osm[class %in% c("forest_terrain", "water", "road") & nzchar(key) & nchar(key) >= 3,
          .(unit_id, name, key, lon, lat, feature_id, subdistrict_name, shrid2_own, osm_tag)]
of <- unique(of, by = c("feature_id", "key"))[, level := "feature"]

tabs <- list(shrid_place = sp, shrid_sub = ss, geo_place = gp, geo_sub = gs,
             geo_feat = gf, osm_place = op, osm_admin = oa, osm_feat = of)
cat("gazetteer rows:", paste(names(tabs), sapply(tabs, nrow), sep = "=", collapse = ", "), "\n")

states <- unique(units_dt$state_name)
state_tabs <- lapply(setNames(states, states), function(st) {
  u <- units_dt[state_name == st, unit_id]
  lapply(tabs, function(t) t[unit_id %in% u])
})

teh <- st_read(file.path(root, "data/osm_admin_boundaries_india.gpkg"),
               layer = "admin_level_6", quiet = TRUE)
teh <- st_make_valid(teh)[, c("osm_id", "name")]

tehsil_lookup <- function(lon, lat) {
  if (!length(lon)) return(data.table(osm_id = character(), name = character()))
  p <- st_as_sf(data.table(lon = lon, lat = lat), coords = c("lon", "lat"), crs = 4326)
  idx <- st_intersects(p, teh)
  data.table(osm_id = vapply(idx, function(i) if (length(i)) teh$osm_id[i[1]] else NA_character_, character(1)),
             name = vapply(idx, function(i) if (length(i)) teh$name[i[1]] else NA_character_, character(1)))
}

#------------------------------------------------------------------------------#
####                         4. Matching functions                          ####
#------------------------------------------------------------------------------#
source_tables <- function(type, source) {
  if (source == "SHRID") {
    if (type %in% c("village", "town", "locality", "camp")) return("shrid_place")
    if (type == "subdistrict") return("shrid_sub")
    if (type == "police_station") return(c("shrid_sub", "shrid_place"))
    return(character(0))
  }
  if (source == "GeoNames") {
    if (type %in% c("village", "town", "locality", "camp")) return("geo_place")
    if (type == "subdistrict") return("geo_sub")
    if (type == "police_station") return(c("geo_sub", "geo_place"))
    if (type %in% c("forest", "water", "road")) return("geo_feat")
    return(character(0))
  }
  if (source == "OSM") {
    if (type %in% c("village", "town", "locality", "camp")) return("osm_place")
    if (type == "subdistrict") return("osm_admin")
    if (type == "police_station") return(c("osm_admin", "osm_place"))
    if (type %in% c("forest", "water", "road")) return("osm_feat")
    return(character(0))
  }
  character(0)
}

match_hits <- function(keyq, cand, within_key, ref_lon, ref_lat, allow_fuzzy = TRUE) {
  if (nrow(cand) == 0 || nchar(keyq) < 3) return(NULL)
  hit <- cand[key == keyq]
  how <- "exact"; jw <- 0
  if (nrow(hit) == 0 && allow_fuzzy && nchar(keyq) >= 5) {
    cand2 <- cand[substr(key, 1, 1) == substr(keyq, 1, 1) & nchar(key) >= 3]
    if (nrow(cand2)) {
      d <- stringdist(keyq, cand2$key, method = "jw", p = 0.1)
      ok <- is.finite(d) & d <= 0.10
      if (any(ok)) {
        m <- min(d[ok])
        hit <- cand2[ok & d == m]
        how <- "fuzzy"; jw <- m
      }
    }
  }
  if (nrow(hit) == 0) return(NULL)

  hit <- copy(hit)
  hit[, jw_dist := if (how == "exact") 0 else jw]
  n_before <- uniqueN(hit$feature_id)
  within_resolved <- FALSE
  if (nzchar(within_key) && n_before > 1) {
    sub_ok <- !is.na(hit$subdistrict_name) & norm(hit$subdistrict_name) == within_key
    teh_ok <- rep(FALSE, nrow(hit))
    need <- !sub_ok
    if (any(need)) {
      tl <- tehsil_lookup(hit$lon[need], hit$lat[need])
      teh_ok[need] <- !is.na(tl$name) & norm(tl$name) == within_key
    }
    keep <- sub_ok | teh_ok
    if (any(keep)) {
      hit2 <- hit[keep]
      if (uniqueN(hit2$feature_id) < n_before) {
        hit <- hit2
        within_resolved <- TRUE
      }
    }
  }

  j <- 1L
  if (!is.na(ref_lon) && !is.na(ref_lat)) {
    dd <- haversine(ref_lon, ref_lat, hit$lon, hit$lat)
    if (any(!is.na(dd))) j <- which.min(dd)
  }
  sh <- unique(hit$shrid2_own[!is.na(hit$shrid2_own)])
  len_ratio <- if (how == "exact") NA_real_ else
    min(nchar(keyq), nchar(hit$key[j])) / max(nchar(keyq), nchar(hit$key[j]))
  list(hits = hit, chosen = hit[j], match = how, jw_dist = hit$jw_dist[j],
       len_ratio = len_ratio, n_candidates = uniqueN(hit$feature_id),
       shrid2 = if (length(sh)) paste(sh, collapse = ";") else NA_character_,
       level = hit$level[j], gaz_tag = hit$osm_tag[j],
       subdistrict = hit$subdistrict_name[j],
       matched_unit_id = hit$unit_id[j], within_resolved = within_resolved)
}

match_in_unit <- function(keyq, unit, tnames, within_key, ref_lon, ref_lat) {
  for (nm in tnames) {
    r <- match_hits(keyq, tabs[[nm]][unit_id == unit], within_key, ref_lon, ref_lat, TRUE)
    if (!is.null(r)) return(r)
  }
  NULL
}

match_outside <- function(keyq, unit, state, tnames, within_key, ref_lon, ref_lat) {
  for (nm in tnames) {
    cand <- state_tabs[[state]][[nm]][unit_id != unit]
    r <- match_hits(keyq, cand, within_key, ref_lon, ref_lat, FALSE)
    if (!is.null(r)) return(r)
  }
  NULL
}

#------------------------------------------------------------------------------#
####                    5. Places and A1 district flags                     ####
#------------------------------------------------------------------------------#
places[, key := norm(name)]
places <- places[nzchar(key)]
places <- merge(places, ev[, .(event_key, state_key)], by = "event_key", all.x = TRUE)
places[, role_rank := fifelse(role == "site", 0L, 1L)]
setorder(places, event_key, key, type, role_rank, place_order)
places <- unique(places, by = c("event_key", "key", "type"))
setorder(places, event_key, place_order)
places[, spec := fcase(
  type %in% c("village", "town"), 1L,
  type %in% c("locality", "camp", "forest", "water", "road"), 2L,
  type %in% c("subdistrict", "police_station"), 3L,
  default = 4L)]

places[, dist_old := mapply(function(s, k) {
  z <- old_list[[s]]
  length(z) > 0 && k %in% z
}, state_key, key)]
places[, dist_expanded := mapply(function(s, k) {
  z <- exp_list[[s]]
  if (is.null(z) || !length(z)) return(FALSE)
  k %in% z || min(stringdist(k, z, method = "jw", p = 0.1)) <= 0.10
}, state_key, key)]
places[, district_hq_name := type == "town" & dist_expanded]
places[, `:=`(keep_before = !dist_old, keep_after = !dist_expanded | type == "town")]

cat("places before district filters:", nrow(places), "\n")
cat("A1 dropped places:", places[keep_before & !keep_after, .N],
    " events:", uniqueN(places[keep_before & !keep_after, event_id_cnty]), "\n")
cat("A1 flagged towns:", places[district_hq_name == TRUE, .N],
    " events:", uniqueN(places[district_hq_name == TRUE, event_id_cnty]), "\n")

srcnames <- c("SHRID", "GeoNames", "OSM")
cand <- rbindlist(lapply(seq_len(nrow(places)), function(i) {
  p <- places[i]
  e <- ev[match(p$event_key, ev$event_key)]
  wk <- if (!is.na(p$within) && nzchar(trimws(p$within))) norm(p$within) else ""
  rbindlist(lapply(srcnames, function(src) {
    tn <- source_tables(p$type, src)
    r <- NULL; outside <- FALSE
    if (length(tn)) {
      r <- match_in_unit(p$key, p$unit_id, tn, wk, e$unit_lon, e$unit_lat)
      if (is.null(r)) {
        r <- match_outside(p$key, p$unit_id, e$state_name, tn, wk, e$unit_lon, e$unit_lat)
        outside <- !is.null(r)
      }
    }
    data.table(event_key = p$event_key, event_id_cnty = p$event_id_cnty,
               series = p$series, unit_id = p$unit_id, state_name = e$state_name,
               place_order = p$place_order, extracted = p$name, type = p$type,
               role = p$role, within = p$within, source = src,
               gazetteer = if (is.null(r)) NA_character_ else r$chosen$name,
               match = if (is.null(r)) NA_character_ else r$match,
               jw_dist = if (is.null(r)) NA_real_ else r$jw_dist,
               len_ratio = if (is.null(r)) NA_real_ else r$len_ratio,
               n_candidates = if (is.null(r)) 0L else as.integer(r$n_candidates),
               lon = if (is.null(r)) NA_real_ else r$chosen$lon,
               lat = if (is.null(r)) NA_real_ else r$chosen$lat,
               level = if (is.null(r)) NA_character_ else r$level,
               shrid2 = if (is.null(r)) NA_character_ else r$shrid2,
               osm_tag = if (is.null(r)) NA_character_ else r$gaz_tag,
               matched_unit_id = if (is.null(r)) NA_character_ else r$matched_unit_id,
               within_resolved = if (is.null(r)) FALSE else r$within_resolved,
               outside_unit = outside,
               keep_before = p$keep_before, keep_after = p$keep_after,
               district_hq_name = p$district_hq_name)
  }))
}), fill = TRUE)
cat("candidate rows (place x source):", nrow(cand), "\n")
cat("within-resolved place x source cases:", cand[within_resolved == TRUE, .N], "\n")

#------------------------------------------------------------------------------#
####                    6. Tehsil and SHRID assignment                      ####
#------------------------------------------------------------------------------#
pts <- unique(cand[!is.na(lon) & !is.na(lat), .(lon, lat)])
pts_sf <- st_as_sf(pts, coords = c("lon", "lat"), crs = 4326)
idx <- st_intersects(pts_sf, teh)
pts[, `:=`(tehsil_osm_id = vapply(idx, function(i) if (length(i)) teh$osm_id[i[1]] else NA_character_, character(1)),
           tehsil_name = vapply(idx, function(i) if (length(i)) teh$name[i[1]] else NA_character_, character(1)))]
cand <- merge(cand, pts, by = c("lon", "lat"), all.x = TRUE)

shp_path <- file.path(shrug_dir, "shrid2_open.shp")
if (!file.exists(shp_path)) {
  unzip(file.path(shrug_pub, "shrug-shrid-poly-shp.zip"), exdir = shrug_dir)
}
shrid_poly <- st_read(shp_path, quiet = TRUE)[, "shrid2"]
shrid_poly$shrid2 <- trimws(shrid_poly$shrid2)

assign_shrid <- function(pts) {
  out <- data.table(lon = pts$lon, lat = pts$lat,
                    shrid2_poly = NA_character_, shrid_snap_poly = NA)
  if (!nrow(pts)) return(out)
  q <- st_as_sf(pts, coords = c("lon", "lat"), crs = 4326)
  sf_use_s2(FALSE)
  inside <- st_intersects(q, shrid_poly)
  if (any(lengths(inside) > 0)) {
    k <- which(lengths(inside) > 0)
    out$shrid2_poly[k] <- vapply(inside[k], function(i) shrid_poly$shrid2[i[1]], character(1))
    out$shrid_snap_poly[k] <- FALSE
  }
  miss <- lengths(inside) == 0
  if (any(miss)) {
    j <- st_nearest_feature(q[miss, ], shrid_poly)
    sf_use_s2(TRUE)
    pp <- st_make_valid(shrid_poly[j, ])
    d <- st_distance(q[miss, ], pp, by_element = TRUE)
    ok <- as.numeric(d) <= 2000
    out$shrid2_poly[which(miss)[ok]] <- pp$shrid2[ok]
    out$shrid_snap_poly[which(miss)] <- ok
  }
  sf_use_s2(TRUE)
  out
}

pts2 <- unique(cand[source %in% c("GeoNames", "OSM") & !is.na(lon) & !is.na(lat), .(lon, lat)])
sh <- assign_shrid(pts2)
cand <- merge(cand, sh, by = c("lon", "lat"), all.x = TRUE)
cand[source %in% c("GeoNames", "OSM"), `:=`(shrid2 = shrid2_poly, shrid_snap = shrid_snap_poly)]
cand[source == "SHRID", shrid_snap := FALSE]
cand[, c("shrid2_poly", "shrid_snap_poly") := NULL]
cat("GeoNames/OSM candidates with a SHRID:",
    cand[source %in% c("GeoNames", "OSM") & !is.na(gazetteer) & !is.na(shrid2), .N],
    " snapped:", cand[source %in% c("GeoNames", "OSM") & shrid_snap %in% TRUE, .N], "\n")

# A5: a candidate matched at sub-district level (SHRID sub-district, GeoNames
# ADM3+, OSM admin 6-8) is a tehsil, not a village. It must not get a village
# shrid2 from its representative point. Keep a pre-A5 copy for the change count.
cand_pre_a5 <- copy(cand)
a5_removed <- cand[level == "subdistrict" & !is.na(shrid2), .N]
cand[level == "subdistrict", `:=`(shrid2 = NA_character_, shrid_snap = NA)]
cat("A5: removed SHRID from", a5_removed, "subdistrict candidate rows\n")

#------------------------------------------------------------------------------#
####                        7. 2001 census village                         ####
#------------------------------------------------------------------------------#
pc_dir <- file.path(tempdir(), "shrug_pc")
dir.create(pc_dir, showWarnings = FALSE)
pc_zip <- file.path(shrug_pub, "shrug-pc-keys-dta.zip")
unzip(pc_zip, files = c("pc01r_shrid_key.dta", "pc01u_shrid_key.dta"), exdir = pc_dir)
pc_r <- as.data.table(read_dta(file.path(pc_dir, "pc01r_shrid_key.dta")))
pc_u <- as.data.table(read_dta(file.path(pc_dir, "pc01u_shrid_key.dta")))

pc01_map <- unique(rbind(
  pc_r[, .(shrid2, code = paste(pc01_state_id, pc01_district_id,
                                pc01_subdistrict_id, pc01_village_id, sep = "-"),
           kind = "rural")],
  pc_u[, .(shrid2, code = as.character(pc01_town_id), kind = "urban")]
))
pc01_map <- pc01_map[!is.na(code) & nzchar(code)]
pc01_codes <- function(ids) {
  if (is.na(ids) || !nzchar(ids)) return(list(code = NA_character_, n = 0L))
  z <- unlist(strsplit(ids, ";", fixed = TRUE))
  codes <- unique(pc01_map[shrid2 %in% z, code])
  list(code = if (length(codes)) paste(codes, collapse = ";") else NA_character_,
       n = length(codes))
}
shrid_pc <- pc01_map[, .(shrid_n_pc01 = uniqueN(code),
                         n_villages = uniqueN(code[kind == "rural"]),
                         has_town = any(kind == "urban")), by = shrid2]
shrid_pc[, shrid_urban := has_town | n_villages > 5]

#------------------------------------------------------------------------------#
####                      8. Place-level comparison                        ####
#------------------------------------------------------------------------------#
place_tier <- function(d, type_group) {
  n <- nrow(d)
  if (n == 0) {
    return(list(n = 0, n_agree = 0, same_tehsil = FALSE, same_shrid = FALSE,
                shrid2_final = NA_character_, tier = "none",
                tier_A1 = "none", tier_A2 = "none", fuzzy_support = FALSE))
  }
  val <- function(src, col) {
    z <- d[source == src, get(col)]
    if (length(z)) z[1] else NA
  }
  lon_sg <- val("SHRID", "lon"); lat_sg <- val("SHRID", "lat")
  lon_go <- val("GeoNames", "lon"); lat_go <- val("GeoNames", "lat")
  lon_os <- val("OSM", "lon"); lat_os <- val("OSM", "lat")
  teh_sg <- val("SHRID", "tehsil_osm_id"); teh_go <- val("GeoNames", "tehsil_osm_id")
  teh_os <- val("OSM", "tehsil_osm_id")
  gaz_sg <- val("SHRID", "gazetteer"); gaz_go <- val("GeoNames", "gazetteer")
  gaz_os <- val("OSM", "gazetteer")

  d_sg <- haversine(lon_sg, lat_sg, lon_go, lat_go)
  d_so <- haversine(lon_sg, lat_sg, lon_os, lat_os)
  d_go <- haversine(lon_go, lat_go, lon_os, lat_os)
  dd <- c(d_sg, d_so, d_go); dd <- dd[!is.na(dd)]
  n_agree_5km <- if (n <= 1) n else if (n == 2) fifelse(any(dd <= 5), 2L, 1L) else
    fifelse(all(dd[1:3] <= 5), 3L, fifelse(any(dd <= 5), 2L, 1L))

  tg <- c(teh_sg, teh_go, teh_os); gn <- c(gaz_sg, gaz_go, gaz_os)
  idx <- which(!is.na(c(lon_sg, lon_go, lon_os)))
  edge <- function(a, b) {
    if (a == b) return(TRUE)
    same_name <- !is.na(gn[a]) && !is.na(gn[b]) && nzchar(gn[a]) && nzchar(gn[b]) &&
      norm(gn[a]) == norm(gn[b])
    same_teh <- !is.na(tg[a]) && !is.na(tg[b]) && tg[a] == tg[b]
    isTRUE(same_name || same_teh)
  }
  n_agree_sub <- if (n <= 1) n else if (n == 2) fifelse(edge(idx[1], idx[2]), 2L, 1L) else {
    e12 <- edge(idx[1], idx[2]); e13 <- edge(idx[1], idx[3]); e23 <- edge(idx[2], idx[3])
    if (e12 && e13 && e23) 3L else if (e12 || e13 || e23) 2L else 1L
  }
  tehs <- tg[idx]
  same_tehsil <- n >= 2 && all(!is.na(tehs)) && length(unique(tehs)) == 1

  sets <- lapply(seq_len(n), function(j) {
    x <- d$shrid2[j]
    if (is.na(x) || !nzchar(x)) character(0) else strsplit(x, ";", fixed = TRUE)[[1]]
  })
  all_sh <- unlist(sets)
  tab <- table(all_sh)
  shared <- names(tab)[tab >= 2]
  same_shrid <- length(shared) > 0
  shrid2_final <- if (same_shrid) shared[which.max(tab[shared])] else {
    ord <- order(match(d$source, c("SHRID", "GeoNames", "OSM")))
    first_sh <- d$shrid2[ord][!is.na(d$shrid2[ord])]
    if (length(first_sh)) strsplit(first_sh[1], ";", fixed = TRUE)[[1]][1] else NA_character_
  }

  single_tier <- if (n == 1 && d$match[1] == "fuzzy") "single_fuzzy" else "single"
  tier_A1 <- if (n_agree_5km >= 2) "high" else if (n == 1) single_tier else
    if (same_tehsil) "tehsil" else "conflict"
  tier_A2 <- if (type_group == "sub") {
    if (n_agree_sub >= 2) "high" else if (n == 1) single_tier else "conflict"
  } else tier_A1
  tier <- if (same_shrid) "shrid" else tier_A2
  n_agree <- if (type_group == "sub") n_agree_sub else n_agree_5km

  list(n = n, n_agree = n_agree, same_tehsil = same_tehsil,
       same_shrid = same_shrid, shrid2_final = shrid2_final,
       tier = tier, tier_A1 = tier_A1, tier_A2 = tier_A2,
       fuzzy_support = any(d$match == "fuzzy"))
}

exact_consistent <- function(d, type_group) {
  n <- nrow(d)
  if (n <= 1) return(TRUE)
  sets <- lapply(d$shrid2, function(x) {
    if (is.na(x) || !nzchar(x)) character(0) else strsplit(x, ";", fixed = TRUE)[[1]]
  })
  all_sh <- unlist(sets)
  if (length(all_sh) && any(table(all_sh) >= 2)) return(TRUE)
  if (type_group == "sub") {
    gn <- d$gazetteer
    same_name <- all(!is.na(gn)) && length(unique(norm(gn))) == 1
    tg <- d$tehsil_osm_id
    same_teh <- all(!is.na(tg)) && length(unique(tg)) == 1
    return(same_name || same_teh)
  }
  dd <- numeric(0)
  for (i in seq_len(n - 1)) for (j in (i + 1):n)
    dd <- c(dd, haversine(d$lon[i], d$lat[i], d$lon[j], d$lat[j]))
  length(dd) > 0 && all(dd <= 5)
}

compute_metrics <- function(cand_sub, places_sub, exact_override) {
  cand_in <- cand_sub[outside_unit == FALSE]
  rbindlist(lapply(seq_len(nrow(places_sub)), function(i) {
    pk <- places_sub[i]
    d <- cand_in[event_key == pk$event_key & place_order == pk$place_order]
    present <- d[!is.na(lon)]
    type_group <- if (pk$type %in% c("subdistrict", "police_station")) "sub" else "point"
    all_pt <- place_tier(present, type_group)
    exact <- present[match == "exact"]
    fuzzy <- present[match == "fuzzy"]
    override <- exact_override && nrow(exact) > 0 && nrow(fuzzy) > 0 &&
      exact_consistent(exact, type_group)
    used <- if (override) exact else present
    pt <- place_tier(used, type_group)
    conflict_resolved_exact <- override && all_pt$tier == "conflict" && pt$tier != "conflict"
    data.table(event_key = pk$event_key, place_order = pk$place_order,
               extracted = pk$name, type = pk$type, role = pk$role,
               role_rank = pk$role_rank, spec = pk$spec,
               n_sources = pt$n, n_sources_agree = pt$n_agree,
               same_tehsil = pt$same_tehsil, same_shrid = pt$same_shrid,
               shrid2_final = pt$shrid2_final,
               tier = pt$tier, tier_A1 = pt$tier_A1, tier_A2 = pt$tier_A2,
               tier_all = all_pt$tier,
               fuzzy_support = any(present$match == "fuzzy"),
               district_hq_name = pk$district_hq_name,
               exact_override = override,
               conflict_resolved_exact = conflict_resolved_exact)
  }), fill = TRUE)
}

select_event <- function(metrics) {
  sel <- metrics[n_sources > 0][order(event_key, spec, role_rank, place_order)][, .SD[1], by = event_key]
  ms <- merge(ev[, .(event_key, event_id_cnty, series, unit_id, state_name, notes, state_key)],
              sel, by = "event_key", all.x = TRUE)
  ms[is.na(tier), `:=`(tier = "none", tier_A1 = "none", tier_A2 = "none",
                       n_sources = 0L, n_sources_agree = 0L,
                       same_tehsil = FALSE, same_shrid = FALSE,
                       fuzzy_support = FALSE, district_hq_name = FALSE,
                       exact_override = FALSE, conflict_resolved_exact = FALSE)]
  ms
}

places_before <- places[keep_before == TRUE]
places_after <- places[keep_after == TRUE]
cand_before <- cand_pre_a5[keep_before == TRUE]
cand_after_pre_a5 <- cand_pre_a5[keep_after == TRUE]
cand_after <- cand[keep_after == TRUE]

metrics_before <- compute_metrics(cand_before, places_before, exact_override = FALSE)
metrics_after_pre_a5 <- compute_metrics(cand_after_pre_a5, places_after, exact_override = TRUE)
metrics_after <- compute_metrics(cand_after, places_after, exact_override = TRUE)
ms_before <- select_event(metrics_before)
ms_after_pre_a5 <- select_event(metrics_after_pre_a5)
ms_after <- select_event(metrics_after)

ms_after[, tier_before := ms_before$tier[match(event_key, ms_before$event_key)]]
ms_after[is.na(tier_before), tier_before := "none"]
ms_after[, `:=`(tier_pre_a5 = ms_after_pre_a5$tier[match(event_key, ms_after_pre_a5$event_key)],
                shrid_pre_a5 = ms_after_pre_a5$shrid2_final[match(event_key, ms_after_pre_a5$event_key)])]
ms_after[is.na(tier_pre_a5), tier_pre_a5 := "none"]
ms_after[, a5_changed := (tier != tier_pre_a5) |
  (is.na(shrid2_final) != is.na(shrid_pre_a5)) |
  (!is.na(shrid2_final) & !is.na(shrid_pre_a5) & shrid2_final != shrid_pre_a5)]
ms_after[, a5_shrid_lost := !is.na(shrid_pre_a5) & is.na(shrid2_final)]
ms_after <- merge(ms_after, shrid_pc[, .(shrid2_final = shrid2, shrid_n_pc01, shrid_urban)],
                  by = "shrid2_final", all.x = TRUE)
ms_after[, pc01 := lapply(shrid2_final, pc01_codes)]
ms_after[, `:=`(pc01_village_code = vapply(pc01, function(z) z$code, character(1)),
                n_pc01 = vapply(pc01, function(z) z$n, integer(1)))]
ms_after[, pc01 := NULL]

# Selected place source columns (after fixes).
sel_after <- ms_after[, .(event_key, place_order)]
cand_sel_after <- merge(sel_after, cand_after, by = c("event_key", "place_order"), all.x = TRUE)

#------------------------------------------------------------------------------#
####                           9. Outputs                                   ####
#------------------------------------------------------------------------------#
cand_out <- cand[, .(event_id_cnty, extracted, type, role, source, gazetteer, match,
                     jw_dist, len_ratio, n_candidates, lon, lat, shrid2, shrid_snap,
                     tehsil_osm_id, tehsil_name, outside_unit, within_resolved,
                     district_hq_name)]
fwrite(cand_out, file.path(out_dir, "place_pilot_candidates.csv"), na = "NA")

wide_source <- function(src, prefix) {
  b <- cand_sel_after[source == src, .(event_key, extracted, gazetteer, match, jw_dist,
                                       len_ratio, lon, lat, tehsil_osm_id, tehsil_name,
                                       shrid2, shrid_snap)]
  setnames(b, c("extracted", "gazetteer", "match", "jw_dist", "len_ratio", "lon", "lat",
                "tehsil_osm_id", "tehsil_name", "shrid2", "shrid_snap"),
           paste0(prefix, c("_extracted", "_gazetteer", "_match", "_jw_dist",
                            "_len_ratio", "_lon", "_lat", "_tehsil_osm_id",
                            "_tehsil_name", "_shrid2", "_shrid_snap")))
  b
}
ms <- ms_after
ms <- merge(ms, wide_source("SHRID", "shrid"), by = "event_key", all.x = TRUE)
ms <- merge(ms, wide_source("GeoNames", "geonames"), by = "event_key", all.x = TRUE)
ms <- merge(ms, wide_source("OSM", "osm"), by = "event_key", all.x = TRUE)

ms_out <- ms[, .(event_id_cnty, event_key, series, unit_id, state_name,
                 extracted, type, role, place_order,
                 n_sources, n_sources_agree, same_tehsil, same_shrid,
                 shrid2_final, pc01_village_code, n_pc01, shrid_n_pc01, shrid_urban,
                 tier, tier_before, tier_pre_a5, fuzzy_support, district_hq_name,
                 exact_override, conflict_resolved_exact, a5_changed, a5_shrid_lost,
                 shrid_extracted, shrid_gazetteer, shrid_match, shrid_jw_dist, shrid_len_ratio,
                 shrid_lon, shrid_lat, shrid_tehsil_osm_id, shrid_tehsil_name,
                 shrid_shrid2, shrid_shrid_snap,
                 geonames_extracted, geonames_gazetteer, geonames_match, geonames_jw_dist,
                 geonames_len_ratio, geonames_lon, geonames_lat,
                 geonames_tehsil_osm_id, geonames_tehsil_name, geonames_shrid2, geonames_shrid_snap,
                 osm_extracted, osm_gazetteer, osm_match, osm_jw_dist, osm_len_ratio,
                 osm_lon, osm_lat, osm_tehsil_osm_id, osm_tehsil_name,
                 osm_shrid2, osm_shrid_snap)]
fwrite(ms_out, file.path(out_dir, "place_pilot_multisource.csv"), na = "NA")
cat("saved", file.path(out_dir, "place_pilot_candidates.csv"), "and",
    file.path(out_dir, "place_pilot_multisource.csv"), "\n")

#------------------------------------------------------------------------------#
####                           10. Report                                   ####
#------------------------------------------------------------------------------#
tier_levels <- c("shrid", "high", "tehsil", "single", "single_fuzzy", "conflict", "none")
fmt_tier <- function(dt, col, label) {
  z <- dcast(dt[, .N, by = c("series", col)], as.formula(paste("series ~", col)),
             value.var = "N", fill = 0)
  z[, label := label]
  z
}
tab <- rbind(fmt_tier(ms, "tier_before", "before"),
             fmt_tier(ms, "tier_pre_a5", "after_A1_A4"),
             fmt_tier(ms, "tier", "after_A1_A5"), fill = TRUE)
for (cc in setdiff(tier_levels, names(tab))) tab[, (cc) := 0L]
for (cc in intersect(tier_levels, names(tab))) set(tab, which(is.na(tab[[cc]])), cc, 0L)
setcolorder(tab, c("label", "series", intersect(tier_levels, names(tab))))
cat("\n=== tier counts by series: before and after the fixes ===\n")
print(tab[order(label, series)])

cat("\n=== A5: subdistrict candidates no longer set a village SHRID ===\n")
cat("candidate rows stripped of shrid2:", a5_removed, "\n")
cat("events changed:", sum(ms$a5_changed),
    "| tier changed:", sum(ms$tier != ms$tier_pre_a5),
    "| SHRID lost:", sum(ms$a5_shrid_lost), "\n")
print(ms[, .(events = .N, changed = sum(a5_changed),
             tier_changed = sum(tier != tier_pre_a5),
             shrid_lost = sum(a5_shrid_lost)), by = series][order(-changed)])

cat("\n=== conflicts resolved by A2 (exact over fuzzy) ===\n")
red <- ms[, .(conflicts_before = sum(tier_before == "conflict"),
              conflicts_after = sum(tier == "conflict"),
              resolved_by_A2 = sum(conflict_resolved_exact == TRUE),
              exact_overrides = sum(exact_override == TRUE)), by = series]
all_red <- ms[, .(series = "All",
                  conflicts_before = sum(tier_before == "conflict"),
                  conflicts_after = sum(tier == "conflict"),
                  resolved_by_A2 = sum(conflict_resolved_exact == TRUE),
                  exact_overrides = sum(exact_override == TRUE))]
print(rbind(red, all_red, fill = TRUE))

cat("\n=== A1 district filter ===\n")
cat("dropped places:", places[keep_before & !keep_after, .N],
    " events:", uniqueN(places[keep_before & !keep_after, event_id_cnty]), "\n")
cat("flagged towns:", places[district_hq_name == TRUE, .N],
    " events:", uniqueN(places[district_hq_name == TRUE, event_id_cnty]), "\n")
print(places[keep_before & !keep_after, .N, by = series][order(-N)])
print(places[district_hq_name == TRUE, .N, by = series][order(-N)])

cat("\n=== len_ratio distribution for fuzzy matches by tier ===\n")
lr <- merge(cand_sel_after[!is.na(gazetteer) & match == "fuzzy",
                           .(event_key, source, len_ratio)],
            ms[, .(event_key, tier)], by = "event_key")
lr_tab <- lr[, .(n = .N, min = min(len_ratio), p25 = quantile(len_ratio, 0.25),
                 median = median(len_ratio), p75 = quantile(len_ratio, 0.75),
                 max = max(len_ratio), mean = mean(len_ratio)), by = tier]
lr_tab <- lr_tab[order(match(tier, tier_levels))]
print(lr_tab[, lapply(.SD, function(z) if (is.numeric(z)) round(z, 3) else z), by = tier])
cat("all fuzzy matches:", nrow(lr), " median len_ratio:",
    round(median(lr$len_ratio, na.rm = TRUE), 3), "\n")
