#------------------------------------------------------------------------------#
### SATP: OpenStreetMap gazetteer for India
#
# Builds a searchable place-name gazetteer for India from the Geofabrik OSM
# extract, with one representative point per feature and the 2001 district unit
# it falls in. Also writes the administrative boundaries used below as polygons.
#
# Why GDAL directly: osmium-tool cannot be installed on this machine (Homebrew
# is not writable for the user) and osmextract would first materialise a
# multi-GB GeoPackage of every OSM layer. The script drives ogr2ogr instead:
# one pass per layer (points, lines, multipolygons), tag filters and the
# representative point (ST_PointOnSurface) pushed into SQL, so only the wanted
# features are ever written to disk. Sizes are computed in SQL with SpatiaLite's
# ellipsoidal ST_Length/ST_Area, so the full geometries are never loaded in R.
#
# Classes kept (named features only, name, name:en, alt_name or old_name):
#   admin           multipolygons with boundary=administrative and admin_level
#                   6, 7, 8, 9 or 10 (tehsil/block level and below)
#   settlement      place=city/town/village/hamlet/suburb/neighbourhood/
#                   locality/isolated_dwelling
#   forest_terrain  landuse=forest; natural=wood/peak/ridge/valley;
#                   boundary=protected_area
#   water           waterway=river/stream; natural=water
#   road            highway=primary/secondary/tertiary/unclassified/track;
#                   railway=rail
#   security        amenity=police; landuse=military; military=*
# A feature that fits several classes takes the first class in this list.
#
# Name variants: name, name:en, alt_name and old_name, each as its own row;
# alt_name and old_name are split on ';' because they often list several names.
# Every non-ASCII string also gets a Latin-ASCII transliteration row
# (name_variant = "translit") when it differs, unless that string is already
# stored for the same feature.
#
# size_km: length in km for line features, sqrt(area in km2) for polygons, 0 for
# nodes.
#
# Output:  data/osm_gazetteer_india.rds  columns osm_id, osm_type, name,
#          name_variant, class, osm_tag, admin_level, size_km, lon, lat, unit_id
#          data/osm_admin_boundaries_india.gpkg, one layer per admin_level
#          (admin_level_6 ... admin_level_10), columns osm_id, osm_type, name,
#          name variants (name_en, alt_name, old_name), admin_level, geometry
#
# Input:   data/raw/osm/india-latest.osm.pbf   (Geofabrik, see the README there)
#          irrigating_peace/mattia/data/intermediate/restricted/india_units_2001.gpkg
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(sf)
  library(data.table)
  library(stringi)
})

root <- Sys.getenv("SATP_ROOT", unset = "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
pbf  <- Sys.getenv("SATP_OSM_PBF", unset = file.path(root, "data/raw/osm/india-latest.osm.pbf"))
gpkg_units <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/india_units_2001.gpkg"
out_rds <- file.path(root, "data/osm_gazetteer_india.rds")
out_admin <- file.path(root, "data/osm_admin_boundaries_india.gpkg")
stopifnot(file.exists(pbf))

work <- file.path(tempdir(), "osm_gazetteer")
dir.create(work, showWarnings = FALSE)
conf <- file.path(work, "osmconf.ini")

#------------------------------------------------------------------------------#
####                        OSM driver configuration                        ####
#------------------------------------------------------------------------------#
# Based on GDAL's osmconf.ini: only the tags below become fields, and
# other_tags is switched off to keep the passes light.
writeLines(c(
  "[general]",
  "closed_ways_are_polygons=aeroway,amenity,boundary,building,craft,geological,historic,landuse,leisure,military,natural,office,place,shop,sport,tourism,highway=platform,public_transport=platform",
  "attribute_name_laundering=yes",
  "",
  "[points]",
  "osm_id=yes", "osm_version=no", "osm_timestamp=no", "osm_uid=no", "osm_user=no", "osm_changeset=no",
  "attributes=name,name:en,alt_name,old_name,place,natural,landuse,waterway,highway,railway,amenity,military,boundary,admin_level",
  "other_tags=no",
  "",
  "[lines]",
  "osm_id=yes", "osm_version=no", "osm_timestamp=no", "osm_uid=no", "osm_user=no", "osm_changeset=no",
  "attributes=name,name:en,alt_name,old_name,place,natural,landuse,waterway,highway,railway,amenity,military,boundary,admin_level",
  "other_tags=no",
  "",
  "[multipolygons]",
  "osm_id=yes", "osm_way_id=yes", "osm_version=no", "osm_timestamp=no", "osm_uid=no", "osm_user=no", "osm_changeset=no",
  "attributes=name,name:en,alt_name,old_name,place,natural,landuse,waterway,highway,railway,amenity,military,boundary,admin_level",
  "other_tags=no"
), conf)

#------------------------------------------------------------------------------#
####                            Tag selection                              ####
#------------------------------------------------------------------------------#
has_name <- "(name IS NOT NULL OR name_en IS NOT NULL OR alt_name IS NOT NULL OR old_name IS NOT NULL)"
in_class <- paste0("(",
  "boundary = 'administrative' AND admin_level IN ('6','7','8','9','10')",
  " OR place IN ('city','town','village','hamlet','suburb','neighbourhood','locality','isolated_dwelling')",
  " OR landuse IN ('forest','military')",
  " OR natural IN ('wood','peak','ridge','valley','water')",
  " OR boundary = 'protected_area'",
  " OR waterway IN ('river','stream')",
  " OR highway IN ('primary','secondary','tertiary','unclassified','track')",
  " OR railway = 'rail'",
  " OR amenity = 'police'",
  " OR military IS NOT NULL",
  ")")
where <- paste(has_name, "AND", in_class)

layers <- c("points", "lines", "multipolygons")
fetch <- function(layer) {
  t0 <- Sys.time()
  # In the multipolygons layer osm_id is the relation id; osm_way_id is set
  # only for polygons built from a single closed way. COALESCE gives every
  # feature an id, and osm_way_id marks the way-built ones.
  extra <- if (layer == "multipolygons") "osm_way_id, " else ""
  id_expr <- if (layer == "multipolygons") "COALESCE(NULLIF(osm_id, ''), osm_way_id) AS osm_id, " else "osm_id, "
  size_sql <- switch(layer,
    points = "0.0",
    lines = "ST_Length(geometry, 1) / 1000.0",
    multipolygons = "sqrt(ST_Area(ST_MakeValid(geometry), 1)) / 1000.0")
  sql <- paste0(
    "SELECT ", id_expr, extra,
    "name, name_en, alt_name, old_name, place, natural, landuse, waterway, highway, ",
    "railway, amenity, military, boundary, admin_level, ",
    size_sql, " AS size_km, ST_PointOnSurface(geometry) AS geom ",
    "FROM ", layer, " WHERE ", where)
  g <- file.path(work, paste0(layer, ".gpkg"))
  unlink(g)
  log <- file.path(work, paste0(layer, ".log"))
  status <- system2(
    "ogr2ogr",
    c("-f", "GPKG", g, pbf,
      "--config", "OSM_CONFIG_FILE", conf,
      "--config", "OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES",
      "--config", "OSM_COMPRESS_NODES", "YES",
      "-dialect", "SQLITE", "-nlt", "POINT", "-nln", layer,
      "-makevalid", "-skipfailures", "-progress",
      "-sql", shQuote(sql)),
    stdout = log, stderr = log)
  if (status != 0) stop("ogr2ogr failed on ", layer, ", see ", log)
  sfd <- st_read(g, quiet = TRUE, stringsAsFactors = FALSE)
  co <- st_coordinates(sfd)
  d <- as.data.table(st_drop_geometry(sfd))
  d[, `:=`(lon = co[, 1], lat = co[, 2], osm_layer = layer)]
  cat(sprintf("%-14s %9.0f features  (%.1f min)\n", layer, nrow(d),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  d
}

cat("reading", basename(pbf), "\n")
raw <- rbindlist(lapply(layers, fetch), fill = TRUE, use.names = TRUE)

# The multipolygons layer's id convention: osm_id is the relation id, while
# osm_way_id is set only for polygons built from a single closed way.
mp <- raw[osm_layer == "multipolygons"]
if (nrow(mp)) {
  mp[, osm_type_previous := fifelse(!is.na(osm_way_id), "relation", "way")]
  mp[, osm_type_fixed := fifelse(!is.na(osm_way_id), "way", "relation")]
  cat("\nmultipolygons by osm_type (previous logic):\n")
  print(mp[, .N, by = osm_type_previous][order(-N)])
  cat("multipolygons by osm_type (fixed logic):\n")
  print(mp[, .N, by = osm_type_fixed][order(-N)])
  dup <- mp[, .N, by = .(osm_type_fixed, osm_id)][N > 1]
  cat("(osm_type, osm_id) pairs with more than one geometry:", nrow(dup), "\n")
  if (nrow(dup)) print(head(dup[order(-N)], 10))
}

if (!"admin_level" %in% names(raw)) raw[, admin_level := NA_character_]
raw[, size_km := as.numeric(size_km)]

#------------------------------------------------------------------------------#
####                          Class and tag label                          ####
#------------------------------------------------------------------------------#
raw[, `:=`(
  place = tolower(place), natural = tolower(natural), landuse = tolower(landuse),
  waterway = tolower(waterway), highway = tolower(highway),
  railway = tolower(railway), amenity = tolower(amenity),
  military = tolower(military), boundary = tolower(boundary))]
raw[, admin_level := suppressWarnings(as.integer(admin_level))]

raw[, cls := fcase(
  osm_layer == "multipolygons" & !is.na(boundary) & boundary == "administrative" &
    !is.na(admin_level) & admin_level %in% 6:10, "admin",
  !is.na(place) & place %in% c("city","town","village","hamlet","suburb",
                               "neighbourhood","locality","isolated_dwelling"), "settlement",
  (!is.na(amenity) & amenity == "police") | (!is.na(landuse) & landuse == "military") |
    !is.na(military), "security",
  (!is.na(landuse) & landuse == "forest") |
    (!is.na(natural) & natural %in% c("wood","peak","ridge","valley")) |
    (!is.na(boundary) & boundary == "protected_area"), "forest_terrain",
  (!is.na(waterway) & waterway %in% c("river","stream")) |
    (!is.na(natural) & natural == "water"), "water",
  (!is.na(highway) & highway %in% c("primary","secondary","tertiary","unclassified","track")) |
    (!is.na(railway) & railway == "rail"), "road",
  default = NA_character_)]

# the tag that put the feature in its class
raw[, osm_tag := fcase(
  cls == "admin",          "boundary=administrative",
  cls == "settlement",     paste0("place=", place),
  cls == "security",       fifelse(!is.na(amenity) & amenity == "police", "amenity=police",
                           fifelse(!is.na(landuse) & landuse == "military", "landuse=military",
                           paste0("military=", military))),
  cls == "forest_terrain", fifelse(!is.na(landuse) & landuse == "forest", "landuse=forest",
                           fifelse(!is.na(boundary) & boundary == "protected_area", "boundary=protected_area",
                           paste0("natural=", natural))),
  cls == "water",          fifelse(!is.na(waterway) & waterway %in% c("river","stream"),
                                   paste0("waterway=", waterway), "natural=water"),
  cls == "road",           fifelse(!is.na(highway), paste0("highway=", highway), "railway=rail"))]
raw <- raw[!is.na(cls)]
raw[cls != "admin", admin_level := NA_integer_]

raw[, osm_type := fcase(
  osm_layer == "points", "node",
  osm_layer == "lines",  "way",
  osm_layer == "multipolygons" & !is.na(osm_way_id), "way",
  osm_layer == "multipolygons", "relation")]

#------------------------------------------------------------------------------#
####                             Name variants                             ####
#------------------------------------------------------------------------------#
names_long <- function(col, label) {
  x <- raw[[col]]
  d <- raw[!is.na(x) & nzchar(trimws(x))]
  parts <- strsplit(d[[col]], ";", fixed = TRUE)
  d <- d[rep(seq_len(nrow(d)), lengths(parts))]
  d[, name := trimws(unlist(parts, use.names = FALSE))]
  d[nzchar(name), .(osm_id, osm_type, name, name_variant = label, class = cls,
                    osm_tag, admin_level, size_km, lon, lat)]
}
gz0 <- rbindlist(list(
  names_long("name", "name"),
  names_long("name_en", "name:en"),
  names_long("alt_name", "alt_name"),
  names_long("old_name", "old_name")))
gz0 <- unique(gz0, by = c("osm_type", "osm_id", "name", "name_variant"))

# Latin-ASCII transliteration for every non-ASCII name, unless the resulting
# string is already stored for the same feature.
src <- unique(gz0[!stri_enc_isascii(name),
                  .(osm_id, osm_type, name, class, osm_tag, admin_level, size_km, lon, lat)])
src[, translit := stri_trans_general(name, "Latin-ASCII")]
src <- src[nzchar(translit) & translit != name]
translit <- unique(src[, .(osm_id, osm_type, name = translit, name_variant = "translit",
                           class, osm_tag, admin_level, size_km, lon, lat)])
translit <- translit[!unique(gz0[, .(osm_type, osm_id, name)]), on = .(osm_type, osm_id, name)]
gz <- rbindlist(list(gz0, translit), use.names = TRUE)
cat("gazetteer rows (one per name variant):", nrow(gz),
    "(", nrow(translit), "translit )\n")

#------------------------------------------------------------------------------#
####                          Assign 2001 units                            ####
#------------------------------------------------------------------------------#
units_full <- st_make_valid(st_read(gpkg_units, quiet = TRUE))
units <- units_full[, "unit_id"]
pts <- st_transform(st_as_sf(gz, coords = c("lon","lat"), crs = 4326), st_crs(units))
idx <- st_intersects(pts, units)
gz[, unit_id := vapply(idx, function(i) if (length(i)) units$unit_id[i[1]] else NA_character_, character(1))]
gz[, n_units_hit := lengths(idx)]
cat("features outside any unit:", round(100 * mean(is.na(gz$unit_id)), 2), "%\n")

gz <- gz[, .(osm_id, osm_type, name, name_variant, class, osm_tag, admin_level,
             size_km, lon, lat, unit_id)]
saveRDS(gz, out_rds)
cat("saved", out_rds, "(", nrow(gz), "rows )\n")

#------------------------------------------------------------------------------#
####                       Administrative boundaries                        ####
#------------------------------------------------------------------------------#
cat("extracting administrative boundaries\n")
t0 <- Sys.time()
sql_admin <- paste0(
  "SELECT COALESCE(NULLIF(osm_id, ''), osm_way_id) AS osm_id, osm_way_id, ",
  "name, name_en, alt_name, old_name, admin_level, ",
  "geometry AS geom FROM multipolygons WHERE ", has_name,
  " AND boundary = 'administrative' AND admin_level IN ('6','7','8','9','10')")
g_admin <- file.path(work, "admin.gpkg")
unlink(g_admin)
log_admin <- file.path(work, "admin.log")
status <- system2(
  "ogr2ogr",
  c("-f", "GPKG", g_admin, pbf,
    "--config", "OSM_CONFIG_FILE", conf,
    "--config", "OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES",
    "--config", "OSM_COMPRESS_NODES", "YES",
    "-dialect", "SQLITE", "-nlt", "PROMOTE_TO_MULTI", "-nln", "admin",
    "-makevalid", "-skipfailures", "-progress",
    "-sql", shQuote(sql_admin)),
  stdout = log_admin, stderr = log_admin)
if (status != 0) stop("ogr2ogr failed on admin boundaries, see ", log_admin)
adm <- st_read(g_admin, quiet = TRUE, stringsAsFactors = FALSE)
adm$admin_level <- suppressWarnings(as.integer(adm$admin_level))
adm$osm_type <- ifelse(!is.na(adm$osm_way_id), "way", "relation")
adm <- adm[!is.na(adm$admin_level), ]
cat(sprintf("admin boundaries: %.0f polygons (%.1f min)\n", nrow(adm),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

unlink(out_admin)
for (lv in sort(unique(adm$admin_level))) {
  x <- adm[adm$admin_level == lv,
           c("osm_id", "osm_type", "name", "name_en", "alt_name", "old_name",
             "admin_level")]
  st_write(x, out_admin, layer = paste0("admin_level_", lv),
           append = file.exists(out_admin), quiet = TRUE)
}
cat("saved", out_admin, "with layers",
    paste(paste0("admin_level_", sort(unique(adm$admin_level))), collapse = ", "), "\n")

#------------------------------------------------------------------------------#
####                                Report                                  ####
#------------------------------------------------------------------------------#
feat <- unique(gz[, .(osm_type, osm_id, class, osm_tag, admin_level, size_km, lon, lat, unit_id)])
unit_state <- unique(as.data.table(st_drop_geometry(units_full))[, .(unit_id, state_name)])
feat <- merge(feat, unit_state, by = "unit_id", all.x = TRUE)

cat("\nfeatures by class (unique features):\n")
print(feat[, .(features = .N, no_unit = sum(is.na(unit_id)),
               pct_no_unit = round(100 * mean(is.na(unit_id)), 2)),
           by = class][order(-features)])

cat("\nadmin features by admin_level:\n")
print(feat[class == "admin", .(features = .N), by = admin_level][order(admin_level)])
cat("admin features with no 2001 unit:",
    feat[class == "admin" & is.na(unit_id), .N], "\n")

# SHRID denominators per state (2001 unit state names and SHRID names differ
# only in punctuation, "and" and "of").
norm_state <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("&", "and", x)
  x <- gsub("[^a-z ]", " ", x)
  x <- gsub("\\band\\b|\\bof\\b", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}
shrid <- fread(file.path(root, "data/shrid_loc_names.csv"))
shrid[, state_key := norm_state(state_name)]
den <- shrid[, .(state_name = state_name[1],
                 shrid_subdistricts = uniqueN(subdistrict_name, na.rm = TRUE),
                 shrid_places = .N), by = state_key]

adm_state <- feat[class == "admin" & admin_level %in% c(6L, 9L) & !is.na(unit_id),
                  .(features = .N), by = .(admin_level, unit_id, state_name)]
adm_state <- adm_state[, .(features = sum(features)), by = .(admin_level, state_name)]
adm_state[, state_key := norm_state(state_name)]

grid <- merge(CJ(admin_level = c(6L, 9L), state_key = den$state_key), den, by = "state_key")
grid <- merge(grid, adm_state[, .(admin_level, state_key, features)],
              by = c("admin_level", "state_key"), all.x = TRUE)
grid[is.na(features), features := 0L]
grid[, denominator := fifelse(admin_level == 6L, shrid_subdistricts, shrid_places)]
grid[, ratio := features / denominator]

for (lv in c(6L, 9L)) {
  cat(sprintf("\nadmin_level %d by state (%s per state):\n", lv,
              if (lv == 6L) "ratio to SHRID sub-districts" else "ratio to SHRID places"))
  x <- grid[admin_level == lv][order(-features),
        .(state_name, features, shrid = denominator, ratio = round(ratio, 3))]
  print(x)
}

cat("\nsize_km distribution, road and forest_terrain (km; polygons use sqrt(area)):\n")
sz <- feat[class %in% c("road", "forest_terrain") & !is.na(size_km),
           .(features = .N,
             min = min(size_km), p10 = quantile(size_km, 0.10), p25 = quantile(size_km, 0.25),
             median = median(size_km), p75 = quantile(size_km, 0.75),
             p90 = quantile(size_km, 0.90), p99 = quantile(size_km, 0.99),
             max = max(size_km), mean = mean(size_km)),
           by = class]
print(sz[, lapply(.SD, function(z) if (is.numeric(z)) round(z, 3) else z), by = class])

feat_ids <- unique(gz[, .(osm_type, osm_id)])
ascii_ids <- unique(gz[stri_enc_isascii(name), .(osm_type, osm_id)])
n_no_ascii <- nrow(feat_ids[!ascii_ids, on = .(osm_type, osm_id)])
cat("\nfeatures with no ASCII name after transliteration:",
    n_no_ascii, "of", nrow(feat_ids),
    "(", round(100 * n_no_ascii / nrow(feat_ids), 2), "%)\n")

# Settlement counts per unit, kept for the coverage report.
u <- as.data.table(st_drop_geometry(units_full))[, .(unit_id, unit_name, state_name, n_shrids)]
settle <- unique(gz[class == "settlement", .(unit_id, osm_type, osm_id)])
per_unit <- settle[, .(n_settlements = .N), by = unit_id]
u <- merge(u, per_unit, by = "unit_id", all.x = TRUE)
set(u, which(is.na(u$n_settlements)), "n_settlements", 0L)
u[, settlements_per_1000_shrids := 1000 * n_settlements / n_shrids]
saveRDS(u, file.path(root, "data/osm_gazetteer_india_units.rds"))
