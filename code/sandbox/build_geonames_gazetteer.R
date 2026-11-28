#------------------------------------------------------------------------------#
### SATP: GeoNames gazetteer for India
#
# Builds a searchable place-name gazetteer for India from the GeoNames country
# dump, with one point per feature and the 2001 district unit it falls in. It is
# the GeoNames counterpart of code/sandbox/build_osm_gazetteer.R and uses the
# same point-in-polygon assignment (india_units_2001.gpkg).
#
# Feature classes kept: P (populated places), H (hydrography), L (areas), T
# (mountains/hills), S (spots/buildings/farms), V (forests), R (roads, railways,
# tunnels), plus administrative units at tehsil/block level and below, A with
# code ADM3, ADM4 or ADM5. The dump's remaining classes are dropped: A
# ADM1/ADM2 (state and district, already represented by the 2001 unit layer)
# and the other A codes duplicate it too, and U (undersea) does not occur in
# India.
#
# Name variants, one row per distinct string within a feature:
#   name       the primary name
#   ascii      asciiname, when it differs from the primary name
#   alternate  each entry of the comma-separated alternatenames field
#   translit   the Latin-ASCII transliteration of every primary and alternate
#              name, kept only when it is not already a name/ascii/alternate
# Duplicates of the same string within a geonameid are collapsed, preferring
# name > ascii > alternate > translit.
#
# Input:   data/input/external/geonames/IN.zip   (GeoNames, see the README there)
#          irrigating_peace/mattia/data/intermediate/restricted/india_units_2001.gpkg
# Output:  data/intermediate/gazetteers/geonames_gazetteer_india.rds  columns geonameid, name,
#          name_variant, feature_class, feature_code, lon, lat, unit_id
#
# Column mapping from the OSM gazetteer: osm_id -> geonameid, osm_tag ->
# feature_class + feature_code, alternatenames -> name_variant rows. osm_type
# and the OSM semantic class column have no GeoNames analogue and are omitted.
#
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(sf)
  library(data.table)
  library(stringi)
})

root <- Sys.getenv("SATP_ROOT", unset = "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
zip_path <- Sys.getenv("SATP_GEONAMES_ZIP", unset = file.path(root, "data/input/external/geonames/IN.zip"))
gpkg_units <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia/data/intermediate/restricted/india_units_2001.gpkg"
out_rds <- file.path(root, "data/intermediate/gazetteers/geonames_gazetteer_india.rds")
stopifnot(file.exists(zip_path), file.exists(gpkg_units))

work <- file.path(tempdir(), "geonames_gazetteer")
dir.create(work, showWarnings = FALSE)
txt <- file.path(work, "IN.txt")
if (!file.exists(txt)) {
  status <- system2("unzip", c("-o", shQuote(zip_path), "IN.txt", "-d", shQuote(work)),
                    stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(status, "status")) && attr(status, "status") != 0)
    stop("unzip failed: ", paste(status, collapse = "\n"))
}
stopifnot(file.exists(txt))

#------------------------------------------------------------------------------#
####                            Read and filter                             ####
#------------------------------------------------------------------------------#
# Columns 1-8 of IN.txt: geonameid, name, asciiname, alternatenames, lat, lon,
# feature class, feature code.
cat("reading", basename(zip_path), "\n")
d <- fread(txt, sep = "\t", header = FALSE, quote = "", encoding = "UTF-8",
           select = c(1, 2, 3, 4, 5, 6, 7, 8),
           col.names = c("geonameid", "name", "asciiname", "alternatenames",
                         "lat", "lon", "feature_class", "feature_code"),
           showProgress = FALSE)
cat("features in dump:", nrow(d), "\n")

old_classes <- c("P", "H", "L", "T", "S", "V", "R")
adm_classes <- c("ADM3", "ADM4", "ADM5")
d <- d[((feature_class %in% old_classes) |
        (feature_class == "A" & feature_code %in% adm_classes)) &
         !is.na(lat) & !is.na(lon) & lat >= -90 & lat <= 90 & lon >= -180 & lon <= 180]
cat("features kept:", nrow(d), "(P H L T S V R:", nrow(d[feature_class %in% old_classes]),
    "| A ADM3-5:", nrow(d[feature_class == "A"]), ")\n")

#------------------------------------------------------------------------------#
####                             Name variants                             ####
#------------------------------------------------------------------------------#
base <- d[, .(geonameid, name, asciiname, alternatenames, feature_class, feature_code, lon, lat)]
has_text <- function(x) !is.na(x) & nzchar(trimws(x))

primary <- base[has_text(name),
                .(geonameid, name = trimws(name), name_variant = "name",
                  feature_class, feature_code, lon, lat)]

ascii <- base[has_text(asciiname)]
ascii <- ascii[is.na(name) | trimws(name) != trimws(asciiname)]
ascii <- ascii[, .(geonameid, name = trimws(asciiname), name_variant = "ascii",
                   feature_class, feature_code, lon, lat)]

alt <- base[has_text(alternatenames)]
parts <- strsplit(alt$alternatenames, ",", fixed = TRUE)
alt <- alt[rep(seq_len(nrow(alt)), lengths(parts))]
alt[, name := trimws(unlist(parts, use.names = FALSE))]
alt <- alt[nzchar(name),
           .(geonameid, name, name_variant = "alternate",
             feature_class, feature_code, lon, lat)]

# Latin-ASCII transliteration of every primary and alternate name.
src <- unique(rbindlist(list(
  primary[, .(geonameid, name, feature_class, feature_code, lon, lat)],
  alt[, .(geonameid, name, feature_class, feature_code, lon, lat)])),
  by = c("geonameid", "name"))
src[, name := stri_trans_general(name, "Latin-ASCII")]
translit <- src[nzchar(name),
                .(geonameid, name, name_variant = "translit",
                  feature_class, feature_code, lon, lat)]

gz <- rbindlist(list(primary, ascii, alt, translit))
n_before_dedupe <- nrow(gz)
gz[, variant_rank := match(name_variant, c("name", "ascii", "alternate", "translit"))]
setorder(gz, geonameid, name, variant_rank)
gz <- unique(gz, by = c("geonameid", "name"))  # one row per distinct string per feature
gz[, variant_rank := NULL]
cat("variant rows before dedupe:", n_before_dedupe,
    "| after dedupe:", nrow(gz), "\n")

#------------------------------------------------------------------------------#
####                          Assign 2001 units                            ####
#------------------------------------------------------------------------------#
units_full <- st_make_valid(st_read(gpkg_units, quiet = TRUE))
units <- units_full[, "unit_id"]

# One point per feature; unit assignment is a property of the feature.
feat <- unique(gz[, .(geonameid, feature_class, feature_code, lon, lat)])
pts <- st_transform(st_as_sf(feat, coords = c("lon", "lat"), crs = 4326), st_crs(units))
idx <- st_intersects(pts, units)
feat[, unit_id := vapply(idx, function(i) if (length(i)) units$unit_id[i[1]] else NA_character_, character(1))]
cat("features outside any unit:", round(100 * mean(is.na(feat$unit_id)), 2), "%\n")

gz <- merge(gz, feat[, .(geonameid, unit_id)], by = "geonameid", all.x = TRUE, sort = FALSE)
gz <- gz[, .(geonameid, name, name_variant, feature_class, feature_code, lon, lat, unit_id)]
saveRDS(gz, out_rds)
cat("saved", out_rds, "(", nrow(gz), "rows )\n")

#------------------------------------------------------------------------------#
####                                Report                                  ####
#------------------------------------------------------------------------------#
old_rows <- gz[geonameid %in% d[feature_class %in% old_classes, geonameid]]
cat("\ngazetteer size:\n")
cat(sprintf("  base class set (P H L T S V R):   %8d rows, %7d geonameids\n",
            nrow(old_rows), uniqueN(old_rows$geonameid)))
cat(sprintf("  new class set (+ A ADM3-5):       %8d rows, %7d geonameids\n",
            nrow(gz), uniqueN(gz$geonameid)))
cat(sprintf("  before dedupe:                    %8d rows, %7d geonameids\n",
            n_before_dedupe, uniqueN(gz$geonameid)))
cat(sprintf("  after dedupe:                     %8d rows, %7d geonameids\n",
            nrow(gz), uniqueN(gz$geonameid)))

ascii_share <- 100 * uniqueN(gz[stri_enc_isascii(name), geonameid]) / uniqueN(gz$geonameid)
cat("\ngeonameids with at least one ASCII-only name:",
    round(ascii_share, 2), "%\n")

cat("\nfeatures and rows by feature class:\n")
feat_rep <- feat[, .(features = .N,
                     no_unit = sum(is.na(unit_id)),
                     pct_no_unit = round(100 * mean(is.na(unit_id)), 2)),
                 by = feature_class][order(-features)]
row_rep <- gz[, .(rows = .N), by = feature_class]
print(merge(feat_rep, row_rep, by = "feature_class", all.x = TRUE)[order(-features)])

cat("\nrows by name variant:\n")
print(gz[, .(rows = .N), by = name_variant][order(-rows)])

cat("\nA features kept, by code:\n")
print(feat[feature_class == "A", .(features = .N), by = feature_code][order(-features)])

cat("\nexample rows for a feature with diacritics:\n")
cat("(ascii = IN.txt asciiname; translit = stri_trans_general(name, \"Latin-ASCII\"))\n")
cand <- gz[stri_trans_general(name, "Latin-ASCII") != name,
           .(n_names = uniqueN(name)), by = geonameid][order(-n_names)]
ex_id <- cand[n_names >= 10][1, geonameid]
if (is.na(ex_id)) ex_id <- cand[1, geonameid]
exb <- base[geonameid == ex_id]
ex_names <- unique(trimws(c(exb$name, exb$asciiname,
                            strsplit(exb$alternatenames, ",", fixed = TRUE)[[1]])))
ex_names <- ex_names[!is.na(ex_names) & nzchar(ex_names)]
ex <- data.table(geonameid = ex_id, name = ex_names,
                 ascii = exb$asciiname[1],
                 translit = stri_trans_general(ex_names, "Latin-ASCII"))
ex <- ex[order(translit == name)]  # rows with diacritics first
print(head(ex, 10))
