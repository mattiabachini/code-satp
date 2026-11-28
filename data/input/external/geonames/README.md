# GeoNames extract — India

`IN.zip` — GeoNames country dump for India.

| | |
|---|---|
| source | https://download.geonames.org/export/dump/IN.zip |
| downloaded | 2026-11-19 (local) |
| server snapshot (`Last-Modified`) | 2026-11-19 02:48:14 UTC |
| size | 15,755,098 bytes |
| MD5 | 4db290ed70726a11c9b9b370b24df0c0 |
| contents | `IN.txt` (69,613,188 bytes), `readme.txt` |
| licence | GeoNames, CC BY 4.0 |

GeoNames rebuilds the country dumps daily, so this is only the snapshot recorded
above; re-download to refresh it. The archive is not tracked by git.

`IN.txt` is the standard GeoNames tab-separated dump with 19 columns:
geonameid, name, asciiname, alternatenames, latitude, longitude, feature class,
feature code, country code, cc2, admin1–4 codes, population, elevation, dem,
timezone, modification date.

Built from this archive by `code/sandbox/build_geonames_gazetteer.R`, which
writes `data/geonames_gazetteer_india.rds`: one row per distinct name variant
for every feature in classes P, H, L, T, S, V and R, plus A with code ADM3, ADM4
or ADM5 (659,232 features, 1,054,947 rows). Each row carries the 2001 district
unit the feature falls in. The variant labels are `name` (primary name), `ascii`
(`asciiname` when it differs from the primary name), `alternate` (each entry of
`alternatenames`) and `translit` (Latin-ASCII transliteration, kept only when it
is not already stored for that feature).
