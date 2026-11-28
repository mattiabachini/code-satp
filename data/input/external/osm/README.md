# OSM extract

`india-latest.osm.pbf` — OpenStreetMap extract of India from Geofabrik.

| | |
|---|---|
| source | https://download.geofabrik.de/asia/india-latest.osm.pbf |
| downloaded | 2026-11-19 (local) |
| server snapshot (`Last-Modified`) | 2026-11-19 00:07:25 UTC |
| size | 1,716,010,705 bytes (1.6 GiB) |
| licence | OpenStreetMap contributors, ODbL 1.0 |

Geofabrik rebuilds `-latest` daily, so the file is only the snapshot recorded
above; re-download to refresh it. The file is not tracked by git.

Built from this extract by `code/sandbox/build_osm_gazetteer.R`, which writes
`data/osm_gazetteer_india.rds` (named features with a representative point and
their 2001 district unit), `data/osm_admin_boundaries_india.gpkg` (named
administrative polygons, one layer per admin_level 6–10) and
`data/osm_gazetteer_india_units.rds` (settlement counts per unit, used for the
coverage report).
