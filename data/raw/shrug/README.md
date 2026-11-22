# SHRID polygon shapefile

`shrid2_open.shp` — SHRID 2.0 village/ward polygons for India, used to attach a
`shrid2` to GeoNames and OSM candidate points.

| | |
|---|---|
| source archive | `irrigating_peace/data/input/public/SHRUG/shrug-shrid-poly-shp.zip` |
| source | SHRUG 2.2, Development Data Lab (https://www.devdatalab.org/shrug) |
| downloaded/extracted | 2026-11-19 (local) |
| archive date | 2026-07-16 |
| size | 884 MB uncompressed |
| features | 595,438 polygons |
| attributes | `shrid2`, `pc11_id`, `n`, `geometry_t`, `polysource`, `n_ncontig`, `maxdist_km` |
| licence | CC BY-NC-SA 4.0 |

The shapefile and its sidecar files are not tracked by git. `open_poly.bib`
holds the SHRUG citation:

Asher, Lunt, Matsuura and Novosad (2021), *Development research at high
geographic resolution: an analysis of night-lights, firms, and poverty in India
using the SHRUG open data platform*, World Bank Economic Review 35(4).

The 2001 census keys (`pc01r_shrid_key.dta`, `pc01u_shrid_key.dta`) are read
directly from `shrug-pc-keys-dta.zip` into a temporary directory by
`code/sandbox/resolve_places_multisource.R`.
