# To-do: place coding of SATP events

Status as of 2026-11-23. Context: the 100-event hand check (`data/intermediate/validation/place_check_hand.xlsx`) is in progress (rows through check ID 61 entered; check IDs 103 and 105 are discussed in the log at the end but not yet entered); the full DeepSeek run waits for it (draft task in `notes/task_full_place_extraction.md`, do not run before the check is scored).

## 1. Coordinate check against the administrative unit (all geocoded events)

Apply to every event that has some degree of geocoding, whatever the source (SHRID, GeoNames, OSM, a model or a geocoder).

- [ ] Write a sandbox script that tests whether the coordinates of each proposed place fall inside the unit named for the event, using the project's 2001 district units (`irrigating_peace/mattia/data/intermediate/restricted/india_units_2001.gpkg`).
- [ ] Use the alias list for districts that did not exist in 2001 (`data/intermediate/places/district_aliases.csv`, `code/sandbox/district_aliases.csv`), so events in Kulgam, Shopian and similar districts are not flagged without reason.
- [ ] District check is the hard flag: `in_district` (TRUE/FALSE) and `km_outside_district` for the failures.
- [ ] Sub-district check is a soft flag, reported separately: flag only if the point is more than about 5 km outside the sub-district; police station areas do not follow census sub-districts, so boundary cases are expected. Use the SHRUG 2011 sub-district polygons (`data/input/public/SHRUG/shrug-pc11subdist-poly-shp.zip`) as a first version.
- [ ] Run it first on the 100 hand-checked rows and see which of the hand-marked errors it would have caught.
- [ ] Keep the flags as columns in the output file (`place_analysis_set.csv`), not as filters.
- [ ] Accepted for now: two villages with the same name inside one district can pass the check; a police station match that is near the named village (example: row 103, Dhola police station for Nao Meseki village) is acceptable.

### Distance to the SHRID village (second check)

For points that did not come from SHRID (OSM, GeoNames, a model): look up the extracted village name among the SHRID places of the event's district and measure the distance from the chosen point to that village. Example: row 105, Galhuta (SHRID 11-01-005-00018-001216) is 6.0 km from the pipeline pin, which was the Mendhar sub-district point.

- [ ] Use the SHRID village polygons (`data/input/external/shrug/shrid2_open.shp`, 595,438 polygons) and measure the distance from the point to the polygon; fall back to the centroid (`data/intermediate/gazetteers/shrid_centroids.rds`) where a village has no polygon.
- [ ] First report the distribution of the distances over the hand-checked rows (mean, median, standard deviation, split by the hand-check answer); set the cutoff afterwards.
- [ ] Split hyphenated names ("Kanghra-Galhutta") into their parts before the lookup, and make sure the extraction returns the village name, not only the sub-district name.

### Other open data to check (from the villageinfo.in methodology page)

villageinfo.in compiles Census of India, the Local Government Directory (LGD), India Post PIN codes and data.gov.in datasets; it names no source for coordinates, so its points are not an independent check. Do not scrape it (it blocks automated access).

- [ ] LGD (Local Government Directory): free downloads of the administrative hierarchy with codes; could help with current names of districts and sub-districts (the alias list).
- [ ] India Post PIN code directory on data.gov.in: check whether it has post office coordinates and use PIN codes as a district cross-check (examples from the hand check: Dani Hama 190006, Srinagar).

## 2. Police stations as places

- [ ] Flag every event whose text says "under X police station" (or thana, PS limits) so it can get another pass.
- [ ] Police station source: the OSM police features of the local India extract (`amenity=police`, 4,203 features in the OSM gazetteer). Do not scrape policestationsindia.com; its terms prohibit bulk download. Its coordinates appear to be OSM's (7 of 10 sampled stations coincide with an OSM point).
- [ ] Name each station with the name tag OSM gives it (3,853 of 4,203 carry a locality, 350 are just "Police Station"). Do NOT name stations by the nearest settlement: in OSM, only about 32% of stations have the locality of their own name as the nearest settlement (33% within 3 km), checked 2026-11-23 on named stations, many of them urban.
- [ ] Coverage benchmark: the PIB release of 11 March 2026 (https://www.pib.gov.in/PressReleasePage.aspx?PRID=2238241&reg=48&lang=2) gives the official number of police stations by state and UT as on 1 Feb 2026 (17,798 in all; for example Chhattisgarh 501, Jharkhand 543, Odisha 681, Assam 335, J&K 268). Counts only, no names or locations. Station-like named OSM features in our gazetteer cover about 18% of the national total, and 5 to 9% in Chhattisgarh, Jharkhand, Bihar and Odisha, 14% in Assam, 32% in J&K (checked 2026-11-23; lower bounds, since unnamed OSM points are not in the gazetteer). OSM alone will not cover the conflict states.
- [ ] Because of this, also try matching "X police station" to the place X (SHRID, GeoNames, OSM settlements) inside the event's district, as the resolver already does for tehsils; a station is usually named after its locality.
- [ ] Validate the naming on the SATP texts: take the "under X police station" mentions, look for X among the OSM stations of the event's district, and compare the position with the hand-checked places; count unique, ambiguous and missing names.
- [ ] Count all `amenity=police` points in the raw India extract (`data/raw/osm/india-latest.osm.pbf`, from Geofabrik, downloaded 2026-11-19), named or not, with the GDAL tool the gazetteer script uses; the 4,203 above are only the named ones in the gazetteer, so the coverage shares are lower bounds. Geofabrik also keeps yearly snapshots (`https://download.geofabrik.de/asia/india/`) if the growth of OSM coverage in the conflict states matters.
- [ ] Other sources to look at, with the licence first: Overture Maps places (open licence; check India coverage and police stations), the CCTNS master code list of police stations (check whether it is public), data.gov.in and state police sites. The Google Places API is not suited: about 20 results per request, so a national sweep needs thousands of requests at roughly 32 to 40 USD per 1,000 beyond a small monthly free allowance (figures from search snippets, to be confirmed), and Google's terms restrict storing the results. Use it for spot checks only, and do not store the output as data.
- [ ] Unnamed stations: assign to a sub-district by point in polygon and leave them unnamed until another source is found (government police station lists, data.gov.in, state police sites).
- [ ] If the match rate is good, use police station points or sub-district polygons for these events.

## 3. Named place is not where the act happened

Seen in the hand check: release site (row 58, Barengapara), origin city (row 60, Warangal). The coordinate check does not catch these.

- [ ] Add a rule to the next prompt version: for abductions, extract only the place of the abduction; for a release, arrest or transfer, return none if only the other place is named.
- [ ] Add a flag column for places that are a district name or district headquarters (`district_hq_name` exists in the resolver) and for events with a "highway from X to Y" or "while being taken" wording.
- [ ] Check how often this happens once the 100 events are scored; decide then whether the flag is enough.

## 4. Resolver rules from the hand check

- [ ] Strip direction words before matching (Pachim, Purba, Uttar, Dakshin and similar): row 44, Pachim Kamarkuchi.
- [ ] Ignore spaces and hyphens in place names: row 45, Danihama vs Dani Hama.
- [ ] Spelling variants that fuzzy matching misses: row 61, Redwani Pine vs Redwani Payeen. Look at a list of common variants (Pine/Payeen, Naupora/Naupura).
- [ ] Places that are not in SHRID, GeoNames or OSM (Chajru, Danihama, Redwani): decide whether an outside source is needed (see 6).

## 5. Entries with more than one event

- [ ] Count how many hand-checked entries describe two or more events (rows 45 and 57 so far) and compare with the share of incidents that are split (2.7% in the granularity report).
- [ ] If the share is large, split these entries before the place step, so each event gets its own place.

## 6. Models

- [ ] Finish the hand check, then score it by tier and by failure type.
- [ ] Fix rules and prompt (items 3 and 4), then rerun on the same events with deepseek-flash and deepseek-pro; check Pro's price first.
- [ ] Tune on the first half of the sheet and keep the second half for testing.
- [ ] Gemini test on the hand-checked rows, free tier, no grounding; key in the environment variable `GEMINI_API_KEY`, never in files. Grounding with Google Maps or Search only for the places still unresolved, with a billing alert, and check the Google Maps terms on storing coordinates first.
- [ ] After that, the full DeepSeek run on the analysis set (draft task in `notes/task_full_place_extraction.md`; new cache `data/intermediate/places/place_cache_analysis.rds`, do not touch `place_cache_pilot.rds`).

## 7. Hand check

- [ ] Continue the check of the remaining rows; one row per place, same check ID for entries with several places, columns L and M with yes / no / cant_tell.
- [ ] Use the same note wording for the same failure type, so the counts are easy: "release site, not event site", "origin city, not event site", "manual addition".
- [ ] Tips for the check: in the OpenStreetMap web map, the "Query features" button and then a click on the map lists the administrative areas enclosing a point (names and admin levels; levels are not consistent across Indian states); `https://nominatim.openstreetmap.org/ui/reverse.html` gives the address hierarchy for a point.
- [ ] Enter check IDs 103 and 105 in the sheet (see the log below).

## 8. Hand-check log (what the first rows showed)

Row = check ID in the sheet; written 2026-11-23, to be replaced by the scored table.

- 44 (Nalbari): "Pachim Kamarkuchi"; SHRID has three Kamarkuchi, one in Nalbari (Barbhag, `11-18-323-02135-303860`) and two in Kamrup Metropolitan; Google picked the wrong one. Pipeline missed it: direction word. Homonym across districts is caught by the district filter.
- 45 (Srinagar): two encounters, Naupora (pipeline OSM/GeoNames point inside Srinagar, plausible, not confirmed) and Danihama (Dani Hama, 34.1568 N, 74.8917 E, postcode 190006, missed because of the space). Entered as two rows.
- 53 (Pulwama unit, text Shopian): Barabug/Barbugh, correct; GeoNames match, not in OSM; Shopian was part of Pulwama in 2001.
- 57 (Poonch): two encounters, Khanetar (correct, within 1 km of the GeoNames village) and the upper reaches of Chajru (in no gazetteer, landmark type). Entered as two rows.
- 58 (West Garo Hills): Barengapara is the release site, abduction place not stated. Failure type "release site, not event site".
- 60 (Warangal): the place is the origin city; event on the Warangal to Hyderabad highway. Failure type "origin city, not event site". I recall the incident near Alair, Nalgonda (not verified).
- 61 (Kulgam, 2001 unit Anantnag): "Redwani Pine" is Redwani Payeen, Qaimoh block (found through a Google overview, to confirm on a map); the pipeline found nothing.
- 103 (Tinsukia, not yet entered): the pipeline matched Dhola police station for the village Nao Meseki; accepted as close enough. Pin inside Tinsukia.
- 105 (Punch, not yet entered): text "Kanghra-Galhutta village, Mendhar sector"; the pipeline pin is the Mendhar sub-district point (33.6082 N, 74.1733 E); SHRID Galhuta (Mendhar sub-district) is at 33.5797 N, 74.2287 E, 6.0 km away; Google and villageinfo.in about 3 km from the SHRID centroid. Suggested entry: site name Kanghra-Galhutta, proposed correct no, right sub-district yes (Mendhar), note "pipeline matched the sub-district name, not the village".
