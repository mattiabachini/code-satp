# How granular can the SATP event data go?

Analysis of the final SATP event file (`satp_events_conf07.rds`, 91,555 events,
2000–2025) run on 2026-11-19. Every figure below comes from exact name matching
of the event text against the SHRID place directory, restricted to the event's
own 2001 district unit; the script is a session-side probe, not in the
repository. Shares are of all events unless the column says "analysis set"
(confidence > 0.7, at least one analysis action, with a district unit).

## 1. What the final file actually carries

| field | coverage |
|---|---|
| `admin1` (state) | 91.7% |
| `admin2` / `unit_id` (2001 district unit) | 85.8% (525 distinct names, 527 unit ids) |
| `admin3`, `location`, `latitude`, `longitude` | 0% |
| `geo_precision` = 2 (district) | 85.8% |
| `geo_precision` = 3 (state only) | 5.9% |
| no location at all | 8.3% |

`geo_precision` never takes the value 1, so the file is, by construction, capped
at district level. Anything below district has to be recovered from `notes`.

Analysis set (confidence > 0.7, at least one analysis action, with a unit):
40,674 events in 376 units.

## 2. How specific is the text?

Share of events whose text contains a cue:

| cue | all events | analysis set |
|---|---|---|
| village / hamlet / tola / para / colony / locality | 25.2% | 25.7% |
| tehsil / tahsil / block / police station / thana / mandal | 18.6% | 18.6% |
| forest / jungle | 8.7% | 11.1% |
| border / LoC | 10.0% | 10.7% |
| "area" / sector / zone | 30.8% | 35.4% |

## 3. How far down can a name be resolved?

SHRID (576,582 census places, 386,421 distinct village/town names, 5,594
sub-districts) joined to the 2001 units via `india_shrid_unit_key.rds`. A match
counts only if the name sits inside the event's own district unit and appears as
a proper noun in the text.

| tier | all events | analysis set |
|---|---|---|
| village/town, strong (local cue or multi-word name) | 14.0% | 17.7% |
| tehsil/block inside the district | 15.7% | 19.8% |
| **below district, usable** | **29.7%** | **37.5%** |
| village/town, weak (bare single word) | 8.3% | 9.3% |
| sub-district name found only in a *different* district | 6.0% | 5.1% |
| district only | 41.9% | 48.2% |
| no district | 14.2% | — |

These are the baseline (SHRID only). Section 7 shows how far the cheap
improvements push them; the ranking logic does not change.

Tiers are ranked: an event that names both a village and a tehsil is counted at
village level, which is why the tehsil share here (19.8% of the analysis set) is
lower than the 24.3% of analysis events that name any tehsil/block.

By series (analysis set, share below district):

| series | n | below district | of which village |
|---|---|---|---|
| Maoist | 11,079 | 53.0% | 31.5% |
| J&K | 16,658 | 34.3% | 11.0% |
| Northeast | 11,982 | 29.1% | 15.0% |
| Punjab | 236 | 27.5% | 15.7% |
| other / all-India | 719 | 13.1% | 5.7% |

By state (analysis set, below district): Odisha 62.2%, Maharashtra 52.7%,
Jharkhand 51.9%, Chhattisgarh 48.0%, Bihar 47.8%, Andhra Pradesh 47.7%,
Meghalaya 39.3%, J&K 34.5%, Manipur 30.1%, Assam 25.8%, Tripura 24.9%.

Over time: 30.6% (2000–05) rising to 49.2% (2021–25), i.e. later reporting is
more place-specific.

## 4. Quality of the sub-district information

- 89.6% of strong village matches resolve to exactly one village name inside the
  district; 10.4% are homonyms within the same district.
- Eyeballing ~35 matches per tier: the strong tier is mostly genuine
  ("Faraspal village", "Porompat", "Kalimela area", "Baliguda Police limits",
  "Chhatarpur sub-division"). Failure modes: partial names ("Ban Dodi" matched
  as "dodi"), person/outfit names that coincide with place names, and obsolete
  district spellings.
- The weak tier (bare single-word match) is noticeably noisier and should be
  treated as a flag, not a variable.
- Sub-district names are usually explicit ("under X police station", "Y
  tehsil", "Z block"), so this tier is the most reliable finer-than-district
  coding.
- Transliteration limits exact matching: among events with a village cue but no
  exact match, a Jaro–Winkler pass (≤0.10) finds a near name in the district for
  ~19.5% of them — roughly 2,000 events, ~5% of the analysis set, could be
  recovered with fuzzy matching plus review.
- Even among events that *do* carry a village cue, a third end up with no usable
  sub-district anchor (34.9% stay district-only, 6.8% have no district at all).

## 5. Would a finer panel be usable?

Analysis set, 376 units, 2000–2025:

| level | units used | cells | events per cell (median) | cells with ≥5 events |
|---|---|---|---|---|
| district | 376 | 3,431 of 9,776 unit-years (35.1%) | 3 | 43% |
| sub-district | 1,120 names | 3,789 sub-district-years | 1 (mean 2.6) | 11.2% |
| village | 3,861 names | 5,756 village-years | 1 | 1.2% |

Half of all sub-district-coded events sit in the small number of cells with ≥5
events (Maoist belt, Kashmir valley), so a sub-district panel is not a panel:
it is a handful of hot spots plus mostly zeros.

## 6. Benchmark: what a professional event dataset achieves

`acled_india.rds` (152,573 events, 2016–2024): `admin3` 99.3%, coordinates
100%, and `geo_precision` 1 (town/village) for 80.0%, 2 (district) for 18.2%,
3 (state) for 1.9%; median 13 distinct locations per district. So local news
text of this kind *does* support place-level coding — the gap is in our parsing,
not in the source. Note, though, that ACLED's precision-1 events are mostly
towns and market centres, not rural villages.

The older geocoding pipeline in `data/satp_location_matched_lat_long.csv`
(9,842 incidents, ~99% geocoded) shows that Google Maps can return coordinates
for nearly every event, but the matched string is often the district or a
nearby town, so the coordinates need validation before being treated as
village-accurate.

## 7. What the seven methods actually deliver

Tested on the full 91,555 events with local data (no paid API). Each row adds to
the row above; the last column is a plausibility estimate from eyeballing
30–50 matches per tier, not a labelled evaluation.

| method | what was run | all events | analysis set | precision (eyeball) |
|---|---|---|---|---|
| baseline | exact SHRID names inside the district | 29.7% | 37.5% | high (strong tier) |
| 2 + 3 | GeoNames India (651,839 features, 93.5% assigned to a 2001 unit; forests, hills, streams, spots) plus a wider cue list (forest, valley, camp, outpost, sector) | 36.4% | 45.4% | ~85% |
| 1 | Jaro–Winkler ≤ 0.10 on cue-adjacent capitalised names, restricted to the district | 40.9% | 51.7% | ~75–80% |
| 6 | spread the location of an incident to its other parts (`incident_uid`) | 42.1% | 52.9% | inherits above |

Composition of the analysis-set events located below district (21,510):

| source | events |
|---|---|
| village/town, exact | 10,986 |
| tehsil/block | 7,248 |
| landmark (forest/hill/area/spot) | 223 |
| fuzzy-only place | 2,590 |
| spread from a sibling part of the incident | 463 |

By series (analysis set, below district): Maoist 69.7%, J&K 50.7%, Punjab 47.5%,
Northeast 42.1%, other 24.1%.

### What each method is worth

1. **Fuzzy matching** — 4,473 new events (4.9% of all; 6.8% of the analysis set),
   of which 43% at distance ≤0.05. Works well on transliteration ("Korobeda" →
   Korhobeda, "Hapatnar" → Hapat nar, "Salwa" → Salwah, "Pargwal" → Pagwal).
   Residual errors: state names ("West Bengal" → "benga"), surnames, and
   district variants. Costs seconds, no API.
2. **GeoNames** — 2,624 events get a place name that only GeoNames has, and a
   further 448 gain a landmark (forest, hill, area, spot) that SHRID cannot
   represent at all. Weakness: GeoNames "populated place" includes surnames and
   administrative names (e.g. "Munda", "Malda"), so the same district-name
   filter has to be applied to it; I did, but ~10–15% leakage remains in the
   landmark tier.
3. **Cue anchoring** — the larger cue set alone promoted 2,920 previously
   "weak" matches to confident ones (the text says "X area", "X valley",
   "X forest", "X camp"). Free and worth keeping.
4. **Geocoding** — probed with Nominatim on 20 events that already have a place
   name: 9 returned a hit, 8 of those 9 inside the assigned district, with OSM
   type village (4), centre (3), town (1), locality (1). With only a district
   name, 8 of 8 queries return the district headquarters — i.e. geocoding adds
   nothing without a place name, as you suspected. Nominatim allows 1 request
   per second, so ~11 hours for the 40,674 analysis events; Google Maps is
   faster but billed.
5. **Language model** — not run: no API key in this environment. At
   deepseek-flash prices (~400 input tokens and ~40 output per event) it is
   about $0.00008 per event, i.e. **~$3–4 for the analysis set and ~$7–9 for all
   91,555 events**; a 500-event pilot is ~$0.05. `code/code_events_deepseek.R`
   already has the cache, batching and cost accounting to carry a
   place-extraction prompt, so the pilot is a prompt plus an output parser.
6. **Incident spread** — only 2.7% of incidents have more than one event part;
   it adds 669 events at sub-district level and 1,122 at district level
   (0.7% and 1.2%). Cheap, no risk, but small.
7. **Priors for the rest** — not run. Recipe: take the ~21.5k located analysis
   events, fit a within-district intensity (kernel density, or logistic on forest
   cover, elevation and past events), and use it to weight the remaining events'
   district cells. This produces probabilities for grid-cell analysis, not
   locations, and should be its own validation exercise.

### Practical reading

- Running 1, 2, 3 and 6 raises coverage below district from 37.5% to 52.9% of the
  analysis set, and to 69.7% in the Maoist belt, at zero API cost.
- The gain is largest where it matters for the paper: Odisha, Chhattisgarh,
  Jharkhand, Maharashtra, and — surprisingly — J&K, where "X area" and "X
  valley" phrases are common (34.3% → 50.7%).
- Precision is the binding constraint, not coverage: I would carry a
  `granularity` column (district / subdistrict / place / landmark) plus a
  `match_type` (exact / fuzzy / spread) and validate the fuzzy tier on 200 hand
  checks before using it as a variable. Until then, use exact matches for
  results and fuzzy matches for robustness.

## 8. Recommendation

1. **Keep the 2001 district unit as the analysis unit.** It is complete,
   stable and dense enough (median 3 events per observed unit-year).
2. **Use sub-district coding as a robustness sample, not a second panel.**
   Restrict to the Maoist belt, J&K tehsils and Odisha/Jharkhand/Maharashtra,
   where 48–62% of events have a tehsil/block name; report the district-level
   estimate on the same subsample.
3. **Do not build village-level panels.** Median one event per village-year and
   3,861 distinct village strings in 25 years make village fixed effects
   meaningless. Village matches are best used for qualitative checks, case
   studies, and validation of the district assignment.
4. **For spatially finer work, geocode events and use grid cells or kernel
   density** (25/50 km cells, or forest/terrain overlays), not administrative
   villages. Names are available for ~30% of events today and for considerably
   more after fuzzy matching; for the rest, coordinates can be assigned at the
   district centroid or randomly within the district, with the `geo_precision`
   flag carried through.
5. **Cheap improvements, in order of value:** (a) fuzzy match the unmatched
   village-cue phrases (+~5% coverage); (b) parse "X police station", "X
   tehsil", "X forest" as pseudo-units, which covers a further slice of the
   text; (c) hand-label ~1,000 events to train a small NER/fine-tuned model on
   place names if village-level coding becomes central to the paper.
