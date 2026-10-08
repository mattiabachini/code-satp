# SATP coded-event schema (draft)

One row per distinct incident. A SATP entry that describes several separate incidents becomes several rows. The layout follows ACLED's event structure (one event per row, two actors, a type hierarchy, a civilian-targeting flag, a location hierarchy with a precision code). SATP's multi-label action flags are kept next to the derived ACLED-style type.

Input: `scraped_data/scraped_incidents.rds` (`code/scrape_data_merge.R`).
Output: `scraped_data/satp_events.rds/.csv`.
Unit assignment (`unit_id`) is added downstream in the Irrigating Peace pipeline from `gid_2`, `latitude`/`longitude` or `shrid`.

## Columns

### Identifiers and provenance
| column | description |
|---|---|
| `event_uid` | unique key of this event row |
| `incident_uid` | key of the SATP entry from `scrape_data_merge.R`; shared by all events split from the same entry |
| `event_part` | position of the event within its entry (1 when the entry holds a single incident) |
| `date` | event date |
| `series` | pipe-separated SATP series tags (`india`, `india-maoistinsurgency`, ...) |
| `source_ids` | pipe-separated original `Incident_ID`s |
| `incident_summary` | whitespace-normalized SATP text of the entry |
| `event_text` | the part of the text that describes this event (equals `incident_summary` for single-incident entries) |

### Location (ACLED: `admin1`, `admin2`, `location`, `geo_precision`)
| column | description |
|---|---|
| `admin1` | state, GADM `NAME_1` spelling |
| `admin2` | district, GADM `NAME_2` spelling |
| `gid_2` | GADM district id; `NA` if no district was resolved |
| `location` | most specific place string extracted (village, town, block) |
| `latitude`, `longitude` | geocoded point; `NA` if not geocoded |
| `geo_precision` | 1 = village or town point, 2 = district only, 3 = state only or region |
| `n_districts` | number of distinct districts named in the text |
| `loc_method` | `gazetteer`, `llm`, `hand` (hand-coded training data) |
| `loc_ambiguous` | TRUE if a district name matched more than one district and was not resolved by a state mention |

### Actors (ACLED: `actor1`, `inter1`, `actor2`, `inter2`, `interaction`)
| column | description |
|---|---|
| `actor1` | perpetrator name as written (e.g. "CPI-Maoist", "Security Forces", "ULFA"); `NA` if the text names none |
| `inter1` | perpetrator type: `State forces`, `Rebel group`; `NA` if the text does not identify the perpetrator |
| `actor2` | target as written (`NA` if none) |
| `inter2` | target type: `State forces`, `Rebel group`, `Civilians`, `Government`, `Property or infrastructure`, `None` |
| `interaction` | `inter1`-`inter2` pair, e.g. `State forces-Rebel group` |
| `civilian_targeting` | TRUE if `inter2` is `Civilians` |

### Event type (ACLED: `event_type`, `sub_event_type`)
Derived from the action flags and `inter2` (rules below).

| column | values |
|---|---|
| `event_type` | `Battles`, `Explosions/Remote violence`, `Violence against civilians`, `Strategic developments` |
| `sub_event_type` | `Armed clash`, `Bombing`, `Attack`, `Abduction/forced disappearance`, `Looting/property destruction`, `Arrests`, `Disrupted weapons use`, `Change to group/activity` |

### Action flags (SATP multi-label, 0/1; an event can have several)
`armed_assault`, `bombing`, `infrastructure`, `abduction`, `arrest`, `seizure`, `surrender`

### Counts
| column | description |
|---|---|
| `fatalities` | total deaths |
| `injuries` | total injuries |
| `n_arrests`, `n_surrenders`, `n_abducted` | counts from the text |

### Quality flags
| column | description |
|---|---|
| `short_summary` | summary under 40 characters |
| `label_source` | `hand` for the ~10,000 hand-coded events, `model` for classifier output, `llm` for DeepSeek output |

## Derivation of the ACLED-style type

An event takes the first matching rule, top to bottom (order is open, see below):

1. `abduction` = 1 -> Violence against civilians / Abduction/forced disappearance
2. `bombing` = 1 -> Explosions/Remote violence / Bombing
3. `armed_assault` = 1 and `civilian_targeting` -> Violence against civilians / Attack
4. `armed_assault` = 1 -> Battles / Armed clash
5. `infrastructure` = 1 -> Strategic developments / Looting/property destruction
6. `arrest` = 1 -> Strategic developments / Arrests
7. `seizure` = 1 -> Strategic developments / Disrupted weapons use
8. `surrender` = 1 -> Strategic developments / Change to group/activity

The analysis indicators (armed assault, bombing, infrastructure, abduction) are read from the flags, not from the derived type, so the priority order does not affect counts.

## Mapping from the existing hand-coded data

`data/satp_classification.csv` columns map as: `perpetrator` -> `inter1` (Security -> State forces, Maoist -> Rebel group, Unknown -> `NA`); `civilians`, `security`, `maoist`, `government_officials`, `private_property`, `government_infrastructure`, `non_maoist_armed_group`, `no_target` -> `inter2`; action columns -> flags; `state`, `district`, `block`, `village_name`, `latitude`, `longitude` -> location. In that file the `latitude` and `longitude` headers are swapped (the `longitude` column holds latitude values), so they must be swapped back on read.

## Coding rules

- **Perpetrator.** `actor1` and `inter1` are filled only when the text identifies the perpetrator. Otherwise both are `NA`; the classifier's "Unknown" is not carried over and no class is imposed.
- **Multi-incident entries.** An entry that describes truly separate incidents (different attack, place or time) is split into one row per incident. Each row keeps the entry's date unless the text gives a different one, and takes its own location, actors, flags and counts from its `event_text`. Several mentions of one incident (e.g. the attack and the security forces' response) stay one row.

## Open decisions

1. **Priority order** for the derived type (above is a default).
2. **Splitting method.** Rule-based splitting cannot separate incidents reliably; this step needs an LLM (DeepSeek) run on entries flagged as long or containing several dates, places or verbs of violence.
3. **Events naming only a state or region.** Kept with `geo_precision` = 3 and `gid_2` = `NA`.
4. **Target classes.** `inter2` collapses SATP's ten target categories into six; confirm the collapse.
