# Task (draft): DeepSeek place extraction on the full analysis set

Status: draft, written 2026-11-23. Do not run before the 100-event hand check is scored and the prompt and resolver rules in `notes/todo.md` (items 3 and 4) have been decided, because a prompt change forces a full rerun (a few USD).

Folder: `code-satp` (work in `code/sandbox/`, do not commit or push).

## Background
- `code/sandbox/pilot_place_extraction.R` asks DeepSeek (deepseek-flash, prompt version 1) to extract the places of an event text and resolves them to SHRID places, GeoNames and OSM inside the event's 2001 district unit.
- `code/sandbox/resolve_places_multisource.R` is the second revision of the resolver. It reads DeepSeek answers from a cache only (no API calls).
- The pilot covered 500 events (cache `data/intermediate/places/place_cache_pilot.rds`). Do not modify or delete this cache, `data/intermediate/places/place_pilot_*.csv`, or `data/intermediate/validation/` (the hand check is in progress).
- The analysis set: events of `satp_events_conf07.rds` with confidence above 0.7, at least one of the four analysis actions and a unit; 40,674 events.

## Steps
1. Write `code/sandbox/extract_places_analysis_set.R` from the pilot script: same model and settings; prompt version from the decision in the todo (items 3 and 4); every analysis-set event; new cache `data/intermediate/places/place_cache_analysis.rds` (copy in the pilot answers only if the prompt version is unchanged); save after every batch; skip cached events.
2. Run on 200 new events first. Report the failure rate, tokens per event and cost. Stop and ask if more than 2% of calls fail or the projected cost of the full run is above 8 USD.
3. If that passes, run the rest. Print one line per batch with the clock time and a start and end line.
4. Report: events extracted, cache size, cost, failures; the share of events with no place and with a sub-district site, compared with the pilot; 10 examples where the model returns a state name, a district name, a person's name, or a release or origin place.
5. Only after the hand check is scored: the resolver on all events (copy `resolve_places_multisource.R`, keep the original unchanged), with `granularity` (village_town, tehsil_block, landmark, district_only), `match_type` (exact, fuzzy, spread, none), the source, the SHRID id, the centroid, the unit, and the coordinate checks of `notes/todo.md` item 1. Outputs `data/intermediate/places/place_analysis_set.csv` and `data/intermediate/places/place_analysis_set_places.csv`.

## Rules
- The API key is in the environment variable `DEEPSEEK_API_KEY`; never print or save it.
- Only add new files; write no data outside `data/`.
- If anything is unclear, stop and ask.
