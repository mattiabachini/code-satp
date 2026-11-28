# SATP event and location pipeline: draft plan

Draft, 2026-11-24. To be refined with the whole pipeline; no prompts written yet.

## Goals, in order

1. **Only real events count as events.** A text that recalls an earlier act, reports investigation or court news about it, an arrest or identification "in connection with" an old incident, a statement, threat or anniversary is not a new event. False events go straight into the conflict counts, so this comes first.
2. **The most accurate location possible.** Village (SHRID) where the text allows it, otherwise police-station area, tehsil, district. The target unit is SHRID, because the village-level group data it will be joined to is SHRID-linked; 2001 census village codes come from the SHRUG keys.

## Working method

- The 100-event check sheet (`data/intermediate/validation/place_check_hand.xlsx`, simple random draw from the 500 pilot events, weights to the analysis set in `place_check_key.csv`) is the development set.
- Mattia's hand-checked rows are the benchmark. Claude and Haiku agents label the remaining rows; disagreements and unclear cases go back to Mattia. A stronger model re-checks where Haiku disagrees with the benchmark or is unsure.
- Changes to the pipeline come from patterns found in these checks, not from more hand checking.
- Claude writes the prompts, Mattia passes them to the DeepSeek agent, Claude reviews the results.
- Every API step is cached by text hash, so reruns cost nothing and interrupted runs resume.

## Stage 1. Structured extraction (one DeepSeek call per text)

Split each text into labelled pieces, each copied word for word from the text, so later checks run cheaply without another model call. Draft fields per act:

- `status`: new / recalled (background) / follow_up (investigation, court, arrest or identification about an earlier act)
- `trigger`: the words that state the act ("killed in an encounter")
- `action`: armed_assault, abduction, bombing, infrastructure, surrender, arrest, seizure
- `actor`, `target` (as written), counts (killed, injured by group, arrested)
- `date_text` ("on December 1")
- `location`: site (with type: village, town, locality, forest, camp, road, water), police_station, subdistrict, district, state; plus `within` relations
- `items` seized or used ("two Chinese grenades", "several extortion notes")
- text-level: `source` ("According to media reports"), number of separate acts
- **Semantic flags** (from Mattia's notes): abduction vs later release, rescue or escape (a release is not a new abduction; geocode the abduction site, not the release site); body recovered vs killing (the body may be found away from where the person was killed); surrender and arrest (often reported at a police or district headquarters, not where the group operated). Each act gets a `location_role`: where the act happened / where it was reported or processed.
- **Multiple acts:** careful splitting when a text reports several incidents ("in another incident"), each with its own location and date (builds on `code/process/split_multi_incident_entries.R`).
- **Multiple dates:** flag texts with more than one date, relative dates ("on Tuesday", "last week", "the night of") and act dates that differ from the report date.

Example: the NDFB encounter in Kakulon reserve forest, Bongaigaon (Dec 1) gives actor "the Army", target "two NDFB terrorists", trigger "killed in an encounter", site "Kakulon reserve forest" (forest), district "Bongaigaon district", state "Assam", date "on December 1", items as listed.

**Additions after reading 50 random check-sheet events (2026-11-24):**
- **Non-event statuses, more specific:** judicial (a 2007 death sentence for a 2005 killing), protest (demonstration against an earlier killing), threat (bomb threat by email, posters), statement or denial (a group's press release denying the encounter), symbolic act (Khalistan flag hoisted), aggregate or background figures ("at least 20 abducted in the last eight months"), identification of an attacker after the fact. Each is a reason kept with the non-event.
- **Act outcome:** successful / foiled / failed / defused (IED missed the vehicle, bomb defused on the track, landmine escaped).
- **Action types missing from the list:** cross-border shelling or ceasefire violation, sexual violence, beating or ransacking without injury counts, lynching by a mob, extortion.
- **Actor types missing:** foreign armed forces (Pakistan Army, Rangers), mobs or civilians, village defence committees, rebel-on-rebel clashes (Maoist vs PLFI). Attribution certainty: named / suspected / claimed / denied / conflicting (HuJI vs LeT).
- **Counts with uncertainty:** ranges and revisions ("80 ... 60 ... 85 dead", "around a dozen", "at least one more", "earlier reports claimed three"); keep low, high and the attributed source, and flag casualties outside India (Pakistani Rangers).
- **Location relations:** at / near / between X and Y / direction from X ("south of Khonsa") / along a border / upper reaches of; the second anchor kept as its own piece.
- **More site types:** market, religious site, school, hospital, government office, security base or post, railway line or station, project or mine site, border sector, border pillar, LoC.
- **More administrative levels:** sub-division (common in the Northeast: "Maibong sub-division"), "area" names ("Imamsahab area", "Gool area").
- **Location roles beyond site:** victim's home or posting ("of Naira", "posted at Srigufwara"), hospital or treatment ("shifted to Srinagar hospital"), operation base ("set off from their camps"), background place; none of these is the act's location.
- **Several sites for one act:** serial blasts at six places in Jaipur; raids across several villages in one night (Pulwama). Keep a list of sites per act, and decide whether each blast or raid is its own act.
- **Vague locations:** "across Doda district", "a forest in Gariabandh District", "a village in Nongkyllem" give no site; precision stops at district or tehsil.
- **Districts named in the text that differ from the 2001 unit** are common (Shopian, Sukma, Seraikela-Kharsawan, South West Khasi Hills, Gariabandh), which confirms the LGD-based district check in stage 2.
- **Time:** date of the act vs date of a consequence (shot on May 12, died May 13), date ranges ("October 17 and October 18"), time of day ("wee hours", "evening").
- **Same incident in several texts:** "as reported earlier", corrections a day later; a cross-text check on the extracted date, site, counts and actor should link duplicates (added to stage 2).
**Additions after a second, targeted batch of 50 (from the 400 pilot events not in the check sheet; weighted to Punjab/other, multi-incident texts, police-station or block mentions, the 2020s and long texts):**
- **District assigned from the wrong place:** SATP gives no district. Our matcher (`code/match_event_districts.R`) takes the first district named in the text. So the Red Fort blast in Delhi (SATP130175) gets Pulwama, the car owner's home district. The act's site in the text must replace the matched unit when they disagree, not only flag it.
- **More non-event reasons:** a minister's speech with violence statistics, forensic and CBI or NIA findings about earlier blasts, a case handed to the CID, a government policy decision (withdrawing ceasefire agreements), a group split announced. Also **fragments**: "the blast appeared to be the handiwork of..." describes no act on its own; it should be linked to the other parts of its incident (`incident_uid`) and not coded alone.
- **Cases needing a decision (Mattia):** BSF firing at Pakistani drones and drones shot down; smugglers fired at with heroin recovered; old rusted grenades found and defused; clashes between police and protesters with injuries; killings inside a group (internal feuds, a faction formed by killing its commander). Are these conflict events, and of which type?
- **Running totals:** "taking the toll to 32" and recaps of earlier days' attacks in the same text; only the new deaths belong to the act.
- **Acts outside India launched from India:** the Army raid into Myanmar (camps 11 km inside the border), with casualty estimates revised several times (30, 38, 68 killed; 7 bodies recovered).
- **Location relations with distances:** "about 100 kilometres from Ranchi", "half a kilometre from the Chief Minister's residence", "11 km inside Myanmar".
- **More location roles that are not the site:** a militant's area of activity ("active in Doda town, Koti, Bharat..."), districts where he was wanted, where recruits came from, where curfew was imposed.
- **More levels and units:** police districts (the "Bijapur police district" in 2007, before it became a revenue district), police ranges and divisions ("Palamu Range", "Bermo Division"), forest divisions ("Kolhan Forest division"), revenue divisions ("Bastar division").
- **More site types:** national highway and toll plaza, bridge or construction site, mobile tower, border outpost (BOP), border sector ("Abohar Sector"), national park ("adjacent to the Manas National Park"), river as an anchor.
- **Targets with roles:** sarpanch, party activist, school principal, alleged police informer, contractor's workers; useful for civilian-targeting measures.
- **Disputed accounts:** alleged fake encounters and protests disputing the police version; flag the act as disputed.
- **Dates:** "the intervening night of October 28 and 29", "(today)", a body found on a date when the killing date is unknown; two acts on different days in one text (an encounter on December 24, protest clashes on December 25).
- **SATP's own corrections:** "SATP had earlier reported that five militants and seven SF personnel were killed" links directly to an earlier entry.

- Sheet issue: the check sheet has 102 rows for 100 events; SATP007602 and SATP015628 (check_ids 45 and 57) appear twice.

Open decision: replace the two current prompts (event coding in `code/process/code_events_deepseek.R`, place extraction in `code/sandbox/pilot_place_extraction.R`) with this one combined call, or run it alongside them. Leaning towards one combined call. Rough cost: $0.20-0.30 per 1,000 texts, ~$10 for the analysis set.

## Stage 2. Cheap deterministic checks on the extracted pieces

- **Verbatim:** every piece must be a substring of the text; otherwise flag (catches invented places or facts).
- **Date:** compare `date_text` with the report date; an act dated weeks or months before the report is old even if `status` says new.
- **District:** compare the district named in the text with the SATP-assigned 2001 unit (aliases from `district_aliases.csv`); a mismatch flags a district-assignment error.
- **Counts:** check counts against the trigger and target (e.g. "killed" with zero fatalities).
- **Actors:** standardise actor and target names with a lookup list.
- **Duplicates across texts:** link texts that report the same incident (same date, site, counts and actor; "as reported earlier", next-day corrections), so one incident is counted once.

## Stage 3. Event filter

- A text is an event if at least one act has `status = new` and passes the date check.
- A text that recalls an old act and also reports a new one is an event, dated to the new act.
- Current gap: the analysis set is selected by the DistilBERT action classifier (`event_confidence` > 0.7); DeepSeek's non-event rule has run on only ~2% of it, and ~6% of those were non-events (roughly 2,000-2,500 of ~40,700). Example: SATP093603 (NIA gives up DNA identification of the Pathankot attackers) sits in the analysis set as a bombing.
- The 300-event hand validation (`data/intermediate/validation/validation_summary.csv`, drawn from this same analysis set) found 8% non-events in J&K and the Northeast, 13% Maoist, 30% Punjab, 48% other: about 10% weighted, ~4,000 events.
- New gate: the stage 1 extraction says at least one act is new, it is an analysis action, and it has at least a district (gaps filled from the extracted district and LGD before dropping). DistilBERT stays as a cross-check; disagreements go to review. Run stage 1 on all 91,555 events, not only the current analysis set (~$20-27).
- Before switching the paper to the new gate: compare district-year counts under the old (conf07) and new gates, and keep conf07 as a robustness check.
- Hand-check sheet gets an `is_event` column, so the 100 also measure the non-event rate; non-events stay out of the location precision figures.

## Stage 4. Location resolution

Inputs: only the location pieces from stage 1, matched inside the event's 2001 unit.

- **Gazetteer by place type:** villages/towns: SHRID, GeoNames P, OSM settlements; forests, roads, water: OSM and GeoNames terrain classes; sub-districts: SHRID sub-districts, GeoNames ADM3+, OSM admin level 6 polygons.
- **Police stations as their own level** (between village and tehsil): match the station to OSM police features (~5,000) and to the town or village it is named after, within the district; possibly official police-station lists for the Maoist districts. A police station never becomes the event's village (lesson from SATP057610: "Khiritand village under Rajpur PS" was proposed as Rajpur village).
- **Same-name places:** use the `within` relation (police station, block) to pick the right one; Chatra has three villages called Rajpur.
- **Exact over fuzzy;** fuzzy thresholds (Jaro-Winkler, length ratio) to be set from the hand checks, not guessed.
- **Every candidate mapped to a SHRID** (point-in-polygon for GeoNames/OSM points) and to the 2001 census code; flags for city-wide SHRIDs (`shrid_urban`) and district-headquarters names (`district_hq_name`).
- **Tiers:** shrid (2+ sources share a SHRID) > high (2+ within 5 km) > single (one exact) > police station / tehsil > single_fuzzy > conflict > district only.
- Google Maps geocoding only as a last resort, for names the free sources cannot resolve or disagree on.

### Sources (from Mattia's notes, with what each adds)

| source | in hand | what it adds |
|---|---|---|
| OSM (Geofabrik) | yes | settlements incl. hamlets/paras, forests, roads, police features; tehsil (admin level 6) and some village (level 9) polygons |
| SHRID / SHRUG | yes | target unit; village polygons and names; keys to 2001 and 2011 census codes |
| GeoNames | yes | extra spellings and alternate names, terrain features |
| Census of India village directories (2001, 2011) | partly (via SHRUG) | official 2001 and 2011 names, sub-district names; spelling variants between censuses |
| Local Government Directory (LGD) | no | current villages, blocks, gram panchayats and sub-districts with LGD codes, a key to 2011 census codes, and records of renamed or split units; blocks are what many texts name ("X block") |
| India Post pincode directory | no | post office names (usually the village or town they sit in) with coordinates, district and taluk; a fourth gazetteer that covers many small places |
| Open Government Data (data.gov.in) | no | possibly police station lists by district (some with coordinates or jurisdiction); to be searched |

### Check cascade: does the point fall where the text says?

Every resolved point goes through nested checks, from coarse to fine. A failed check downgrades the event's level; it never silently keeps a wrong village. The result is stored per event as a vector of passed checks plus the final level.

1. **State:** the point lies in the state the text names (or the SATP state).
2. **District:** the point lies in the event's 2001 unit polygon (1-2 km tolerance at borders); a mismatch also checks the district named in the text, which flags SATP district-assignment errors.
3. **Tehsil / block:** if the text names a tehsil or block, the point lies in that polygon (OSM level 6, SHRID sub-district, LGD block once added). Tolerance because 2001, 2011 and current boundaries differ.
4. **Police station:** if the text names one, the point lies within a plausible distance of the station (no station polygons, so distance as a proxy, e.g. 15-20 km, to be set from the data), and in the same tehsil where that is known.
5. **SHRID polygon:** the SHRID polygon containing the point has a name that matches the extracted name (exact or close); a point from OSM or GeoNames that lands in a differently named SHRID is flagged (possible hamlet of that village, or a wrong match).
6. **Uniqueness:** an exact name match counts as a village only if it is unique inside the narrowest unit the text gives (police station area or tehsil, else district). About 10% of exact names repeat within a district, so "100% match" alone is not enough.

**LGD village file (downloaded 2026-11-24, `data/input/external/lgd/All_Villagesof_India_2026-11-24.csv`):** 677,707 villages with current state, district and sub-district (7,102) names and codes, plus 2001 and 2011 census codes (~94-95% have them). 92.7% of villages with a 2011 code link to a SHRID through SHRUG's `pc11r_shrid_key`; Chhattisgarh 97%, Jharkhand 91%, Bihar 87%. It lets checks 3 and 6 run on codes instead of polygons: "is the matched SHRID listed under the sub-district the text names?" and "is this name unique in that sub-district?", with no boundary tolerance needed. It also maps current district and tehsil names (e.g. districts created after 2001, such as Sukma or Bijapur) to their villages, so a text naming a new district can be checked against the 2001 unit. About 8% of linked villages are spelled differently from SHRID (Bandekarwadi / bandakerwadi), which adds spelling variants for matching. It has no blocks; blocks need LGD's block-village mapping download.

Fallback order when a check fails: SHRID village -> police-station area -> tehsil polygon -> district. Each step records why it fell back.

### Police stations: how to match them (open question in Mattia's notes)

Candidate approaches, to compare on the check sheet:
- **Name of the seat:** most stations are named after the town or village they sit in; match the name to settlements inside the district and prefer towns or larger villages.
- **OSM police features:** `amenity=police` points named "X Police Station" or "X PS" inside the district.
- **Official lists:** state police and data.gov.in station lists (some give the station's villages or coordinates), starting with the Maoist states (Jharkhand, Chhattisgarh, Odisha, Bihar, Maharashtra).
- **Agreement:** a station whose seat-village match, OSM police point and official list agree is a confident station location; the station's tehsil then becomes the fallback location for the event.

## Stage 5. Validation and scaling

- Metrics on the 100 (weighted to the analysis set): non-event precision and recall; share of events at village, police-station and tehsil level; location accuracy by tier.
- Iterate stages 1-4 on the 500 pilot events until the metrics are acceptable.
- Then run on the full analysis set (~40,700 events), then on all 91,555.

## Lessons from the hand checks so far

- SATP057610: police station proposed as the village; the site village (Khiritand, likely a hamlet) is in no gazetteer.
- SATP093603: investigation news about an old attack classified as an event; also the date is the report date, not the act date.
- Manual OSM searches can land on a similar name in another state (Raipur, Chhattisgarh for Rajpur, Chatra); check the district and state in the result.
- District spelling variants (Budgam/Badgam, Ahmedabad/Ahmadabad) were treated as places.
- Some SHRIDs are whole cities (Srinagar, Ahmedabad), which are not village-level.

## Current estimate (pilot, before the latest fixes)

Of the ~40,700 analysis events: ~25,500 placed in a SHRID village, of which ~18,300 confirmed by two sources or an exact match. Precision still to be measured.

## Status of earlier prompts to confirm

- Resolver fixes (district aliases, exact over fuzzy, len_ratio, urban flag, tehsil check A5) and the police-station-never-a-village fix: check whether they ran before the check sheet was built.

## Codebook from a 500-event read (2026-11-24)

The extraction schema now has a codebook and a JSON schema built from 500 random event parts drawn from all 131,029. The files:
- `notes/satp-codebook.md`
- `notes/satp-extraction-schema.json` (v1)

The draw and the readings are in `data/intermediate/validation/`:
- `codebook_read_500.csv`
- `codebook_read_500_reading.csv`
- `codebook_read_500_passA_notes.md`
- `codebook_read_500_passB_coding.tsv`

The codebook replaces the field list in Stage 1 above wherever the two differ. Its section 6 lists the location checks to merge into Stage 4: nesting, district mismatch, the newspaper's home state, responding police station, distances and the satp_clean benchmark. Its section 7 compares the reading with conf07.
