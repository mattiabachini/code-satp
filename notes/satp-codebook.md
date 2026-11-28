# SATP event codebook, draft v1 (2026-11-24)

This codebook goes with `notes/satp-extraction-schema.json`, and nothing has been run with it yet. It is built from a reading of 500 event parts drawn at random from all 131,029 (seed 20261124; the draw is `data/intermediate/validation/codebook_read_500.csv`, my reading is `codebook_read_500_reading.csv`). Pass A was an open reading of 1–250, which produced schema v0. Pass B hand-coded 251–500 into v0 and logged every gap, which produced v1.

Examples are cited by SATP id, and the number in brackets is the row in the draw.

## 0. What the raw texts are (500 random event parts)

| Reading | Share | Notes |
|---|---|---|
| New violent act | 29% (145) | clashes, attacks, IEDs, grenades, abductions, looting, arson |
| New arrest, surrender, recovery, release or other non-violent act | 37% (186) | the largest group; arrests, surrenders and seizures are outside the analysis set by design; about 3 points are bandhs, blockades, martyrs' meetings and posters |
| Follow-up of an earlier act | 1% (6) | identification, death added later, search for bodies |
| No new act | 33% (163) | statements, policy, courts, talks, statistics, foreign news |

On a fresh read, a third of all event parts contain no new act at all. In the conf07 analysis set the share is much smaller (see section 7).

## 1. Is there a new act? (text_status)

- **Rule.** A text has a new act if it reports something that was done (a trigger verb) on or shortly before the report date, in India or by Indian forces.
- **No new act:**
  - statements, threats, diktats;
  - calls for a bandh;
  - policy, courts, bail, NIA charge sheets;
  - talks;
  - statistics and annual reviews;
  - alerts;
  - interviews;
  - foreign news;
  - searches with no outcome;
  - summaries of multi-day operations.
- **Text with a new act inside a non-event frame:**
  - SATP055017 [253] is court news only. SATP094618 [355] is an investigation story that reports two new arrests. Code the arrests.
  - SATP082101 [119] is a party statement that ends with a new confrontation between the public and security forces. Code the confrontation.
- **Recalled acts** go into `acts` with status `recalled`, so the model shows it saw them and later code drops them. For example, the GOC statement SATP060346 [78] recalls the Lolab killings.
- **Follow-ups never make a new event:**
  - identification of people killed the day before (SATP112679 [234], SATP064148 [329]);
  - a victim who dies later (SATP025371 [270], SATP024231 [454]), status `follow_up_adds_casualty`; the death is added to the original act;
  - a search for bodies after a clash (SATP061899 [313]).
- **A continuing battle is different.** "Two more terrorists eliminated in the gunbattle that began on January 9" (SATP022334 [226]) is a new act with new deaths, linked to the first report.
- **Fragments** ("In the second incident", "in the same district", a sentence with no verb) get `fragment = true`. They are read together with the other parts of the same incident_uid.

## 2. Event type and sub-type (ACLED plus SATP)

Use ACLED's six types. The SATP additions are sub-types, mostly under strategic developments.

| Situation in SATP | event_type | sub_event_type |
|---|---|---|
| Encounter, gunfight, ambush on security forces | battles | armed_clash |
| Pakistan Army ceasefire violation with firing | battles | armed_clash (interstate) |
| Pakistan shelling with mortars | explosions_remote_violence | shelling_artillery_missile |
| IED or landmine goes off | explosions_remote_violence | remote_explosive_landmine_ied |
| Grenade attack | explosions_remote_violence | grenade |
| Militants killed by their own device (SATP018024 [495]) | explosions_remote_violence | remote_explosive_landmine_ied, no actor 2 |
| Civilian shot, abducted, beheaded | violence_against_civilians | attack / abduction_forced_disappearance |
| Body found with bullet wounds, perpetrator unknown | violence_against_civilians | +body_found |
| Killed "while trying to escape" from custody | violence_against_civilians | +custodial_killing |
| Hostage freed or released | strategic_developments | +hostage_released |
| Stone pelting at a cordon and search operation | riots | violent_demonstration |
| Lynching of abductors by a mob | riots | mob_violence |
| Road blockade, sit-in, Black Day, student strike | protests | peaceful_protest |
| Arrest | strategic_developments | arrests |
| Surrender | strategic_developments | +surrender |
| IED found and defused, arms dump found | strategic_developments | disrupted_weapons_use |
| Arms or explosives recovered with no IED | strategic_developments | +arms_recovery |
| Camp or hideout raided and empty | strategic_developments | +hideout_or_camp_busted |
| Rifles looted, trucks burnt for levy, tracks blown up | strategic_developments | looting_property_destruction |
| Extortion call or demand note | strategic_developments | +extortion_demand |
| Bandh enforced by an armed group, trees felled on roads | strategic_developments | +bandh_or_strike_enforced |
| Infiltration bid foiled with firing | battles | armed_clash, plus +infiltration_foiled |
| Drone fired at or recovered | strategic_developments | +drone_interception |
| Old grenade or shell found or exploding (SATP120646 [54]) | not_conflict | +legacy_ordnance |
| Accord signed (BTC accord, SATP017176 [367]) | not an event (decided 2026-11-25) | none |
| Group threatens civilians over dress, religion, language or ethnicity (KYKL phanek diktat, SATP007625 [308]) | strategic_developments | +threat_or_diktat (decided 2026-11-25) |
| Group bans a newspaper or an organisation, with no threat to people (NDFB ban on Bodossa, SATP004156 [298]) | not an event (decided 2026-11-25) | none |

An act can have a second sub-type, for example an attack on police in which rifles were also taken (SATP085431 [278]).

## 3. Actors: two on each side

- Fill the four slots: actor 1, supporting actor 1, actor 2, supporting actor 2. Each slot has the name as written, an actor type (ACLED inter code), a short group name and the unit as written. Third and later actors go to `more_actors_verbatim`.
- **Side 1 is the side that started the act.** In an operation, the security forces are actor 1. In an ambush or attack, the attacker is actor 1. In a surrender, the surrendering group is actor 1. When the text does not say who started it, actor 1 is named first.
- Joint security teams with three or more units are common (SATP100572 [52]; SATP130900 [211] names five units). The first two named units go in the slots and the rest in `more_actors_verbatim`.
- Several groups can sit on one side (SATP022564 [333]: Al-Badr, LeT, TJI and HM killed together).
- **Attribution:** stated, suspected, claimed, denied, disputed or unidentified.
  - "Unidentified gunmen" and "miscreants" are `unidentified_armed_group`, not a rebel group.
  - A group's denial goes in `attribution` and `links` (SATP067223 [33], SATP080411 [117]).
- Civilians can be actors: residents caught a militant and handed him over (SATP073758 [393]). So can church leaders who receive a surrender (SATP013121 [446]).

## 4. Conflict, crime or neither (conflict_class)

The decision tree, applied to each act:

1. **Is a state force fighting another state?** (Pakistan Army firing or shelling, or an exchange across the LoC) → `interstate`.
2. **Is an armed group, its members or cadres, its overground workers, or state forces acting against them a party to the act?**
   - Yes, and the act is political violence, a counter-insurgency action or a group's own operation (including extortion collected for the group) → `conflict`.
   - Yes, but the act is ordinary crime committed by or for such a person (an arms dealer supplying "various groups", drug smuggling by a BKI member) → `crime_linked_to_group`.
3. **No group, or a group's name used falsely?**
   - Impostors extorting "in the name of" CPI-Maoist or ULFA (SATP122629 [69], SATP020210 [209]) → `crime`, with the false claim kept in `conflict_reason_verbatim`.
   - Fake currency, hawala cash, plain robbery, illegal possession of a pistol with no group named → `crime`.
4. **Not violence and not crime** (illegal migration, a welfare protest by refugees) → `not_conflict`.
5. **Anything else** → `unclear`.
   - "Unidentified miscreants" in Manipur, a former cadre killed with "old rivalry" not ruled out (SATP110907 [244]), and grenades exploding among children with no perpetrator all count as unclear.
   - Unclear is kept. The analysis can include or drop it.

**Former members acting alone** (SATP067951 [417], a former cadre making extortion calls) are crime unless the text links them to an active group.

**Pro-government militias** ("counter-insurgents" looting a bank, SATP021574 [175]) are an actor type of their own, `political_militia`. Their acts count as conflict when they target people because of the conflict, and as crime_linked_to_group when the act is plain theft.

The class stays a variable rather than a filter, so results can be shown with conflict only, with conflict plus crime_linked, and with unclear added.

**Decided by Mattia (2026-11-25):**
- **Peace accords and agreements are not events.** They are coded `no_new_act`, with reason `talks_or_negotiation_news`. The sub-type `agreement` is dropped.
- **Party-political violence counts as conflict.** Bomb-making, bomb attacks and clashes by party workers (Kannur, CPI-M against RSS, TMC against CPI-M) are `conflict`, with actor type `political_militia`. The motive is coded `political_or_insurgent`.

- **Identity threats and diktats count; plain bans do not.** A group's threat against civilians defined by ethnicity, religion, dress or language is an event: strategic development, sub-type `+threat_or_diktat`, class `conflict`, motive `ethnic` or `religious`. These acts belong to the between-group frictions set, which collects acts with an ethnic or religious motive. Example: KYKL warning women who do not wear the phanek (SATP007625 [308]). A ban on a newspaper or an organisation with no threat to people is not an event (NDFB and Bodossa, SATP004156 [298]). The Meetei Mayek signboard diktat (SATP040987 [387]) is in the check sheet to test where the line falls.

**Still open:**
- drones and narcotics in Punjab, which is still the borderline list from the earlier batch.

## 5. Places: level, role and relation

Every place named for an act gets three things:
- its **level**, from state down to locality, plus landmark, market, road, bridge or culvert, forest, hill, river, mine, border sector or pillar, sea, foreign place, police station, police outpost, group camp;
- its **role** (below);
- its **relation**: under, near, between, "4 km from", east of, bordering, along.

**Role is the key field.** Only `site`, `site_area`, `reference` and `route` locate the act. The other roles are kept so the location checks can use them, and so they are never mistaken for the site.

| Role | Example |
|---|---|
| site | "near Sonhat village under Raghunathnagar Police Station limits in Balrampur District" (SATP090040 [6]) |
| site_area | the police station in "under X PS limits", the tehsil, the block |
| reference | "near Jhargram town", "outside DM College", "on the outskirts of Siliguri" |
| route | "on way to Balumath from Chatra via Panki", "between Dania station and Jageshwar station" |
| responding_ps | "A team from Heingang Police Station retrieved the grenade" (SATP091445 [323]) |
| origin_residence | "a resident of Seelo village", "of Kukshi village under Fakiragram PS", "a native of Kutumba PS" |
| posting_or_base | "posted at Mazgaon", "stationed at Ramagiri", "the 134 CRPF battalion stationed in West Singhbhum" |
| fir_registered | "an FIR lodged at Gidhaur Police Station" (SATP122629 [69]) |
| court_or_custody | "produced in Khatra sub-divisional court" |
| body_found, body_taken_to, died_at | "bodies found on a village road at Bandhgora"; "kept at North PS Kohima"; "airlifted to Ranchi" |
| activity_area | "active in Kuemari and Kiskoda areas", "operated in the Lohit river islands" |
| earlier_act_site | where an arrested person's earlier crime took place |
| site_from_actor_unit | no place in the text, only "Delhi Police Special Cell" |

**Rules:**
- **"Of X":** a person "of X village" marks where they come from, not the site. When nothing else locates the act (SATP043716 [350], SATP098370 [168]), the origin may be used as the site, and the act gets a fallback flag and a lower precision.
- **Surrenders** at a district police HQ or before the SP are located at that office, and the group's activity area is kept separately.
- **Police-unit names** ("Bishnupur police commandos", "Thoubal District Police") give the unit's home district, which is not necessarily the site (SATP020503 [42]).
- **Borders:**
  - a place "on the Churachandpur-Tamenglong district border", on the "Assam-Arunachal border" or "along the Odisha-Jharkhand border" names two units;
  - both are kept, and the place can be resolved at most to the pair, unless a village is found inside one of them.
- **Sectors and pillars:** LoC and IB sectors (Nowgam, Machil, Keran, Tanghdar, Nowshera, Balakote, Hiranagar, Arnia), border pillars and border outposts are their own level. Each needs a sector → district lookup table.
- **Police districts** (Kulgam police district in 2006, SATP033697 [287]) and revenue units (tehsil, mandal, circle, gram panchayat) are named levels.

## 6. Checks the location pipeline can run with these fields (to merge into Stage 4 of the plan)

1. **Nesting check.** Every site place must fall inside its named parent:
   - village inside police station, police station inside district, district inside state;
   - checked with SHRID polygons, LGD codes and police-station points.
   - Example: Kaifundai Kuki village → Tousem PS → Tamenglong district.
2. **District from the act's site, not from the first district named.**
   - SATP provides no district field. The current unit_id comes from our own matcher (`code/match_event_districts.R`), which takes the first "X district" mention in the text, whatever role that place plays.
   - The act's site district should replace it. For example, the Red Fort blast in Delhi (SATP130175) is matched to Pulwama because Pulwama is the car owner's home district.
   - When the text itself names the wrong district, the gazetteer check flags it. For example, the text puts Nowhatta and Makhdoom Sahib in Baramulla, but both are in Srinagar (SATP011757 [473]).
3. **Newspaper check.** A list of newspapers with their home state and scale (local, regional or national) flags events whose resolved state differs from a local paper's home state.
   - It also helps when the text names no state, as in SATP069830 [325]: Tekang–Tizit river junction, Nagaland Post.
   - 42% of all texts name a newspaper. Local and web outlets are common (Sangai Express, Nagaland Post, Daily Excelsior, Dharitri, lagatar.in, Prameyanews, kalingatv). So are official X accounts (Manipur Police, Assam Rifles, BSF Punjab Frontier, Eastern Army Command).
4. **Responding PS:** if the text names the police station that responded, use it as the site's area when no "under X PS" phrase is given.
5. **Distance check:** "20 km from Panki PS" or "4 km from Mile station" must agree with the resolved point within a tolerance.
6. **Benchmark:** `data/final/satp_clean.csv` holds 9,921 Maoist events from 2005 to 2016 with village, block and coordinates. It can score the resolved places for the Maoist series in Stage 5.

## 7. What this reading says about the conf07 gate

My reading set against conf07 (157 of the 500 parts are in conf07):

| My reading | In conf07 | Not in conf07 |
|---|---|---|
| New violent act | 130 | 15 |
| Arrest, surrender, recovery and other new acts | 15 | 171 |
| Follow-up of an earlier act | 6 | 0 |
| No new act | 6 | 157 |

Of the 157 parts conf07 keeps, 130 (83%) are new violent acts.
- **Follow-ups (6):** identifications, a later death, a search for bodies. These are double counts.
- **No new act (6):** statements, a forensic finding, an election turnout report.
- **IED or arms finds (8):** these may belong in the set by design, through the bombing action.
- **Other (7):** two drones, an opium crop, old grenades, a commemoration, a martyrs' meeting and an empty camp.

conf07 misses 15 new violent acts. Three are from 2025, outside its years. Of the remaining 12:
- 6 have no district unit, so the text gives no district the matcher found (SATP015152 [423], an encounter with 11 killed in Beerwah, among them);
- 3 fall below the 0.7 confidence threshold;
- 3 are lootings, which are not an analysis action.

Within 2000–2024, conf07 catches about 92% of new violent acts. About 8% of what it keeps is a double count or not an event, and another 9% are non-violent strategic developments. The 300-event hand validation put non-events at about 10%, which agrees with this. Two fixes come out of it:
- the stage 1 status field (new, follow-up or recalled) removes the double counts and non-events;
- the place extraction gives a district for the texts that lost theirs.

## 8. How complete schema v1 is

- In pass B, 32 of the 250 texts carried a note.
- Of these, 17 texts needed a field added to v0. That came to 14 distinct additions:
  - status `follow_up_adds_casualty`;
  - the flags `attempt_failed` and `multi_site`;
  - a secondary sub-type;
  - the conflict class `not_conflict`;
  - the target fields `intended` and `identity`;
  - weapon `blunt_or_beating`;
  - date precision `announced_only` and a report date;
  - place levels `police_outpost` and `group_designated_camp`;
  - place roles `route`, `responding_ps`, `died_at` and `site_from_actor_unit`;
  - the no-act reason `operation_summary`.
- The other notes are resolution rules or decisions, not missing fields.
- With v0, 233 of the 250 texts (93%) coded with no missing field. With v1, all 250 code.

## Case check for Mattia (2026-11-25)

47 cases from the 500-event read are in `data/intermediate/validation/codebook_case_check.xlsx` (built by `code/sandbox/make_codebook_case_check.py`). The four groups are event or not, conflict class, location, and conf07 disagreements. The answers become decision rules and test cases for the prompt.
