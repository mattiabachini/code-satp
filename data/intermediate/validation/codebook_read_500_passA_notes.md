# Pass A notes (open reading)

## Batch 1 (1-50)
Status counts (my reading): new violent act 17 [2,7,8,16,20,24,26,30,31,32,34,38,40,42,43,44,(28)]; new arrest/recovery 13 [3,4,5,6,9,10,13,27,35,36,37,39,50]; non-event 20 (statements, policy, diplomacy, meetings, bandh called/peaceful, denial, appeals, media protest) [1,11,12,14,15,17,18,19,21,22,23,29,33,41,45,46,47,48,49 + 25? no 25 = arrest].
- 25 is arrest (new) + recalled killing of SP (2014) -> recalled act embedded.
- Gate: nearly all arrests/recoveries are OUT of conf07 even with p=0.96-0.98 -> check how "analysis action" is defined (arrest/seizure excluded?).
- 40 is a 2025 killing, outside conf07 (year cut?).
- 26: accidental blast from object picked up on railway track (old ordnance case) - no actor, no district in text.
- 13: hawala cash seizure before elections - pure crime, no group (crime).
- 36: FICN smuggling arrest, ATS - crime with security framing (crime-linked?).
- 30: SIMI suspect shoots ATS constable + civilians; motive = revenge for crackdown -> conflict.
- 7: MLA shot by Maoists; target role = MLA, candidate, party CPI-ML (target_affiliation).
- 24: disputed account ("fake encounter" claim by villagers).
- 16: multiple suspected perpetrators (KPLT, UPLF), attribution dispute; police station named as the victim's posting, not the site ("SP of Hamren Police Station"); site = Rongfu along Assam-Meghalaya border; "night of June 5" bodies recovered June 6.
- 42: police unit named by district ("Bishnupur police commandos") but site = Thoubal area -> unit's home district is not the site.
- 34/40: sector/LoC names (Nowgam sector, Machil sector, Toot Mari Gali) - border sectors as location level; "today" date.
- 9: "resident of Seelo village" = origin, not site; two newspapers (Financial Express, Greater Kashmir).
- 6: "Sonhat village under Raghunathnagar Police Station limits in Balrampur District" = ideal village+PS+district chain.
- 28: "hillock under Bhandaria police station, adjoining Garwah district" - PS + adjacency.
- 17: bandh across Magadh division (list of 5 districts) - region-level location; bandh passed peacefully = non-violent strategic development.
- 37: Border Pillar number 86 along Indo-Myanmar border - border pillar as location.
- 44: Maoists loot fair price shop - looting/property; "July 25 night".
- 41/45/46/48/15: foreign/diplomacy, outside India -> non-event, in_india=no.
- 33: group denies an earlier SF claim -> statement about older act (links to earlier event).
- 19: media protest against AR notification -> protest? peaceful, non-violent; by newspapers.
- 49: bandh by ILP agitating groups (KSU) - civil society bandh, not armed group.
- 27: "on an unspecified date" -> date unknown flag.
- Supporting actors seen: 5 Police+Army; 9 CRPF+SOG; 16 KPLT+UPLF (suspected); 25 Army+Police; 42 police commandos+CRPF; 50 Police+Army.
- Newspapers present in ~35/50.

## Batch 2 (51-100)
Status: new violent 17 [52,53?,58,59,63,64,66,72,83,87,89,91,93,95,97,99, 57 camp raid]; new arrest/surrender/recovery 17 [55,61,67,69,71,73,75,80,81,82,84,85,90,92,94,96,100]; non-event 14 [51,56,62,65,68,70,74,76,77,78,86,88,98, 60 mostly outside India]; ordnance 54; strike 79.
- Gate (conf07) false positives: 54 rusted grenades from digging; 78 GOC statement recalling Lolab killings. 53 = body of militant injured in earlier encounter (follow-up death -> risk of double count).
- ACLED fit: IED found/defused (55,67,71,73,85,96) = Strategic development / Disrupted weapons use. Camp destroyed (57) = Battles? no: SF raid on empty camp = Strategic development (Disrupted weapons use / seizure). Peace talks (86) = Strategic development / Agreement (non-violent) -> keep as event type but outside violence set. Strike enforced (79) = SATP bandh type.
- Outside India: 60 (Nepal Police arrests in Jhapa + Indian Police in civil attire in Nepal), 62 (joined IS in Syria), 76 (Dhaka airport). Need in_india per act and "Indian actor acting abroad".
- Crime vs conflict: 69 Naxal impostor extorting in the name of CPI-Maoist/TPC = crime posing as group (code crime, flag "claimed affiliation false"); 76 FICN = crime; 84 Pakistani spy = espionage (state vs state, conflict-adjacent; new sub-type).
- Location roles new: posting/base of victim or unit (72 "posted at Mazgaon", 96 "stationed at Ramagiri", 16 "SP of Hamren PS"); FIR registered at another PS (69 Gidhaur PS vs site under Kunda PS); origin/residence (75 "of Matirai Roaja Para under Salema PS" vs surrendered at B.C. Para; 64 militant resident of Shalidara); earlier-crime location (92 arson at Kharaundh, Nawada vs arrest at Lausinha, Gaya).
- Location levels new: mohalla/locality (52 Astan Mohalla of Awanura, 59 Lone Mohalla); gram panchayat (96 Gupteshwar Panchayat under Baiparaguda block); mandal/"administrative division" (89 Komarada); subdivision (69 Lawalaung, 83 Margherita); tehsil (97 Surankote); hill (83 Mulong Pahar II, 96 Ghodaghat Hill); riverbank (96 Saberi); forest named after villages (67 forests of Kodomali, Ichradi...; 81 Kajra forest); relational "south of Ledo" (83); interstate border (63 Assam-Arunachal border, PS Bordumsa is in Arunachal; 87 Jharkhand-Chhattisgarh border) -> state ambiguity.
- Multi-actor sides: 52 3RR+55RR+SOG; 57 DRG+STF+ITBP; 81 CRPF+STF+District Police; 87 CRPF+JJ+local police -> often 3 state units: keep 2 slots + "other_actors" list? (Mattia asked for 2 per side; 3+ is common for joint SF teams -> collapse to "joint state forces" with unit list).
- Multi-act texts: 66 two encounters, two districts; 99 abduction then killing; 87 landmine + firing; 85 two dates two districts; 60 several arrests on different dates.
- Dates: "According to a delayed report" (63,83); "media reports of April 18" (80) -> report date not act date; 91 no date in text; 52 "late on August 12"; 64 "night of April 4".
- Counts provisional: 52 bodies yet to be recovered; 59 operation continuing.
- Sources: kalingatv.com, The Print, lagatar.in, ommcomnews.com, opindia.com, Manipur Police on X, PTI, IBN Live, First Post -> local web outlets + official social media; need source_type (newspaper, agency, official, web portal) and a local/national scale table.
- No location at all: 94 (KLO surrender before STF; no state) -> SATP metadata only.

## Batch 3 (101-150)
Status: new violent 18 [101,102,106,109,111,118,119(part 2 confrontation),122,129,132,134,135,138,139,141,144,147,149]; new arrest/surrender/recovery/release 9 [104,112,116,120,121,126,130,143 release,145]; new non-violent group activity 4 [103 bandh observed, 108 blockade continuing, 128 martyrs' meeting, 133 shutdown protest]; non-event 19 [105,107,110,113,114,115(call),117,123,124,125,127(call),131,136,137,140,142,146,148,150].
- conf07 false positives: 114 (forensic result on a 2010 bomb), 146 (IGAR statement about IEDs smuggled), 128 (martyrs' memorial meeting, not violent). 119 is a statement + a new confrontation (mixed text).
- conf07 misses (violent, not in set): 144 blast of forest shed (p .54), 147 rifles looted from forest camps (p .95), 143 hostage released (follow-up).
- New event types: ceasefire violation / cross-border shelling Pakistan vs India (134) = state vs state; custodial killing "while trying to escape" (122, disputed-type); internal group discipline/court-martial (136); disowning arrested cadres (117); martyrs' meeting held despite combing (128); economic blockade (108); leaflets claiming responsibility (109) -> claim field.
- Motives stated: revenge for a killed Naxalite (109); unpaid extortion (141); informer suspicion (132, recalled). -> motive field (verbatim) helps crime vs conflict.
- Actors: "unidentified miscreants" (149) vs "unidentified militants/terrorists" -> actor wording matters for crime/conflict; "Pakistan" (134) as state actor; ULFA faction unknown (147); pro-talks NDFB leader shot by unidentified militants (118, likely inter-factional).
- Targets with roles: forest guards, village chowkidar (109), NSS member & block president of Manjhi Pargana Mahal (111), contractor, former village headman (141), bank branch manager (143), businessman's son (101), widow of alleged informer (132).
- Location: tehsil written "village X of tehsil Y" (104), "X tehsil (revenue unit)" (132); revenue circle Goreswar (138); subdivision (116); "unspecified place in Mizoram" (121) -> state only and not SATP state; reserve forest + named forest camps (147 Dibru-Saikhowa); "jungle near Lepung area" (141); LoC sector + list of villages hit (134); FIR at PS + handed to another city's police + court location (112) -> roles fir_ps, court, custody; origin village vs arrest site (112, 145).
- Fragments: 130 "In the second incident", 141 "In the first incident" only first described.
- Dates: "According to delayed reports" (141); report date given separately (149 "reports ... on August 22"); "September 15-evening".
- Abduction lifecycle: abducted (101), family says no demand (105), released (143) -> link by incident.

## Batch 4 (151-200)
Status: new violent 9 [165,168,169 protest clash,171,172,175,178,199,200]; new arrest/surrender/recovery/release 16 [153,155,160,176,179,181,182,183,184 release from custody,185 arrest(+recalled abduction),186,187,188,194, 173 arrest (crime)]; new non-violent group activity 3 [167 strike observed, 191 martyrs' week banners, 166 recruitment camps (weeks, vague)]; non-event 22 [151,152,154,156,157,158,159,161,162,163,164,170,174,177,180,189,190,193,195,197 (strike called),198 (expulsion)]; outside India acts 2 [192 Nijjar in Canada, 196 Indians abducted in Mali].
- Low violent share in random draw (vs conf07) -> most raw parts are non-events or arrests.
- Crime-like: 175 "counter-insurgents" (pro-govt militia/Ikhwan) loot bank at gunpoint -> militia actor, crime motive, but armed group -> crime_linked_to_group; 182 Pakistani drug boat off Porbandar (narco; ATS+NCB); 187 BSF narcotics recovery (Punjab drone-type); 176 FICN via ISI; 173 extortion by student-organisation member.
- Judicial/administrative non-events: acquittal (174), bail (177), NIA property attachment (159), banning (195), advisory (151), CBI inquiry decision (156), release from CID custody (184, borderline).
- Search operation launched with no outcome (189) -> non-event (no act yet).
- Location: landmarks/institutions (165 General Bus Stand Anantnag, 155 FCS godown area, 199 RIMS Sweepers' Colony, 174 Nuasahi Church); market/chowk (160 Mehasi chowk, 181 Shivpur Bazar); nullah + named forest + LoC sector + tehsil (178); sea "off Porbandar coast" (182) -> offshore; place spanning two districts (172 Langol "in Imphal West and Thoubal districts"); PS roles: site PS (181 Navinagar), native PS (181 Kutumba), FIR PS (181 Mali, Kutumba); villagers "of Narlapat village, from Thuamul-Rampur PS limits" (168, origin = probable site).
- Dates: "this afternoon" (165); "on Friday [July 29]" (191).
- Sources: X posts (187 BSF on X), Webindia123, latestly.com, News 18, Chennai Online.
- Target roles: police informers (200), villagers, rickshaw driver, constable on duty, NSCN-IM activist among victims (172).

## Batch 5 (201-250)
Status: new violent 17 [205,214,215,217,219,221,226,233,234(identification follow-up),235,239,240,241,243,244,249,250]; arrest/surrender/recovery 20 [201,202,206,207,208,209,210,211,212,216,218,220,224,225,228,229,231,232,242,247]; non-event 13 [203 court,204 threat,213,222,223,227 boycott planned,230,236 alert,237,238,245 investigation,246 released,248].
- conf07 misses: 226 (two more killed in continuing gunbattle, p .97), 240 (3 killed, delayed report, p .87).
- conf07 double-count risk: 234 identifies militants killed the day before (follow-up), 226 continuing battle reported per day.
- Crime vs conflict: 209 pseudo-ULFA extortion (crime, impostor); 229 student union leaders extorting for GNLA (crime_linked, funds group); 216 FICN racket (crime); 244 ex-Maoist killed "on suspicion of police informer" but SHO says old rivalry possible (unclear).
- Actors: 211 five SF units on one side; 201 mass surrender under named policy; 225 "militant of unspecified group"; 235 unknown perpetrator (body with bullet, hands tied).
- Location: direction relation 215 "East of Mongjam under Sogolmang PS"; mine 249 "Aamdai Ghati iron ore mine area under Chhotedongar PS"; urban 214 "Bulbul Lankar in Rainawari area of downtown Srinagar"; activity area 201 "involved in violent activities across Kuemari and Kiskoda areas"; body kept at PS 235 (North PS Kohima); origin vs site 244 (left Gobi Korasi, body in Lethia Balchand), 212 natives of Guntur towns arrested in Bhopal; abducted "of Matonga Chowk" returning home (250, site unclear); "bordering Bhutan" 232; border sectors 239 Hiranagar, Samba; IB/LoC.
- Report date vs act date: 206 tweet April 4 about April 1 arrest; 234 reports March 10.
- Sources: Daijiworld, Prameyanews, devdiscourse.com, Siasat Daily, tweets by Assam Rifles/Eastern Army Command/Manipur Police.

## Pass A totals (250, raw random draw)
- new violent act ~78 (31%); arrest/surrender/recovery/release ~75 (30%); non-violent group activity (bandh enforced, blockade, martyrs' meeting, propaganda) ~9 (4%); non-event ~82 (33%); act outside India ~6 (2%).
- conf07 (79 of 250 in set): clear false positives 6 (54 ordnance, 78 statement, 114 forensic, 128 meeting, 146 statement, 139 drone/narcotics borderline) + double-count risks (53, 234); clear violent misses outside conf07 at least 6 (26, 44, 144, 147, 226, 240) + 175, 143 borderline.
