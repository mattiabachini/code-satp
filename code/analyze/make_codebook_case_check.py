import csv, re, openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
base='/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/'
rows={int(r['read_order']):r for r in csv.DictReader(open(base+'data/intermediate/validation/codebook_read_500.csv'))}
fix=lambda t: re.sub(r'<U\+([0-9A-Fa-f]{4})>', lambda m: chr(int(m.group(1),16)), t)

E,C,L,G='1 Event or not','2 Conflict, crime or unclear','3 Location','4 conf07 disagreement'
cases=[
# --- event or not
(387,E,"A group's diktat on language (Meetei Mayek signboards), with no stated threat. Event like the phanek diktat, or not an event like the newspaper ban?","Event: strategic development, threat or diktat; conflict; ethnic motive (between-group frictions)"),
(270,E,"A victim of an earlier grenade attack dies three days later. New event, or a death added to the October 3 event?","Not a new event: add one death to the October 3 attack"),
(53,E,"Body of a militant injured in an earlier encounter is found. New event?","Not a new event: add one militant death to the November 18 encounter"),
(234,E,"Identification of two militants killed the day before, plus weapons found at the site. New event?","Not a new event: follow-up of the March 9 encounter; weapons go to that event"),
(226,E,"Gunbattle that began January 9 kills two more on January 11. One event over three days, or one event per day?","One event per day with new deaths; linked to the January 9 battle"),
(128,E,"Maoist martyrs' memorial meeting held in a forest despite combing. Event?","Event: strategic development, change to group activity; conflict; not violent"),
(421,E,"Maoist bandh called over four districts, with little impact. Event?","Event: strategic development, bandh enforced; conflict; site = Kandhamal district only"),
(342,E,"Week-long operation that 'cleared several villages' along the Myanmar border, no dated act. Event?","Not an event: operation summary"),
(376,E,"Police announce nine arrests over an unstated period in two districts. Event, and on which date?","Event: arrests, dated to the announcement (date precision: announced only); district = Anantnag and Pulwama, so state-level unless split"),
(166,E,"Maoist recruitment camps 'in the past few weeks'. Event?","Not an event: no dated act (intelligence or assessment)"),
(143,E,"ANVC frees a bank manager held for a week. New event?","Event: hostage released, linked to the abduction a week earlier"),
(306,E,"Kuki Black Day gathering commemorating 1990s killings. Event, and is it conflict?","Event: peaceful protest; conflict (ethnic, between-group frictions)"),
# --- conflict class
(263,C,"Civilian shot dead by 'unidentified gunmen' in Imphal West, no group named. Conflict or unclear?","Unclear"),
(244,C,"Former Maoist leader killed 'on suspicion of being police informer'; the SHO says old rivalry is possible. Conflict or unclear?","Unclear (disputed motive)"),
(417,C,"Former CPI-ML cadre makes extortion calls and has gelatine sticks. Conflict, crime linked to a group, or crime?","Crime (former member acting alone)"),
(69,C,"Impostor demands levy 'in the name of' CPI-Maoist and TPC. Crime?","Crime; false claim kept"),
(229,C,"Student-union leaders arrested for extorting on behalf of GNLA. Conflict or crime linked to a group?","Conflict (overground workers of an active group)"),
(175,C,"'Counter-insurgents' loot a rural bank at gunpoint. Conflict or crime linked to a group?","Crime linked to a group; actor = pro-government militia"),
(291,C,"BSF recovers a drone near the Punjab border, nothing else said. Event, and which class?","Event: drone interception; unclear"),
(396,C,"BSF recovers 559 g of heroin in a border village. Event, and which class?","Event: arms or drug recovery; crime (no group named)"),
(459,C,"BKI militant, also a drug smuggler, arrested in Mumbai with weapons. Conflict or crime linked to a group?","Conflict (named militant of an active group)"),
(374,C,"Grenade blast kills four children, no perpetrator named. Conflict or unclear?","Unclear (could be an old grenade picked up)"),
(395,C,"Mob lynches abductors; police do not know if they belong to a group. Class?","Unclear; riot, mob violence"),
(456,C,"BSF shoots two Bangladeshis crossing the fence to Pakistan. Class?","Unclear (border enforcement, no armed group)"),
(448,C,"Families killed by 'suspected counter-insurgents in Army uniform'; officials blame militants. Class?","Conflict; attribution disputed; actor 1 unidentified armed group"),
(432,C,"Rohingya arrested at Agartala station. Event?","Not conflict (illegal migration); drop"),
(84,C,"Pakistani spy arrested by military intelligence in Gurdaspur. Class?","Interstate"),
(54,C,"Rusted grenades found while digging, destroyed by SFs. Event?","Not conflict: old ordnance"),
(26,C,"Ball-like object picked up on a railway track explodes at home. Class?","Not conflict: old ordnance (no actor, no district)"),
# --- location
(350,L,"ULFA cadre 'of Kukshi village under Fakiragram PS' arrested, no other place. Is Kukshi the arrest site?","Kukshi = origin, used as fallback site and flagged; PS and district are reliable"),
(42,L,"Encounter by Bishnupur police commandos 'in the Thoubal area'. Site district?","Thoubal (the unit's home district is not the site)"),
(276,L,"Encounter near Tingmun village 'on the Churachandpur-Tamenglong district border'. Site district?","Village decides if found; otherwise keep both districts, flag border"),
(411,L,"Tracks blown 'between Dania station (Bokaro) and Jageshwar station (Bihar)'. Which district?","Two acts: Ramgarh, and Bokaro (Dania side, since the text names Bokaro)"),
(480,L,"Cash seized from a TPC cadre 'on way to Balumath from Chatra via Panki in Palamu'. Site?","Palamu district (route; no village)"),
(423,L,"Encounter at 'Mirpora Beerwah', no district named; outside conf07. Site?","Mirpora village in Beerwah tehsil, Budgam district (via gazetteer)"),
(473,L,"Text puts Nowhatta and Makhdoom Sahib in Baramulla; both are in Srinagar. Which district?","Srinagar (gazetteer overrides the text's district)"),
(325,L,"Arrest at 'Junction of Tekang - Tizit River', no state; report from Nagaland Post. Site?","Mon district, Nagaland (Tizit river), newspaper supports it; flag low confidence"),
(363,L,"'In Jammu', arrests at Mokhey and Ramgarh. Jammu district or the Jammu division?","Division: resolve by the villages (Ramgarh is in Samba, formerly Jammu district)"),
(172,L,"Four shot dead 'at Langol in the Imphal West and Thoubal districts'. District?","Imphal West (Langol hills sit near Imphal; check the village)"),
(413,L,"Fourteen surrender at the district police HQ in Karimnagar. Site?","Karimnagar town (surrender site), origin villages not used"),
(260,L,"Delhi Police Special Cell busts a fake currency racket; no place except origin Bijnore. Site?","Delhi, inferred from the arresting unit; flagged"),
# --- conf07
(144,F:=None,"",""),
]
cases=[c for c in cases if c[1]]
cases+=[
(144,G,"Maoists blast a forest shed; outside conf07 (model confidence 0.53). Real event the gate misses?","Yes: violent event, attack on property, conflict"),
(147,G,"Suspected ULFA loot four rifles from forest camps; outside conf07 (looting is not an analysis action). Should looting count?","Yes: count as violent (looting, property destruction)"),
(44,G,"Maoists raid a fair price shop and take 300 bags of rice; outside conf07. Should it count?","Yes: count as violent; conflict"),
(114,G,"Forensic news on a 2010 stadium bomb; inside conf07. Event?","No: judicial or investigation; conf07 false positive"),
(78,G,"GOC statement recalling the Lolab killings; inside conf07. Event?","No: statement; conf07 false positive"),
(146,G,"Assam Rifles IG statement on IEDs smuggled into Nagaland; inside conf07. Event?","No: intelligence; conf07 false positive"),
]
wb=openpyxl.Workbook(); ws=wb.active; ws.title='check'
hdr=['check','case','group','event_id_cnty','event_date','series','in_conf07','question','my_coding','event_text','agree','your_coding','note']
ws.append(hdr)
for k,(n,g,q,m) in enumerate(cases,1):
    r=rows[n]
    ws.append(['',k,g,r['event_id_cnty'],r['event_date'],r['series_short'],'yes' if r['in_conf07']=='TRUE' else 'no',q,m,fix(r['event_text']),'','',''])
widths=dict(A=6,B=5,C=16,D=13,E=11,F=8,G=8,H=45,I=40,J=80,K=10,L=30,M=30)
for c,w in widths.items(): ws.column_dimensions[c].width=w
for row in ws.iter_rows(min_row=2):
    for c in row: c.alignment=Alignment(wrap_text=True,vertical='top')
for c in ws[1]: c.font=Font(bold=True); c.fill=PatternFill('solid',fgColor='DDDDDD')
fill=PatternFill('solid',fgColor='FFF2CC')
for i in range(2,ws.max_row+1):
    for col in 'AKLM': ws[f'{col}{i}'].fill=fill
ws.freeze_panes='D2'; ws.auto_filter.ref=ws.dimensions
dv=DataValidation(type='list',formula1='"yes,no,cant_tell"',allow_blank=True); ws.add_data_validation(dv); dv.add(f'K2:K{ws.max_row}')
dv2=DataValidation(type='list',formula1='"x"',allow_blank=True); ws.add_data_validation(dv2); dv2.add(f'A2:A{ws.max_row}')
r2=wb.create_sheet('how_to')
for line in [
 ["How to fill this sheet"],
 ["agree: yes if my coding is right, no if not, cant_tell if the text does not let you decide"],
 ["your_coding: when you answer no, write what it should be"],
 ["check: put x when done; note: anything else (a rule, a doubt)"],
 ["Groups: 1 event or not; 2 conflict, crime or unclear; 3 location; 4 conf07 disagreements"],
 ["Source: the 500-event read of 2026-11-24 (codebook_read_500.csv, seed 20261124)"],
 ["Your answers become decision rules and test cases for the extraction prompt"]]:
    r2.append(line)
r2.column_dimensions['A'].width=100
wb.save(base+'data/intermediate/validation/codebook_case_check.xlsx'); print(len(cases),'cases')
