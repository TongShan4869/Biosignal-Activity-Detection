


### Universal variables ###

sampling_rate = 500


### Participant codes ###

participants = ['03FH', '3B8D', '3RFH', '4JF9', '93DK', '93JD', 'AP3H', 'F408', 'H39D', 'JD3K', 'K2Q2', 'KF93', 'KS03', 'LAS2', 'LDM5', 'LK27', 'ME93']

def part_to_path(participant):
    return 'CSVs/' + participant + '.csv'


### Columns in the CSVs ###

columns = ['ecg:dry', 'ecg:gel', 'eda:dry', 'eda:gel', 'emg:Left Bicep', 'temp:temp', 'ppg:Left index finger', 'ppg:Left Wrist', 'acc_e4:z', 'acc_e4:x', 'acc_e4:y', 'acc_chest:x', 'acc_chest:y', 'acc_chest:z']


### Devices ###

devices = ['ScientISST Chest', 'ScientISST Forearm', 'Empatica E4']

col_to_device = {
    'ecg:dry': 'ScientISST Chest',
    'ecg:gel': 'ScientISST Chest',
    'eda:dry': 'Empatica E4',
    'eda:gel': 'ScientISST Forearm',
    'emg:Left Bicep': 'ScientISST Forearm',
    'temp:temp': 'Empatica E4',
    'ppg:Left index finger': 'ScientISST Forearm',
    'ppg:Left Wrist': 'Empatica E4',
    'acc_e4:z': 'Empatica E4',
    'acc_e4:x': 'Empatica E4',
    'acc_e4:y': 'Empatica E4',
    'acc_chest:x': 'ScientISST Chest',
    'acc_chest:y': 'ScientISST Chest',
    'acc_chest:z': 'ScientISST Chest',
}

device_to_cols = {
    'ScientISST Chest': ['ecg:dry', 'ecg:gel', 'acc_chest:x', 'acc_chest:y', 'acc_chest:z'],
    'ScientISST Forearm': ['eda:gel', 'emg:Left Bicep', 'ppg:Left index finger'],
    'Empatica E4': ['eda:dry', 'temp:temp', 'ppg:Left Wrist', 'acc_e4:z', 'acc_e4:x', 'acc_e4:y']
}


### Utility functions ###

def pair_to_range(pair):
    return range(pair[0], pair[1])


### Measurement ranges ###

measurement_ranges = {
	'ScientISST Chest': {
		'03FH': [(138582, 143182), (186941, 520441), (537079, 1029179)],
		'3B8D': [(16880, 49480), (300822, 331622), (347344, 373144), (391438, 445238), (470212, 622512), (725856, 969456), (994208, 1367808), (1378265, 1750365)],
		'3RFH': [(0, 92400), (523139, 587139), (651185, 674785), (702728, 824028), (1000538, 1034238), (1185172, 1243372), (1297514, 1505314), (1537492, 1758092), (1794660, 2203760), (2223992, 2383692)],
		'4JF9': [(0, 3136), (3142, 4182), (4203, 5227), (5248, 51000), (122318, 156418), (189820, 239820), (305604, 363804), (408504, 471104), (545048, 745648), (759622, 1004422), (1513396, 1676596), (1704623, 1989423)],
		'93DK': [(0, 75400), (155236, 182036), (212374, 239274), (261374, 287974), (323008, 429208), (482945, 602945), (627367, 743067), (755449, 770049), (784869, 853769), (870463, 1047363), (1067087, 1292787), (1307289, 1364089)],
		'93JD': [(0, 46500), (195391, 268191), (278949, 302649), (324463, 366963), (406117, 541717), (565575, 683575), (702193, 709569), (709585, 710625), (710641, 846393), (865591, 1021491), (1033624, 1258324), (1267980, 1387580), (1398305, 1549905), (1560549, 1663749)],
		'AP3H': [(2344, 131544), (337376, 371176), (397596, 425896), (452628, 494828), (595600, 642800), (659280, 968680), (1046396, 1477196), (1496582, 1716582)],
		'F408': [(870, 19770), (94624, 127724), (142083, 164783), (179667, 230167), (248428, 315828), (410141, 795741), (819663, 1178163)],
		'H39D': [(0, 24000), (39142, 69042), (277516, 306816), (320514, 514514), (523404, 785004), (968748, 1270448), (1310890, 1323790), (2429926, 2529026), (2557534, 2653534), (3244050, 3572150), (3596584, 3603384), (3634720, 3702220), (3749778, 3832378)],
		'JD3K': [(15910, 37710), (66084, 113684), (124632, 143232), (162192, 197292), (218314, 358014), (445016, 540616), (568565, 706465), (720627, 1284127), (1301826, 1487026)],
		'K2Q2': [(59458, 109458), (137224, 183824), (202030, 235830), (273510, 320210), (401153, 425453), (461579, 579079), (684687, 967587), (981862, 1427362), (1497172, 1617672), (1633189, 1637381), (1637385, 1638425), (1638449, 1729589)],
		'KF93': [(47637, 79157), (79162, 80202), (80224, 81248), (81269, 136437), (172177, 214677), (231223, 249123), (269329, 334729), (375344, 516644), (537976, 539000), (539019, 540059), (540067, 541107), (541128, 589976), (608658, 970458), (982634, 1164986), (1165005, 1166061), (1166067, 1167123), (1167146, 1313534), (1324216, 1594916)],
		'KS03': [(1031, 36231), (235992, 295792), (363483, 470083), (497169, 652469), (672155, 724955), (775305, 809805), (935675, 971175), (1013426, 1055926)],
		'LAS2': [(34957, 55157), (137307, 171507), (185717, 219217), (240065, 278865), (511793, 567293), (711917, 738917), (890003, 892003), (1031637, 1119037), (1146764, 1229064), (1249410, 1281010), (1318815, 1324215), (1472544, 1476444), (1497690, 1510990), (1574251, 1589251), (1615823, 1704623)],
		'LDM5': [(53556, 82256), (198428, 229628), (242506, 264106), (295778, 348678), (408736, 565236), (594131, 880031), (896857, 1407357), (1420391, 1660791), (1697167, 1780467)],
		'LK27': [(39388, 119688), (349132, 404132), (426045, 449545), (470395, 527595), (562745, 727045), (739347, 1024947), (1041200, 1395800), (1432354, 1605854), (1617184, 1758484), (1769934, 1893734)],
		'ME93': [(44713, 139313), (153324, 243624), (251658, 381058), (393340, 528540)]},
	'ScientISST Forearm': {
		'03FH': [(55472, 163972), (186231, 519731), (536375, 1027175)],
		'3B8D': [(14450, 46950), (298378, 329178), (344900, 370400), (388982, 443082), (467758, 619758), (638908, 703608), (723411, 1361211), (1375799, 1444299), (1503617, 1749217)],
		'3RFH': [(302663, 326363), (523304, 585904), (651346, 674946), (702874, 822374), (1001032, 1034432), (1185338, 1243538), (1297682, 1505382), (1794821, 2202521), (2224173, 2383873)],
		'4JF9': [(72426, 79226), (121462, 155362), (188973, 238773), (304787, 363587), (407649, 470049), (544231, 744931), (758791, 1003591), (1015554, 1085054), (1092858, 1273858), (1284448, 1504648), (1512544, 1671844), (1702390, 1988290)],
		'93DK': [(155243, 182043), (212381, 239181), (261365, 287965), (323011, 429211), (482951, 602951), (627385, 743085), (755465, 1047465), (1067101, 1372701)],
		'93JD': [(6640, 45740), (231451, 267351), (278305, 301905), (405472, 540872), (564943, 682943), (864387, 1020587), (1032866, 1257466), (1267212, 1386712), (1397550, 1549050), (1559788, 1663188)],
		'AP3H': [(0, 129000), (335042, 368142), (395238, 423538), (450283, 492383), (593256, 640456), (656930, 966230), (997341, 1474741), (1493785, 1713985)],
		'F408': [(0, 18800), (93761, 126661), (142105, 164205), (178783, 229283), (247548, 314948), (409264, 794664), (818777, 1176877)],
		'H39D': [(1300780, 1497580), (1807338, 2183838), (2429808, 2528808), (2557403, 2663803), (2825298, 2825598)],
		'JD3K': [(9348, 35848), (64368, 111868), (122904, 141404), (160474, 195474), (216599, 355799), (443309, 546409), (565195, 566095), (718709, 1282409), (1299303, 1485103)],
		'K2Q2': [(136721, 183221), (201513, 235213), (272999, 318099), (400658, 424858), (461057, 578557), (684161, 966861), (981363, 1426763), (1464473, 1617473), (1632648, 1729048)],
		'KF93': [(42678, 134978), (170740, 213340), (229794, 247594), (267904, 333304), (373924, 515124), (536569, 969069), (980765, 1311865), (1322785, 1593385)],
		'KS03': [(0, 34700), (235170, 293870), (362442, 468642), (496100, 651400), (671114, 723914), (934620, 978820), (1012384, 1054784)],
		'LAS2': [(34908, 55108), (142838, 171538), (185668, 220568), (240022, 279022), (511836, 567536), (711942, 753042), (849906, 958206), (1031594, 1119194), (1145235, 1229135), (1249355, 1268155), (1318767, 1451467), (1472479, 1704779)],
		'LDM5': [(51724, 80524), (196796, 227896), (240878, 262278), (294151, 346951), (407089, 563489), (592509, 877409), (895241, 1405241), (1418770, 1686370), (1694886, 1798386)],
		'LK27': [(39140, 119240), (348876, 403976), (425788, 449188), (470121, 527121), (562493, 727793), (739091, 1025191), (1393249, 1393649), (1615043, 1758143), (1769665, 1893465)],
		'ME93': []},
	'Empatica E4': {
		'03FH': [(0, 49500), (178000, 1189000)],
		'3B8D': [(0, 1905750)],
		'3RFH': [(414232, 870982), (984232, 2504482)],
		'4JF9': [(106277, 2047277)],
		'93DK': [(132438, 1556688)],
		'93JD': [],
		'AP3H': [(310690, 1728190)],
		'F408': [(82270, 1406020)],
		'H39D': [(1155790, 4694290)],
		'JD3K': [(0, 1527750)],
		'K2Q2': [(0, 2022750)],
		'KF93': [(0, 1638000)],
		'KS03': [(207211, 1232461)],
		'LAS2': [(0, 1289250), (1361500, 1727500)],
		'LDM5': [(0, 1847250)],
		'LK27': [(0, 1950000)],
		'ME93': [(0, 568500)]}
}


### Activities ###

activities = {
    '03FH': ['(blank)', 'baseline,', 'run,', 'walk_after,'],
    '3B8D': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    '3RFH': ['baseline,', '(blank)', 'lift,', 'greetings,', 'gesticulate,', 'jumps,', 'walk_before,', 'run,', 'walk_after,'],
    '4JF9': ['baseline,', '(blank)', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'run,walk_after,'],           # all 'walk_after' is also 'run'
    '93DK': ['baseline,', '(blank)', 'lift,', 'greetings,', 'jumps,', 'walk_before,', 'run,', 'walk_after,'],
    '93JD': ['baseline,', '(blank)', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    'AP3H': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before_downstairs,', 'run,', 'walk_after,'],
    'F408': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    'H39D': ['baseline,', '(blank)', 'lift,', 'walk_before,', 'run,', 'sprint,run,', 'walk_after,'],                              # all 'sprint' is also 'run'
    'JD3K': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    'K2Q2': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'walk_before_downstairs,', 'walk_before,', 'run,', 'walk_after,'],
    'KF93': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    'KS03': ['(blank)', 'baseline,', 'walk_before,', 'walk_before_downstairs,', 'gesticulate,', 'walk_before_elevatorup,', 'greetings,', 'lift,'],
    'LAS2': ['(blank)', 'lift-1,', 'lift-2,', 'greetings,', 'jumps,', 'gesticulate,', 'walk_before_elevatordown,', 'walk_before,', 'run,'],
    'LDM5': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    'LK27': ['(blank)', 'baseline,', 'lift,', 'greetings,', 'gesticulate,', 'walk_before,', 'run,', 'walk_after,'],
    'ME93': ['(blank)', 'run,']
}

activity_ranges = {
    '03FH': {
        '(blank)': [(0, 138582), (143182, 189205), (1087000, 1189468)],
        'baseline,': [(138582, 143182)],
        'run,': [(189205, 856030)],
        'walk_after,': [(856030, 1087000)]},
    '3B8D': {
        '(blank)': [(0, 16880), (49480, 303335), (331622, 350665), (373144, 393805), (445238, 470212), (622512, 643735), (1750364, 1906007)],
        'baseline,': [(16880, 49480)],
        'lift,': [(303335, 331622)],
        'greetings,': [(350665, 373144)],
        'gesticulate,': [(393805, 445238)],
        'walk_before,': [(470212, 622512)],
        'run,': [(643735, 1585775)],
        'walk_after,': [(1585775, 1750364)]},
    '3RFH': {
        'baseline,': [(0, 326363)],
        '(blank)': [(326363, 526536), (587139, 653202), (674785, 704082), (824028, 1002136), (1034238, 1141052), (1243371, 1297062), (2383690, 2504638)],
        'lift,': [(526536, 587139)],
        'greetings,': [(653202, 674785)],
        'gesticulate,': [(704082, 824028)],
        'jumps,': [(1002136, 1034238)],
        'walk_before,': [(1141052, 1243371)],
        'run,': [(1297062, 2278956)],
        'walk_after,': [(2278956, 2383690)]},
    '4JF9': {
        'baseline,': [(0, 51000)],
        '(blank)': [(51000, 123567), (156418, 206232), (239820, 309512), (363805, 411857), (471106, 561612)],
        'lift,': [(123567, 156418)],
        'greetings,': [(206232, 239820)],
        'gesticulate,': [(309512, 363805)],
        'walk_before,': [(411857, 471106)],
        'run,': [(561612, 1998777), (2047277, 2047347)],            # very short second period
        'run,walk_after,': [(1998777, 2047277)]},
    '93DK': {
        'baseline,': [(0, 75400)],
        '(blank)': [(75400, 158313), (182043, 212963), (239180, 266038), (287965, 396518), (429211, 491048), (1544438, 1556688)],
        'lift,': [(158313, 182043)],
        'greetings,': [(212963, 239180)],
        'jumps,': [(266038, 287965)],
        'walk_before,': [(396518, 429211)],
        'run,': [(491048, 1372700)],
        'walk_after,': [(1372700, 1544438)]},
    '93JD': {
        'baseline,': [(0, 46500)],
        '(blank)': [(46500, 195391), (268191, 278950), (302650, 324463), (366963, 406116), (541716, 565575), (1258324, 1267980), (1387580, 1663749)],
        'lift,': [(195391, 268191)],
        'greetings,': [(278950, 302650)],
        'gesticulate,': [(324463, 366963)],
        'walk_before,': [(406116, 541716)],
        'run,': [(565575, 1258324)],
        'walk_after,': [(1267980, 1387580)]},
    'AP3H': {
        '(blank)': [(0, 2344), (131544, 339760), (371176, 399300), (425896, 454144), (494828, 595600), (642800, 663050), (1716582, 1728471)],
        'baseline,': [(2344, 131544)],
        'lift,': [(339760, 371176)],
        'greetings,': [(399300, 425896)],
        'gesticulate,': [(454144, 494828)],
        'walk_before_downstairs,': [(595600, 642800)],
        'run,': [(663050, 1672450)],
        'walk_after,': [(1672450, 1716582)]},
    'F408': {
        '(blank)': [(0, 870), (19770, 95076), (127723, 144756), (164782, 179826), (230166, 250186), (315826, 408840), (1406270, 1406567)],
        'baseline,': [(870, 19770)],
        'lift,': [(95076, 127723)],
        'greetings,': [(144756, 164782)],
        'gesticulate,': [(179826, 230166)],
        'walk_before,': [(250186, 315826)],
        'run,': [(408840, 1174270)],
        'walk_after,': [(1174270, 1406270)]},
    'H39D': {
        'baseline,': [(0, 24000)],
        '(blank)': [(24000, 39142), (69042, 277516), (306816, 320396), (3876396, 3886594), (3931994, 4694665)],
        'lift,': [(39142, 69042)],
        'walk_before,': [(277516, 306816)],
        'run,': [(320396, 968748), (1323790, 3876396)],
        'sprint,run,': [(968748, 1323790)],
        'walk_after,': [(3886594, 3931994)]},
    'JD3K': {
        '(blank)': [(0, 15910), (37710, 64865), (113684, 127260), (143232, 163155), (197292, 224180), (358014, 446390), (1525000, 1528125)],
        'baseline,': [(15910, 37710)],
        'lift,': [(64865, 113684)],
        'greetings,': [(127260, 143232)],
        'gesticulate,': [(163155, 197292)],
        'walk_before,': [(224180, 358014)],
        'run,': [(446390, 1421930)],
        'walk_after,': [(1421930, 1525000)]},
    'K2Q2': {
        '(blank)': [(0, 136721), (183221, 201513), (235213, 272998), (318098, 400658), (424858, 461056), (578556, 684160), (966860, 981362), (1426762, 2022796)],
        'baseline,': [(136721, 183221)],
        'lift,': [(201513, 235213)],
        'greetings,': [(272998, 318098)],
        'walk_before_downstairs,': [(400658, 424858)],
        'walk_before,': [(461056, 578556)],
        'run,': [(684160, 966860)],
        'walk_after,': [(981362, 1426762)]},
    'KF93': {
        '(blank)': [(0, 47637), (136437, 173955), (214677, 233400), (249122, 271045), (334728, 401705), (516644, 546155), (1594915, 1638484)],
        'baseline,': [(47637, 136437)],
        'lift,': [(173955, 214677)],
        'greetings,': [(233400, 249122)],
        'gesticulate,': [(271045, 334728)],
        'walk_before,': [(401705, 516644)],
        'run,': [(546155, 1551540)],
        'walk_after,': [(1551540, 1594915)]},
    'KS03': {
        '(blank)': [(0, 1031), (36231, 240181), (295792, 366336), (470083, 512571), (652468, 674186), (724955, 938861), (971175, 1014671), (1055926, 1232703)],
        'baseline,': [(1031, 36231)],
        'walk_before,': [(240181, 295792)],
        'walk_before_downstairs,': [(366336, 470083)],
        'gesticulate,': [(512571, 652468)],
        'walk_before_elevatorup,': [(674186, 724955)],
        'greetings,': [(938861, 971175)],
        'lift,': [(1014671, 1055926)]},
    'LAS2': {
        '(blank)': [(0, 35210), (55108, 145045), (171538, 187375), (220568, 241150), (279022, 511915), (1704778, 1727679)],
        'lift-1,': [(35210, 55108)],
        'lift-2,': [(145045, 171538)],
        'greetings,': [(187375, 206595)],
        'jumps,': [(206595, 220568)],
        'gesticulate,': [(241150, 279022)],
        'walk_before_elevatordown,': [(511915, 567536)],
        'walk_before,': [(567536, 849500)],
        'run,': [(849500, 1704778)]},
    'LDM5': {
        '(blank)': [(0, 53556), (82256, 199320), (229628, 244305), (264106, 297330), (348678, 448150), (565236, 598455), (1847000, 1847570)],
        'baseline,': [(53556, 82256)],
        'lift,': [(199320, 229628)],
        'greetings,': [(244305, 264106)],
        'gesticulate,': [(297330, 348678)],
        'walk_before,': [(448150, 565236)],
        'run,': [(598455, 1779500)],
        'walk_after,': [(1779500, 1847000)]},
    'LK27': {
        '(blank)': [(0, 39140), (119240, 352810), (403976, 429025), (449188, 475560), (527121, 566440), (727792, 746000), (1758142, 1763540), (1893464, 1950093)],
        'baseline,': [(39140, 119240)],
        'lift,': [(352810, 403976)],
        'greetings,': [(429025, 449188)],
        'gesticulate,': [(475560, 527121)],
        'walk_before,': [(566440, 727792)],
        'run,': [(746000, 1758142)],
        'walk_after,': [(1763540, 1893464)]},
    'ME93': {
        '(blank)': [(0, 43725), (541000, 568593)],
        'run,': [(43725, 541000)]}
}







'''

### Code for generating measurement_ranges ###

measurement_ranges = dict()

for device in device_to_cols.keys():
    measurement_ranges[device] = dict()
    for participant in participants:
        measurement_ranges[device][participant] = []

df = pd.DataFrame()
current_participant = ''

def find_measurement_ranges(device, participant):
    col = device_to_cols[device][0]
    global df
    global current_participant
    global measurement_ranges
    if current_participant != participant:
        df = pd.read_csv('CSVs/' + participant + '.csv')
        current_participant = participant
    mrs = []
    current_start = None
    in_range = False
    for i in df.index:
        if in_range:
            if np.isnan(df[col][i]):
                mrs.append((current_start, i))
                in_range = False
        else:
            if not np.isnan(df[col][i]):
                current_start = i
                in_range = True
    if in_range:
        mrs.append((current_start, len(df.index)))
    measurement_ranges[device][participant] = mrs
    print(device, participant, mrs)

for participant in participants:
    for device in device_to_cols.keys():
        find_measurement_ranges(device, participant)
    df.drop(df.index, inplace=True)

'''
