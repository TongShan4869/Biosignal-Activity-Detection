
# Participant codes

participants = ['03FH', '3B8D', '3RFH', '4JF9', '93DK', '93JD', 'AP3H', 'F408', 'H39D', 'JD3K', 'K2Q2', 'KF93', 'KS03', 'LAS2', 'LDM5', 'LK27', 'ME93']

# Devices

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
		'ME93': [(0, 568500)]
	}
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