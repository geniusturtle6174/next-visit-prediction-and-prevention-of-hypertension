import numpy as np

import fea_names
from util import dt_to_float_ym, parse_float

ONE_HOT_MAPS = {
    'gender': {'1': 0, '2': 1},
    'bloodtype': {'': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
    'chkarea': {'2': 0, '3': 1, '4': 2, '7': 3},
}


def get_marriage_fea(dt_data, header_to_row_idx):
    if dt_data[header_to_row_idx['marriage_98']] != '':
        return [float(dt_data[header_to_row_idx['marriage_98']])]
    return [parse_float(dt_data[header_to_row_idx['marriage_14']], val_of_empty=0)]


def get_smoke_fea(dt_data, header_to_row_idx):
    smoke_or_not = parse_float(dt_data[header_to_row_idx['smokeornot_03']], val_of_empty=1)
    smoke_year = parse_float(dt_data[header_to_row_idx['smokeyear_09']], val_of_empty=0)
    nsmoke_year = parse_float(dt_data[header_to_row_idx['nsmokeyear_09']], val_of_empty=0)
    smokeamoun_97 = parse_float(dt_data[header_to_row_idx['smokeamoun_97']], val_of_empty=0)
    smokeamoun_03 = parse_float(dt_data[header_to_row_idx['smokeamoun_03']], val_of_empty=0)
    smokeamoun = 0
    if smokeamoun_97 > 0:
        smokeamoun = smokeamoun_97 - 1 if smokeamoun_97 >= 2 else smokeamoun_97
    elif smokeamoun_03 > 0:
        smokeamoun = smokeamoun_03
    return [smoke_or_not, smoke_year, nsmoke_year, smokeamoun]


def get_drink_fea(dt_data, header_to_row_idx):
    drink_or_not = parse_float(dt_data[header_to_row_idx['drinkornot_98']], val_of_empty=0)
    drink_habit = parse_float(dt_data[header_to_row_idx['drinkhabit_97']], val_of_empty=0)
    drink_year = parse_float(dt_data[header_to_row_idx['drinkyear']], val_of_empty=0)
    drink_kind1 = parse_float(dt_data[header_to_row_idx['drinkkind1']], val_of_empty=0)
    drink_kind2 = parse_float(dt_data[header_to_row_idx['drinkkind2']], val_of_empty=0)
    drink_kind3 = parse_float(dt_data[header_to_row_idx['drinkkind3']], val_of_empty=0)
    drink_kind4 = parse_float(dt_data[header_to_row_idx['drinkkind4']], val_of_empty=0)
    drink_kind_avg = (15 * drink_kind1 + 22.5 * drink_kind2 + 37.5 * drink_kind3 + 45 * drink_kind4) / 4
    cc_table = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [112.5, 169.5, 562.5, 900],
        [262.5, 395.5, 1312.5, 210],
        [525, 791, 2625, 4200],
    ]
    drink_cc = 0 if drink_or_not == 0 or drink_habit == 0 else cc_table[int(drink_or_not)-1][int(drink_habit)-1] * drink_kind_avg
    return [drink_or_not, drink_habit, drink_year, drink_kind1, drink_kind2, drink_kind3, drink_kind4, drink_cc]


def get_cocohabit_fea(dt_data, header_to_row_idx):
    cocohabit = parse_float(dt_data[header_to_row_idx['cocohabit_98']], val_of_empty=1)
    cocohabit_year = parse_float(dt_data[header_to_row_idx['cocohabityear']], val_of_empty=0)
    cocohabit_amoun = parse_float(dt_data[header_to_row_idx['cocohabitamoun']], val_of_empty=0)
    n_cocohabit_year = parse_float(dt_data[header_to_row_idx['ncocohabityear']], val_of_empty=0)
    return [cocohabit, cocohabit_year, cocohabit_amoun, n_cocohabit_year]


def get_nutri_fea(dt_data, header_to_row_idx):
    return [
        parse_float(dt_data[header_to_row_idx['nutrino']], val_of_empty=1),
        parse_float(dt_data[header_to_row_idx['nutri01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri02']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri03']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri04']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri05']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri06']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri07']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri10']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri11']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri12']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri13']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri14']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['nutri15']], val_of_empty=0),
    ]


def get_sport_fea(dt_data, header_to_row_idx):
    first_sport = parse_float(dt_data[header_to_row_idx['firstsport']], val_of_empty=0)
    first_sport_frequ = parse_float(dt_data[header_to_row_idx['firstsportfrequ']], val_of_empty=5)
    first_sport_time = parse_float(dt_data[header_to_row_idx['firstsporttime']], val_of_empty=0)
    first_sport_breath = parse_float(dt_data[header_to_row_idx['firstsportbreath']], val_of_empty=5)
    fs_table = [
        [5.5, 7.0, 7.0, 7.0],
        [1.5, 5.5, 7.0, 7.0],
        [0.0, 1.5, 3.5, 5.5],
        [0.0, 0.0, 1.5, 3.5],
        [0.0, 0.0, 0.0, 0.0],
    ]
    first_sport_strength = first_sport * fs_table[int(first_sport_frequ)-1][0 if first_sport_time == 0 else int(first_sport_time) - 1]
    return [first_sport, first_sport_frequ, first_sport_time, first_sport_breath, first_sport_strength]


def get_one_hot_fea(raw_val, oh_map):
    fea = [0] * len(oh_map)
    fea[oh_map[raw_val]] = 1
    return fea


def get_sbp(dt_data, header_to_row_idx):
    return max(list(map(lambda x: parse_float(x, val_of_empty=-1), [
        dt_data[header_to_row_idx['g_sll']],
        dt_data[header_to_row_idx['g_slr']],
        dt_data[header_to_row_idx['g_ssl']],
        dt_data[header_to_row_idx['g_ssr']],
    ])))


def get_dbp(dt_data, header_to_row_idx):
    return max(list(map(lambda x: parse_float(x, val_of_empty=-1), [
        dt_data[header_to_row_idx['g_dll']],
        dt_data[header_to_row_idx['g_dlr']],
        dt_data[header_to_row_idx['g_dsl']],
        dt_data[header_to_row_idx['g_dsr']],
    ])))


def one_dt_to_fea(dt_data, this_dt, base_dt, header_to_row_idx):
    height = parse_float(dt_data[header_to_row_idx['g_hei']], val_of_empty=-1)
    weight = parse_float(dt_data[header_to_row_idx['g_wei']], val_of_empty=-1)
    bmi = parse_float(dt_data[header_to_row_idx['g_bmi']], val_of_empty=-1)
    waist_circum = parse_float(dt_data[header_to_row_idx['g_wc']], val_of_empty=-1)
    hip_circum = parse_float(dt_data[header_to_row_idx['g_hc']], val_of_empty=-1)
    whr = parse_float(dt_data[header_to_row_idx['g_whr']], val_of_empty=-1)
    sbp = get_sbp(dt_data, header_to_row_idx)
    dbp = get_dbp(dt_data, header_to_row_idx)

    # get_one_hot_fea(dt_data[header_to_row_idx['chkarea']], ONE_HOT_MAPS['chkarea']) + \

    return \
    get_one_hot_fea(dt_data[header_to_row_idx['gender']], ONE_HOT_MAPS['gender']) + \
    [
        # parse_float(dt_data[header_to_row_idx['psick09']], val_of_empty=0),
        # parse_float(dt_data[header_to_row_idx['mdrug04']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick09']], val_of_empty=0),
    ] + \
    get_one_hot_fea(dt_data[header_to_row_idx['bloodtype']], ONE_HOT_MAPS['bloodtype']) + \
    get_smoke_fea(dt_data, header_to_row_idx) + \
    get_drink_fea(dt_data, header_to_row_idx) + \
    get_cocohabit_fea(dt_data, header_to_row_idx) + \
    get_nutri_fea(dt_data, header_to_row_idx) + \
    get_sport_fea(dt_data, header_to_row_idx) + \
    [
        parse_float(dt_data[header_to_row_idx['solvent']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['asbestos']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['radiation']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['vegetarian']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['workstreng']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['sleeptime_09']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['sameagehealth']], val_of_empty=5),

        parse_float(dt_data[header_to_row_idx['relate33b_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['allergydrug']], val_of_empty=2),
        parse_float(dt_data[header_to_row_idx['relate29a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate30a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate22a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate10a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate34b']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate35b']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate31b']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate23b']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['chestpain_noneex']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['chestpain_ex']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate23a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate11a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate12a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate15a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate16a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate17a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate24a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate20a']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mcstop']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mcstopage']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['aboutbonejoints']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opbrain_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opeye_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opent_03']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick14']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug08']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick27']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['oplung_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick22']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick13']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick11']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick11']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick12']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick12']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug03']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opheart_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug07']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick10']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug05']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick10']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick17']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick18']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opbubble_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick16']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['relate21b']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug13']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opstomach_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opodigest_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opcaecum_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick19']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opur_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick20']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opsubur_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opfemale_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick23']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick21']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opbone_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick13']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug06']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opchest_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['opt3_01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick01']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick05']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick05']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick04']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick04']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick02']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick02']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick06']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick06']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick07']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick25']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick03']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick03']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['psick08']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['rsick08']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug14']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug09']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug02']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug10']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug15']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug11']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug12']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['mdrug18']], val_of_empty=0),

        parse_float(dt_data[header_to_row_idx['lvh1']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh2']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh3']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh4']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh5']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh6']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh7']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh8']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh9']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh10']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh11']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['lvh12']], val_of_empty=0),

        parse_float(dt_data[header_to_row_idx['af1']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af2']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af3']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af4']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af5']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af6']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af7']], val_of_empty=0),
        parse_float(dt_data[header_to_row_idx['af8']], val_of_empty=0),

        parse_float(dt_data[header_to_row_idx['age']], val_of_empty=-1),
        height,
        weight,
        bmi,
        parse_float(dt_data[header_to_row_idx['g_fat']], val_of_empty=-1),
        waist_circum,
        hip_circum,
        whr,
        parse_float(dt_data[header_to_row_idx['g_pul']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['g_rr']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['g_cc']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['g_cci']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['g_ed']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['cbc_leu']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_ery']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_hemo']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_hema']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_mcv']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_mch']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_mchc']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_rdw']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cbc_pla']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['wbc_n']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['wbc_l']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['wbc_m']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['wbc_e']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['wbc_b']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['dm_fg']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['lf_tb']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_db']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_tp']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_alb']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_alb']], val_of_empty=0) / parse_float(dt_data[header_to_row_idx['lf_glo']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_glo']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_alp']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_got']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_gpt']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_ggt']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_ldh']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['lf_got']], val_of_empty=-1) / parse_float(dt_data[header_to_row_idx['lf_gpt']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['rf_bun']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['rf_cre']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['rf_egfr']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['ua_ua']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['l_tg']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['l_chol']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['l_hdlc']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['l_ldlc']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['l_ch']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['l_chol']], val_of_empty=-1) - parse_float(dt_data[header_to_row_idx['l_hdlc']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['cpi_ca']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cpi_p']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['cpi_fe']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['isd_hbagsv']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['isd_hbabsv']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['tu_afp']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['tu_cea']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['tf_ft4']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['tf_tsh']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['i_crp']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['i_raf']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['ur_leu']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_app']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_pro']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_glu']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_bil']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_urob']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_ob']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_ket']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_nit']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_sg']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ur_ph']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['as_liverus']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_idu']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_cbdu']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_gall']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_kid']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_hpv']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_pan']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['as_spl']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['x_che']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['x_kub']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['ekg_ekg']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['ent_ear']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ent_nose']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ent_thr']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ent_np']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ent_op']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['ent_neck']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['pf_fvc']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['pf_fev1']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['pf_mmf']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['au_l']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['au_r']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['vi_vanr']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['vi_vanl']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['vi_color']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['vi_stra']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['vi_ast']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['vi_tonl']], val_of_empty=-1),
        parse_float(dt_data[header_to_row_idx['vi_tonr']], val_of_empty=-1),

        parse_float(dt_data[header_to_row_idx['bmd_bmd']], val_of_empty=-1),

        # this_dt % 100,
        # base_dt % 100,
        # dt_to_float_ym(this_dt) - dt_to_float_ym(base_dt),
        int(sbp >= 130 or dbp >= 80),
        int( sbp - dbp >=  60),
        (sbp + 2 * dbp) / 3,
        sbp - dbp,
        sbp,
        dbp,
    ]


def get_mul_by_bp_fea(fea):
    if fea.ndim == 3: # (batch, time, fea)
        fea_sbp = fea[:, :, -3:-2] * fea
        fea_dbp = fea[:, :, -2:-1] * fea
    elif fea.ndim == 2: # (time, fea)
        fea_sbp = fea[:, -3:-2] * fea
        fea_dbp = fea[:, -2:-1] * fea
    else:
        raise ValueError('fea.ndim is {}, should be 3 or 2.'.format(fea.ndim))
    return np.concatenate((fea_sbp, fea_dbp, fea), axis=-1)


def get_time_delta_fea(fea):
    if fea.ndim == 3: # (batch, time, fea)
        fea_pad = np.pad(fea, ((0, 0), (1, 0), (0, 0)), mode='edge')
        fea_delta = fea_pad[:, 1:, :] - fea_pad[:, :-1, :]
    elif fea.ndim == 2: # (time, fea)
        fea_pad = np.pad(fea, ((1, 0), (0, 0)), mode='edge')
        fea_delta = fea_pad[1:, :] - fea_pad[:-1, :]
    else:
        raise ValueError('fea.ndim is {}, should be 3 or 2.'.format(fea.ndim))
    return np.concatenate((fea_delta, fea), axis=-1)


def one_dt_to_ans(dt_data, header_to_row_idx):
    return np.array([
        get_sbp(dt_data, header_to_row_idx),
        get_dbp(dt_data, header_to_row_idx)
    ])


def raw_dict_to_fea(raw_dict, mode, config, header_to_row_idx):
    dt_sorted = sorted(list(raw_dict.keys()))
    cut_idx = len(dt_sorted) - 1
    for i in range(len(dt_sorted)-1, 0, -1):
        t_curr = dt_to_float_ym(dt_sorted[i])
        t_prev = dt_to_float_ym(dt_sorted[i-1])
        if t_curr - t_prev > (config['MAX_YEAR_DIFFERENCE'] - 1e-6) or t_prev < config['IGNORE_YEAR_BEFORE']:
            break
        cut_idx -= 1
    dt_sorted = dt_sorted[cut_idx:]
    len_at_least = config['LEN_AT_LEAST']
    # Compose feature
    if mode == 'train':
        '''
        Return shape:
            fea_all: (batch, time, fea)
            ans_all: (batch, time, ans)
        '''
        dt_sorted = dt_sorted[:-1] # Last one is for test only, cannot be used at training stage
        if len(dt_sorted) <= (len_at_least + 1): # Shorter and equal
            if config['ALLOW_SHORTER'] and len(dt_sorted) >= 2: # At least 1 for input, 1 for target
                fea_one = []
                ans_one = []
                for dt, next_dt in zip(dt_sorted[:-1], dt_sorted[1:]):
                    fea = one_dt_to_fea(raw_dict[dt], dt, dt_sorted[-1], header_to_row_idx)
                    ans = one_dt_to_ans(raw_dict[next_dt], header_to_row_idx)
                    fea_one.append(fea)
                    ans_one.append(ans)
                fea_one = np.stack(fea_one).astype('float32')[np.newaxis, :, :]
                ans_one = np.stack(ans_one).astype('float32')[np.newaxis, :]
                if config['MUL_BY_BP']:
                    fea_one = get_mul_by_bp_fea(fea_one)
                return fea_one, ans_one
            return None, None
        fea_all = []
        ans_all = []
        for look_back in range(config['MAX_NUM_FOR_ONE_CUSTOMER']):
            if len(dt_sorted) - len_at_least - look_back - 1 < 0:
                break
            fea_one = []
            ans_one = []
            curr_dts = dt_sorted[-(len_at_least+look_back+1):-(look_back+1)]
            next_dts = dt_sorted[-(len_at_least+look_back):][:len_at_least] # Cannot use `-look_back` because it maybe zero
            for dt, next_dt in zip(curr_dts, next_dts):
                fea = one_dt_to_fea(raw_dict[dt], dt, dt_sorted[-(look_back+1)], header_to_row_idx)
                ans = one_dt_to_ans(raw_dict[next_dt], header_to_row_idx)
                fea_one.append(fea)
                ans_one.append(ans)
            fea_all.append(np.stack(fea_one).astype('float32'))
            ans_all.append(np.stack(ans_one).astype('float32'))
        fea_all = np.stack(fea_all).astype('float32')
        ans_all = np.stack(ans_all).astype('float32')
        if config['MUL_BY_BP']:
            fea_all = get_mul_by_bp_fea(fea_all)
        return fea_all, ans_all
    elif mode == 'test':
        '''
        Return shape:
            fea_all: (time, fea)
            ans_all: (ans)
            fea_next: (fea)
        '''
        if len(dt_sorted) <= 1:
            return None, None, None
        fea_all = []
        for dt in dt_sorted[-(len_at_least+1):-1]:
            fea = one_dt_to_fea(raw_dict[dt], dt, dt_sorted[-1], header_to_row_idx)
            fea_all.append(fea)
        fea_all = np.vstack(fea_all).astype('float32')
        ans_all = one_dt_to_ans(raw_dict[dt_sorted[-1]], header_to_row_idx).astype('float32')
        fea_next = one_dt_to_fea(raw_dict[dt_sorted[-1]], dt_sorted[-1], dt_sorted[-1], header_to_row_idx)
        if 'MUL_BY_BP' in config and config['MUL_BY_BP']:
            fea_all = get_mul_by_bp_fea(fea_all)
        return fea_all, ans_all, np.array(fea_next)
    else:
        raise Exception('Unsupported mode!')


def modify_by_slope(curr_x, curr_y, new_x, slope):
    new_y = curr_y
    if curr_y > 0:
        b = curr_y - slope * curr_x
        new_y = slope * new_x + b
    return new_y


def modify_fea(fea_mat, name_to_modify, to_ratio=None, by_val=None, n_year=1):
    '''
    Shape of `fea_mat`: (time, fea)
    '''
    if to_ratio is None and by_val is None:
        raise Exception('Error: both to_ratio and by_val are None!')

    if n_year <= 0:
        raise ValueError('n_year should be > 0')

    if name_to_modify == 'weight':
        slopes = {
            'waist_circum': {
                'male': 2.257514,
                'female': 1.9999647,
            },
            'hip_circum': {
                'male': 1.5024748,
                'female': 1.4860785,
            },
            'g_fat': {
                'male': 1.2805022,
                'female': 1.7864822,
            },
        }
        idx_weight = fea_names.FEA_NAMES_APAMI_TO_IDX[name_to_modify]
        idx_bmi = fea_names.FEA_NAMES_APAMI_TO_IDX['bmi']
        idx_waist_c = fea_names.FEA_NAMES_APAMI_TO_IDX['waist_circum']
        idx_hip_c = fea_names.FEA_NAMES_APAMI_TO_IDX['hip_circum']
        idx_whr = fea_names.FEA_NAMES_APAMI_TO_IDX['whr']
        idx_g_fat = fea_names.FEA_NAMES_APAMI_TO_IDX['g_fat']
        idx_age = fea_names.FEA_NAMES_APAMI_TO_IDX['age']

        last_fea = np.copy(fea_mat[-1, :])
        last_weight = last_fea[idx_weight]
        last_bmi = last_fea[idx_bmi]
        last_waist_c = last_fea[idx_waist_c]
        last_hip_c = last_fea[idx_hip_c]
        last_whr = last_fea[idx_whr]
        last_g_fat = last_fea[idx_g_fat]
        gender = 'm' if last_fea[fea_names.FEA_NAMES_APAMI_TO_IDX['gender_m']] == 1 else 'f'

        if to_ratio is not None and last_weight != -1 and last_bmi != -1 and last_bmi >= 24:
            to_ratio = 24 / last_bmi
            new_weight = last_weight * to_ratio
            new_bmi = last_bmi * to_ratio
            new_waist_c = modify_by_slope(last_bmi, last_waist_c, new_bmi, slopes['waist_circum']['male'] if gender == 'm' else slopes['waist_circum']['female'])
            new_hip_c = modify_by_slope(last_bmi, last_hip_c, new_bmi, slopes['hip_circum']['male'] if gender == 'm' else slopes['hip_circum']['female'])
            new_whr = new_waist_c / new_hip_c if new_waist_c > 0 and new_hip_c > 0 else last_whr
            new_g_fat = modify_by_slope(last_bmi, last_g_fat, new_bmi, slopes['g_fat']['male'] if gender == 'm' else slopes['g_fat']['female'])

            last_fea[idx_weight] = new_weight
            last_fea[idx_bmi] = new_bmi
            last_fea[idx_waist_c] = new_waist_c
            last_fea[idx_hip_c] = new_hip_c
            last_fea[idx_whr] = new_whr
            last_fea[idx_g_fat] = new_g_fat
            last_fea[idx_age] += 1

            stack_list = [fea_mat]
            for _ in range(n_year):
                stack_list.append(np.copy(last_fea))
                last_fea[idx_age] += 1

            fea_mat = np.vstack(stack_list)
            if fea_mat.shape[0] > 5:
                fea_mat = fea_mat[-5:]

        elif to_ratio is not None:
            pass
        else:
            raise NotImplementedError('By_val not implemented for weight.')

    else:
        raise Exception('Unknown feature name!')

    return np.copy(fea_mat)


def modify_fea_legacy(fea_mat, name_to_modify, to_ratio=None, by_val=None):
    '''
    Shape of `fea_mat`: (time, fea)
    '''
    if to_ratio is None and by_val is None:
        raise Exception('Error: both to_ratio and by_val are None!')

    if name_to_modify == 'weight':
        slopes = {
            'waist_circum': {
                'male': 2.2573533,
                'female': 2.0000217,
            },
            'hip_circum': {
                'male': 1.5024492,
                'female': 1.4861282,
            },
            'g_fat': {
                'male': 1.2803953,
                'female': 1.7864776,
            },
        }
        idx_weight = fea_names.FEA_NAMES_APAMI_TO_IDX[name_to_modify]
        idx_bmi = fea_names.FEA_NAMES_APAMI_TO_IDX['bmi']
        idx_waist_c = fea_names.FEA_NAMES_APAMI_TO_IDX['waist_circum']
        idx_hip_c = fea_names.FEA_NAMES_APAMI_TO_IDX['hip_circum']
        idx_whr = fea_names.FEA_NAMES_APAMI_TO_IDX['whr']
        idx_g_fat = fea_names.FEA_NAMES_APAMI_TO_IDX['g_fat']
        for dim in range(fea_mat.shape[0]):
            curr_weight = fea_mat[dim, idx_weight]
            curr_bmi = fea_mat[dim, idx_bmi]
            curr_waist_c = fea_mat[dim, idx_waist_c]
            curr_hip_c = fea_mat[dim, idx_hip_c]
            curr_whr = fea_mat[dim, idx_whr]
            curr_g_fat = fea_mat[dim, idx_g_fat]
            gender = 'm' if fea_mat[dim, fea_names.FEA_NAMES_APAMI_TO_IDX['gender_m']] == 1 else 'f'
            if curr_weight == -1 or curr_bmi == -1:
                return np.copy(fea_mat)
            if to_ratio is not None:
                if curr_bmi >= 24:
                    to_ratio = 24 / curr_bmi
                    new_weight = curr_weight * to_ratio
                    new_bmi = curr_bmi * to_ratio
                    new_waist_c = modify_by_slope(curr_bmi, curr_waist_c, new_bmi, slopes['waist_circum']['male'] if gender == 'm' else slopes['waist_circum']['female'])
                    new_hip_c = modify_by_slope(curr_bmi, curr_hip_c, new_bmi, slopes['hip_circum']['male'] if gender == 'm' else slopes['hip_circum']['female'])
                    new_whr = new_waist_c / new_hip_c if new_waist_c > 0 and new_hip_c > 0 else curr_whr
                    new_g_fat = modify_by_slope(curr_bmi, curr_g_fat, new_bmi, slopes['g_fat']['male'] if gender == 'm' else slopes['g_fat']['female'])
                else:
                    new_weight = curr_weight
                    new_bmi = curr_bmi
                    new_waist_c = curr_waist_c
                    new_hip_c = curr_hip_c
                    new_whr = curr_whr
                    new_g_fat = curr_g_fat
            else:
                raise NotImplementedError('By_val not implemented for weight.')
            fea_mat[dim, idx_weight] = new_weight
            fea_mat[dim, idx_bmi] = new_bmi
            fea_mat[dim, idx_waist_c] = new_waist_c
            fea_mat[dim, idx_whr] = new_whr
            fea_mat[dim, idx_g_fat] = new_g_fat
    else:
        raise Exception('Unknown feature name!')

    return np.copy(fea_mat)
