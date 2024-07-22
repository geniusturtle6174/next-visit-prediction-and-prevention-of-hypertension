import util

FEA_CONFIG = {
    'MAIN_NAME': 'mjdata20211225',
    'REC_START_IDX': 5,
}

header_to_row_idx = util.get_header_to_row_idx('data/' + FEA_CONFIG['MAIN_NAME'] + '.csv', FEA_CONFIG['REC_START_IDX'])
for h in header_to_row_idx:
    if h.startswith('f'):
        print(h)
