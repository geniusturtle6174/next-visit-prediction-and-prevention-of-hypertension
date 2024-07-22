import csv

import numpy as np

import util

MAIN_NAME = 'mjdata20211225'
PID_IDX = 0
YR_IDX = 3
MON_IDX = 1
REC_START_IDX = 5

EXT_NAME = 'mjdata_lvh_20220515'
EXT_PID_IDX = 0
EXT_YR_IDX = 4
EXT_MON_IDX = 2
EXT_CHKAREA_IDX = 1
EXT_REC_START_IDX = 5

print('Loading files...')
with open('data/' + MAIN_NAME + '.csv', 'r', encoding='utf-8') as fin:
    cnt = list(csv.reader(fin))
print('Finished.')

print('Loading extra files...')
with open('data/' + EXT_NAME + '.csv', 'r', encoding='utf-8') as fin:
    ext_cnt = list(csv.reader(fin))
print('Finished.')

customer_to_records = {}
headers = cnt[0][REC_START_IDX:]
for idx, row in enumerate(cnt[1:]):
    if idx % 100000 == 0:
        print('Processing row {}/{}'.format(idx, len(cnt[1:])))
    pid = row[PID_IDX]
    yr = int(row[YR_IDX])
    mon = int(row[MON_IDX])
    sub_key = int('{}{:02d}'.format(yr, mon))
    if pid not in customer_to_records:
        customer_to_records[pid] = {}
    if sub_key not in customer_to_records[pid]:
        customer_to_records[pid][sub_key] = row[REC_START_IDX:]
        assert len(headers) == len(customer_to_records[pid][sub_key])
    else:
        print('\tCustomer {} visit twice in a month!'.format(pid))

headers.append(ext_cnt[0][EXT_CHKAREA_IDX])
headers.extend(ext_cnt[0][EXT_REC_START_IDX:])
for idx, row in enumerate(ext_cnt[1:]):
    if idx % 100000 == 0:
        print('Processing extra row {}/{}'.format(idx, len(ext_cnt[1:])))
    pid = row[EXT_PID_IDX]
    yr = int(row[EXT_YR_IDX])
    mon = int(row[EXT_MON_IDX])
    sub_key = int('{}{:02d}'.format(yr, mon))
    if pid not in customer_to_records:
        print('NEW CUSTOMER: {}, NEED CHECK!'.format(pid))
        continue
    if sub_key not in customer_to_records[pid]:
        print('NEW VISIT: {}, NEED CHECK!'.format(sub_key))
        continue
    elif len(customer_to_records[pid][sub_key]) < len(headers):
        customer_to_records[pid][sub_key].append(row[EXT_CHKAREA_IDX])
        customer_to_records[pid][sub_key].extend(row[EXT_REC_START_IDX:])
        assert len(headers) == len(customer_to_records[pid][sub_key])
    elif len(customer_to_records[pid][sub_key]) >= len(headers):
        print('\tCustomer {} visit twice in a month!'.format(pid))

header_to_row_idx = {h: i for i, h in enumerate(headers)}

print('Writing...')
util.save_pkl('data/' + MAIN_NAME + '.pkl', customer_to_records)
util.save_pkl('data/' + MAIN_NAME + '_header_to_row_idx.pkl', header_to_row_idx)
