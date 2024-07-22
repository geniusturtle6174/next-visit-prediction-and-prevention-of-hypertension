import argparse
import os
import time
import warnings

import numpy as np

import fea_util
import util

parser = argparse.ArgumentParser()
parser.add_argument('save_fea_dir')
parser.add_argument('--len_at_least', '-len', type=int, default=5)
parser.add_argument('--ignore_year_before', '-ign', type=int, default=2010)
parser.add_argument('--mul_by_bp', '-mbp', action='store_true')
parser.add_argument('--many_to_many', '-mtm', action='store_true')
args = parser.parse_args()

FEA_CONFIG = {
    'MAX_YEAR_DIFFERENCE': 3,
    'LEN_AT_LEAST': args.len_at_least,
    'MAX_NUM_FOR_ONE_CUSTOMER': 4,
    'IGNORE_YEAR_BEFORE': args.ignore_year_before,
    'MAIN_NAME': 'mjdata20211225',
    'MUL_BY_BP': args.mul_by_bp,
    'ALLOW_SHORTER': True,
}

if not os.path.exists(args.save_fea_dir):
    os.makedirs(args.save_fea_dir, 0o755)
    print('Fea will be saved in {}'.format(args.save_fea_dir))
else:
    warnings.warn('Dir {} already exist, fea files will be overwritten.'.format(args.save_fea_dir))
util.write_cfg(os.path.join(args.save_fea_dir, 'config.yml'), FEA_CONFIG)

print('Loading pkl...')
tic = time.time()
customer_to_records = util.load_pkl('data/' + FEA_CONFIG['MAIN_NAME'] + '.pkl')
toc = time.time()
print('Loading finished, time elapsed (s):', toc - tic)

header_to_row_idx = util.load_pkl('data/' + FEA_CONFIG['MAIN_NAME'] + '_header_to_row_idx.pkl')

tic = time.time()
fea_and_ans = {} # For training
for i, c in enumerate(customer_to_records):
    if i % 10000 == 0:
        print('(TR) Processing customer {}/{}...'.format(i, len(customer_to_records)))
        toc = time.time()
        print('\tTime elapsed (s):', toc - tic)
    fea, ans = fea_util.raw_dict_to_fea(
        customer_to_records[c], 'train', FEA_CONFIG, header_to_row_idx,
    )
    if fea is None:
        continue
    if not args.many_to_many:
        ans = ans[:, -1, :]
    key = fea.shape[1]
    if key not in fea_and_ans:
        fea_and_ans[key] = {
            'fea_all': [],
            'ans_all': [],
        }
    fea_and_ans[key]['fea_all'].append(fea)
    fea_and_ans[key]['ans_all'].append(ans)
    # if i == 10000:
    #     break

tic = time.time()
chid_to_fea = {}
for i, c in enumerate(customer_to_records):
    if i % 10000 == 0:
        print('(TE) Processing customer {}/{}...'.format(i, len(customer_to_records)))
        toc = time.time()
        print('\tTime elapsed (s):', toc - tic)
    # Test
    fea_te, ans_te, fea_next = fea_util.raw_dict_to_fea(
        customer_to_records[c], 'test', FEA_CONFIG, header_to_row_idx,
    )
    if fea_te is None or len(fea_te) == 0:
        continue
    chid_to_fea[c] = {
        'fea': fea_te,
        'ans': ans_te,
        'fea_next': fea_next,
    }
    # if i == 10000:
    #     break

del customer_to_records

for key in fea_and_ans:
    print('Concatenating TR data to one array for key {}...'.format(key))
    fea_all = np.concatenate(fea_and_ans[key]['fea_all'])
    ans_all = np.concatenate(fea_and_ans[key]['ans_all'])
    print('\tFinished, shape:', fea_all.shape, ans_all.shape)

    np.save(os.path.join(args.save_fea_dir, 'fea_all_{}.npy'.format(key)), fea_all)
    np.save(os.path.join(args.save_fea_dir, 'ans_all_{}.npy'.format(key)), ans_all)

print('TE data raw size:', len(chid_to_fea))
util.save_pkl(os.path.join(args.save_fea_dir, 'chid_to_fea.pkl'.format(key)), chid_to_fea)
