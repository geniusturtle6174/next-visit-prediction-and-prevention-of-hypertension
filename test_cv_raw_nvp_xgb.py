import argparse
import os
import time
import warnings
from shutil import copyfile

import numpy as np

import util
from fea_util import get_time_delta_fea, modify_fea
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('output_result_dir', help='Results directory name')
parser.add_argument('--n_fold_train', type=int, default=5)
parser.add_argument('--modify_fea', '-mf', type=int, default=0)
args = parser.parse_args()

tr_param = util.load_cfg(os.path.join(args.model_dir, 'config.yml'))

if not os.path.exists(args.output_result_dir):
    os.makedirs(args.output_result_dir, 0o755)
    print('Results will be saved in {}'.format(args.output_result_dir))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.output_result_dir))

print('Loading models and copying VA results...')
models = {}
for fold in range(args.n_fold_train):
    models[fold] = util.load_pkl(os.path.join(args.model_dir, 'model_{}.pkl'.format(fold)))
    copyfile(
        os.path.join(args.model_dir, 'va_pred_{}.npy'.format(fold)),
        os.path.join(args.output_result_dir, 'va_pred_{}.npy'.format(fold))
    )
    copyfile(
        os.path.join(args.model_dir, 'va_ans_{}.npy'.format(fold)),
        os.path.join(args.output_result_dir, 'va_ans_{}.npy'.format(fold))
    )

print('Loading chid_to_fea...')
tic = time.time()
chid_to_fea = util.load_pkl(os.path.join(tr_param['fea_dir'], 'chid_to_fea.pkl'))
toc = time.time()
print('Loading finished, time elapsed (s):', toc - tic)

chid_to_idx = {c: i for i, c in enumerate(chid_to_fea)}
idx_to_chid = {i: c for c, i in chid_to_idx.items()}
util.save_pkl(os.path.join(args.output_result_dir, 'idx_to_chid.pkl'), idx_to_chid)

DIM_SBP = -2
DIM_DBP = -1
use_data_cnt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'all': 0}
for i, c in enumerate(chid_to_fea):
    if i % 1000 == 0:
        print('Feature check for {}/{}'.format(i, len(chid_to_fea)))
    fea = chid_to_fea[c]['fea']
    ans = chid_to_fea[c]['ans']
    chid_to_fea[c]['use'] = True
    # Skip no ans
    if np.min(ans) <= 0:
        chid_to_fea[c]['use'] = False
        continue
    # Skip no bp
    fea_bp_min = np.min(fea, axis=0)
    if fea_bp_min[-2] < 0 or fea_bp_min[-1] < 0:
        chid_to_fea[c]['use'] = False
        continue
    use_data_cnt['all'] += 1
    vc = chid_to_fea[c]['fea'].shape[0]
    if vc not in use_data_cnt:
        use_data_cnt[vc] = 0
    use_data_cnt[vc] += 1
    # Record SBP/DBP
    chid_to_fea[c]['sbp_all'] = fea[:, DIM_SBP]
    chid_to_fea[c]['dbp_all'] = fea[:, DIM_DBP]
    # Personal patch
    if not tr_param['no_patch']:
        fea = patch_for_one_person(fea)
    # Get time delta
    if tr_param['delta']:
        fea = get_time_delta_fea(fea)
    # Modify value
    if args.modify_fea > 0:
        fea = modify_fea(fea, 'weight', to_ratio=0.9, n_year=args.modify_fea)
    # Padding and flatten
    if fea.shape[0] < tr_param['fea_cfg']['LEN_AT_LEAST']:
        # fea = np.pad(fea, ((0, tr_param['fea_cfg']['LEN_AT_LEAST']-fea.shape[0]), (0, 0))) # append
        fea = np.pad(fea, ((tr_param['fea_cfg']['LEN_AT_LEAST']-fea.shape[0], 0), (0, 0))) # preprend
    use_dims = tr_param.get('use_dims')
    if use_dims is not None:
        fea = fea[:, use_dims]
    # has_mh = 1 if np.max(fea[:, -2]) >= tr_param['hyp_ths'][0] or np.max(fea[:, -1]) >= tr_param['hyp_ths'][1] else 0
    is_pad = np.max(fea, axis=-1, keepdims=True)
    fea = np.concatenate((fea, is_pad), axis=-1)
    fea = fea.flatten()
    # fea = np.hstack((fea, has_mh))
    # Answer and pkl
    raw_ans = None
    if tr_param['run_classification_prob']:
        raw_ans = np.copy(ans)
        ans = [1 if ans[0] >= tr_param['hyp_ths'][0] or ans[1] >= tr_param['hyp_ths'][1] else 0]
    chid_to_fea[c]['fea'] = fea
    chid_to_fea[c]['ans'] = ans
    chid_to_fea[c]['raw_ans'] = raw_ans
    # if i == 5000:
    #     break

print('Test data num:', len(chid_to_fea))
print('Used data num:', use_data_cnt)

print('Writing chid_to_fea...')
util.save_pkl(os.path.join(args.output_result_dir, 'chid_to_fea.pkl'), chid_to_fea)
print('Finished.')

for fold in range(args.n_fold_train):
    idx_all = []
    fea_all = []
    ans_all = []
    recog_all = []
    print('Processing fold {}/{}'.format(fold, args.n_fold_train))
    for c in chid_to_fea:
        if 'use' not in chid_to_fea[c] or chid_to_fea[c]['use'] == False:
            continue
        idx_all.append(chid_to_idx[c])
        fea_all.append(chid_to_fea[c]['fea'])
        ans_all.append(chid_to_fea[c]['ans'])
    idx_all = np.vstack(idx_all)
    fea_all = np.vstack(fea_all)
    ans_all = np.vstack(ans_all)
    print('Data shpaes:', idx_all.shape, fea_all.shape, ans_all.shape)
    recog_all = models[fold].predict_proba(fea_all)
    result_all = np.hstack([idx_all, ans_all, recog_all])
    print('Fold {} finished, result shape: {}'.format(fold, result_all.shape))
    np.save(os.path.join(args.output_result_dir, 'result_{}.npy'.format(fold)), result_all)
    print('Results saved at:', args.output_result_dir)
