import os
import argparse

import numpy as np
import shap

import util
from fea_util import get_time_delta_fea
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('--n_fold_train', type=int, default=5)
args = parser.parse_args()

tr_param = util.load_cfg(os.path.join(args.model_dir, 'config.yml'))

print('Loading models...')
models = {}
for fold in range(args.n_fold_train):
    models[fold] = util.load_pkl(os.path.join(args.model_dir, 'model_{}.pkl'.format(fold)))

fea_cfg = tr_param['fea_cfg']
fea_dir = tr_param['fea_dir']

fea_all = []
ans_all = []
max_len = 0
for key in range(1, fea_cfg['LEN_AT_LEAST']+1):
    if not os.path.exists(os.path.join(fea_dir, 'ans_all_{}.npy'.format(key))):
        continue
    print('Loading data for key', key)
    fea_one_len = np.load(os.path.join(fea_dir, 'fea_all_{}.npy'.format(key)))
    ans_one_len = np.load(os.path.join(fea_dir, 'ans_all_{}.npy'.format(key)))
    if ans_one_len.ndim >= 3:
        raise Exception('Do not use this script for many-to-many!')
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tRemoving ans-missing data...')
    ans_min = np.min(ans_one_len, axis=1)
    keep_idx = np.where(ans_min > 0)[0]
    fea_one_len = fea_one_len[keep_idx]
    ans_one_len = ans_one_len[keep_idx]
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tRemoving fea-bp-missing data...')
    fea_bp_min = np.min(fea_one_len, axis=1)[:, -2:]
    fea_bp_min = np.min(fea_bp_min, axis=1)
    keep_idx = np.where(fea_bp_min>0)[0]
    fea_one_len = fea_one_len[keep_idx]
    ans_one_len = ans_one_len[keep_idx]
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    if not tr_param['no_patch']:
        print('\tPatching values...')
        for n in range(fea_one_len.shape[0]):
            fea_one_len[n] = patch_for_one_person(fea_one_len[n])
    if tr_param['delta']:
        print('\tGetting time delta...')
        fea_one_len_new = np.concatenate((fea_one_len, fea_one_len), axis=-1)
        for n in range(fea_one_len.shape[0]):
            fea_one_len_new[n] = get_time_delta_fea(fea_one_len[n])
        fea_one_len = np.copy(fea_one_len_new)
        print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tComputing pos ratio...')
    is_hybp = [sbp >= tr_param['hyp_ths'][0] or dbp >= tr_param['hyp_ths'][1] for sbp, dbp in zip(ans_one_len[:, 0], ans_one_len[:, 1])]
    print('\tFinished, result: {:.4f}%'.format(np.mean(is_hybp) * 100))
    print('\tnp array to list...')
    for fea, ans in zip(fea_one_len, ans_one_len):
        fea_all.append(fea)
        ans_all.append(ans)
    if key > max_len:
        max_len = key

del fea_one_len, ans_one_len

print('Padding all mats to length', max_len)
for idx in range(len(fea_all)):
    visit_num = fea_all[idx].shape[0]
    if visit_num < max_len:
        # fea_all[idx] = np.pad(fea_all[idx], ((0, max_len-visit_num), (0, 0)))
        fea_all[idx] = np.pad(fea_all[idx], ((max_len-visit_num, 0), (0, 0)))

fea_all = np.array(fea_all)
ans_all = np.array(ans_all)
print('Overall shapes:', fea_all.shape, ans_all.shape)

print('Computing has_mh Info')
has_mh = np.zeros((fea_all.shape[0], 1))
sbp_high_idx = np.where(np.max(fea_all[:, :, -2], axis=1) >= tr_param['hyp_ths'][0])[0]
dbp_high_idx = np.where(np.max(fea_all[:, :, -1], axis=1) >= tr_param['hyp_ths'][1])[0]
has_mh[sbp_high_idx] = 1
has_mh[dbp_high_idx] = 1
print('Finished, mh ratio: {:.4f}%'.format(100*np.mean(has_mh)))

if tr_param['use_mh_type'] in (0, 1):
    keep_idx = np.where(has_mh == tr_param['use_mh_type'])[0]
    fea_all = fea_all[keep_idx]
    ans_all = ans_all[keep_idx]
    print('Use only mh type', tr_param['use_mh_type'], 'for training, shape:', fea_all.shape)

if tr_param['use_dims'] is not None:
    print('Use only dims:', tr_param['use_dims'])
    fea_all = fea_all[:, :, tr_param['use_dims']]
    use_dims = deepcopy(tr_param['use_dims'])
    print('New shape:', fea_all.shape)

print('Adding is_pad feature and reshape')
is_pad = np.max(fea_all, axis=-1, keepdims=True)
fea_all = np.concatenate((fea_all, is_pad), axis=-1)
fea_all = np.reshape(fea_all, (fea_all.shape[0], -1))
print('Finished, shape:', fea_all.shape)

for fold in range(args.n_fold_train):
    print('Computing SHAP values for fold {}/{}'.format(fold, args.n_fold_train))
    explainer = shap.Explainer(models[fold], fea_all)
    shap_values = explainer(fea_all)
    util.save_pkl(os.path.join(args.model_dir, 'shap_{}.pkl'.format(fold)), shap_values)
