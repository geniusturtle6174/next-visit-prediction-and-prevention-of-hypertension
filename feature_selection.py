import argparse
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import util
from fea_names import FEA_NAMES_278 as FEA_NAMES
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('fea_dir')
parser.add_argument('--hyp_ths', type=int, nargs=2, default=[130, 80])
args = parser.parse_args()

fea_cfg = util.load_cfg(os.path.join(args.fea_dir, 'config.yml'))

fea_all = []
ans_all = []
max_len = 0
for key in range(1, fea_cfg['LEN_AT_LEAST']+1):
    if not os.path.exists(os.path.join(args.fea_dir, 'ans_all_{}.npy'.format(key))):
        continue
    print('Loading data for key', key)
    fea_one_len = np.load(os.path.join(args.fea_dir, 'fea_all_{}.npy'.format(key)))
    ans_one_len = np.load(os.path.join(args.fea_dir, 'ans_all_{}.npy'.format(key)))
    if ans_one_len.ndim >= 3:
        raise Exception('Do not use this script for many-to-many!')
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tRemoving ans-missing data...')
    ans_min = np.min(ans_one_len, axis=1)
    keep_idx = np.where(ans_min>0)[0]
    fea_one_len = fea_one_len[keep_idx]
    ans_one_len = ans_one_len[keep_idx]
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tRemoving fea-bp-missing data...')
    fea_bp_min = np.min(fea_one_len, axis=1)[:, -3:-1]
    fea_bp_min = np.min(fea_bp_min, axis=1)
    keep_idx = np.where(fea_bp_min>0)[0]
    fea_one_len = fea_one_len[keep_idx]
    ans_one_len = ans_one_len[keep_idx]
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tPatching values...')
    for n in range(fea_one_len.shape[0]):
        fea_one_len[n] = patch_for_one_person(fea_one_len[n])
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
        fea_all[idx] = np.pad(fea_all[idx], ((max_len-visit_num, 0), (0, 0)))

fea_all = np.array(fea_all)
ans_all = np.array(ans_all)
print('Overall shapes:', fea_all.shape, ans_all.shape)

print('Adding is_pad feature')
is_pad = np.max(fea_all, axis=-1, keepdims=True)
fea_all = np.concatenate((fea_all, is_pad), axis=-1)
fea_names_wtih_is_pad = FEA_NAMES + ['is_pad']
print('Finished, shape:', fea_all.shape, len(fea_names_wtih_is_pad))

sort_idx = np.lexsort((ans_all[:, 0]-args.hyp_ths[0], ans_all[:, 1]-args.hyp_ths[1]))
fea_all = fea_all[sort_idx]
ans_all = ans_all[sort_idx]
print('Sorted shapes:', sort_idx.shape, fea_all.shape, ans_all.shape)

fea_all = fea_all[::2]
ans_all = ans_all[::2]
print('Reduced shapes:', fea_all.shape, ans_all.shape)

raw_ans_all = np.copy(ans_all)
sbp_high_idx = np.where(raw_ans_all[:, 0] >= args.hyp_ths[0])[0]
dbp_high_idx = np.where(raw_ans_all[:, 1] >= args.hyp_ths[1])[0]
ans_all = np.zeros(ans_all.shape[0])
ans_all[sbp_high_idx] = 1
ans_all[dbp_high_idx] = 1
neg_ratio = np.mean(ans_all==0)
pos_ratio = np.mean(ans_all==1)
print('Use classification, ths: {}/{}, neg ratio: {:.4f}%, pos ratio: {:.4f}%'.format(
    args.hyp_ths[0],
    args.hyp_ths[1],
    100 * neg_ratio,
    100 * pos_ratio,
))

is_selected = np.zeros(fea_all.shape[-1])
selected_idx = []
best_rr_hist = []
n_fea_to_select = 20
best_rr = 0
for nth_round in range(n_fea_to_select):
    sel_idx_round = None
    for dim in range(fea_all.shape[-1]):
        if is_selected[dim] == 1:
            continue
        model = KNeighborsClassifier(n_neighbors=3, n_jobs=7)
        curr_idx = selected_idx + [dim]
        curr_fea = np.reshape(fea_all[:, :, curr_idx], (fea_all.shape[0], -1))
        model.fit(curr_fea, ans_all)
        y_hat = model.predict(curr_fea)
        recog_rate = np.mean(y_hat == ans_all)
        if recog_rate > best_rr:
            best_rr = recog_rate
            sel_idx_round = dim
            print('Round {}, dim {}, rr {:.4f}%, found new best'.format(nth_round, dim, recog_rate*100))
        else:
            print('Round {}, dim {}, rr {:.4f}%'.format(nth_round, dim, recog_rate*100))
    if sel_idx_round is not None:
        is_selected[sel_idx_round] = 1
        selected_idx.append(sel_idx_round)
        best_rr_hist.append(best_rr)
        print('Select idx {} in this round, all selected idx are {}'.format(
            sel_idx_round, selected_idx
        ))
        for si, rr in zip(selected_idx, best_rr_hist):
            print('{} {:.2f}%'.format(fea_names_wtih_is_pad[si], 100 * rr))
    else:
        print('Nothing selected in this round, stop. Selection history is:')
        for si, rr in zip(selected_idx, best_rr_hist):
            print('{} {:.2f}%'.format(fea_names_wtih_is_pad[si], 100 * rr))
        break
