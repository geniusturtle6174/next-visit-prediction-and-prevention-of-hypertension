import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from fea_names import FEA_NAMES_APAMI as FEA_NAMES
from fea_names import FEA_NAMES_APAMI_TO_IDX as FEA_NAMES_TO_IDX
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('fea_dir')
parser.add_argument('--no_patch', action='store_true')
args = parser.parse_args()

fea_all = []
ans_all = []
sbp_all = []
dbp_all = []
mh_all = []
for key in range(1, 6):
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
    if not args.no_patch:
        print('\tPatching values...')
        for n in range(fea_one_len.shape[0]):
            fea_one_len[n] = patch_for_one_person(fea_one_len[n])
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    for fea, ans in zip(fea_one_len, ans_one_len):
        fea_all.append(fea[-1])
        ans_all.append(ans[0] >= 130 or ans[1] >= 80)
        sbp_all.append(ans[0])
        dbp_all.append(ans[1])
        mh_all.append(np.any(fea[:, -2] >= 130) or np.any(fea[:, -1] >= 80))

fea_all = np.vstack(fea_all)
ans_all = np.hstack(ans_all).astype('float32')
sbp_all = np.hstack(sbp_all).astype('float32')
dbp_all = np.hstack(dbp_all).astype('float32')
mh_all = np.hstack(mh_all).astype('float32')
print('Overall shapes:', fea_all.shape, ans_all.shape, mh_all.shape)

NAMES_TO_OBSERVE = ['bmi']

# ===== Overall
idx_normal = np.where(ans_all == 0)[0]
idx_hybp = np.where(ans_all == 1)[0]
print('Idx shapes:', idx_normal.shape, idx_hybp.shape)
for name in NAMES_TO_OBSERVE:
    i = FEA_NAMES_TO_IDX[name]
    bins = np.linspace(fea_all[:, i].min(), fea_all[:, i].max(), 50)
    corr_ans = scipy.stats.pearsonr(fea_all[:, i], ans_all)[0]
    corr_sbp = scipy.stats.pearsonr(fea_all[:, i], sbp_all)[0]
    corr_dbp = scipy.stats.pearsonr(fea_all[:, i], dbp_all)[0]
    print('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(name, corr_ans, corr_sbp, corr_dbp))
    # if abs(corr) > 0.1:
    #     continue
    plt.figure()
    # plt.hist(fea_all[idx_normal, i], weights=np.ones(idx_normal.shape[0])/idx_normal.shape[0], bins=bins, alpha=0.5, label='Normal')
    # plt.hist(fea_all[idx_hybp, i], weights=np.ones(idx_hybp.shape[0])/idx_hybp.shape[0], bins=bins, alpha=0.5, label='Hybp')
    plt.boxplot([fea_all[idx_normal, i], fea_all[idx_hybp, i]], flierprops={'marker': '+'})
    plt.xticks(ticks=[1, 2], labels=['No NV Hybp', 'Has NV Hybp'])
    plt.ylabel(name)
    # if i == 0:
    #     plt.legend()
print('----------')
plt.show()
exit()
# ===== Has MH
idx_mh = np.where(mh_all == 1)[0]
idx_normal = np.where(np.logical_and(ans_all == 0, mh_all == 1))[0]
idx_hybp = np.where(np.logical_and(ans_all == 1, mh_all == 1))[0]
print('Idx shapes:', idx_normal.shape, idx_hybp.shape)

plt.figure()
for name in NAMES_TO_OBSERVE:
    i = FEA_NAMES_TO_IDX[name]
    bins = np.linspace(fea_all[idx_mh, i].min(), fea_all[idx_mh, i].max(), 50)
    corr_ans = scipy.stats.pearsonr(fea_all[idx_mh, i], ans_all[idx_mh])[0]
    corr_sbp = scipy.stats.pearsonr(fea_all[idx_mh, i], sbp_all[idx_mh])[0]
    corr_dbp = scipy.stats.pearsonr(fea_all[idx_mh, i], dbp_all[idx_mh])[0]
    print('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(name, corr_ans, corr_sbp, corr_dbp))
    plt.subplot(4, 5, i+1)
    # plt.figure()
    # plt.hist(fea_all[idx_normal, i], bins=bins, alpha=0.5, label='Normal')
    # plt.hist(fea_all[idx_hybp, i], bins=bins, alpha=0.5, label='Hybp')
    plt.boxplot([fea_all[idx_normal, i], fea_all[idx_hybp, i]], flierprops={'marker': '+'})
    plt.title(name)
    # if i == 0:
    #     plt.legend()
print('----------')

# ===== No MH
idx_mh = np.where(mh_all == 0)[0]
idx_normal = np.where(np.logical_and(ans_all == 0, mh_all == 0))[0]
idx_hybp = np.where(np.logical_and(ans_all == 1, mh_all == 0))[0]
print('Idx shapes:', idx_normal.shape, idx_hybp.shape)

plt.figure()
for name in NAMES_TO_OBSERVE:
    i = FEA_NAMES_TO_IDX[name]
    bins = np.linspace(fea_all[idx_mh, i].min(), fea_all[idx_mh, i].max(), 50)
    corr_ans = scipy.stats.pearsonr(fea_all[idx_mh, i], ans_all[idx_mh])[0]
    corr_sbp = scipy.stats.pearsonr(fea_all[idx_mh, i], sbp_all[idx_mh])[0]
    corr_dbp = scipy.stats.pearsonr(fea_all[idx_mh, i], dbp_all[idx_mh])[0]
    print('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(name, corr_ans, corr_sbp, corr_dbp))
    plt.subplot(4, 5, i+1)
    # plt.hist(fea_all[idx_normal, i], bins=bins, alpha=0.5, label='Normal')
    # plt.hist(fea_all[idx_hybp, i], bins=bins, alpha=0.5, label='Hybp')
    plt.boxplot([fea_all[idx_normal, i], fea_all[idx_hybp, i]], flierprops={'marker': '+'})
    plt.title(name)
    # if i == 0:
    #     plt.legend()

# plt.show()
