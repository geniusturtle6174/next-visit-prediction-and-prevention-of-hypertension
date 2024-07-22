import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

parser = argparse.ArgumentParser()
parser.add_argument('fea_dir')
args = parser.parse_args()

fea_all = []
ans_all = []
last_all = []
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
    # print('\tPatching values...')
    # for n in range(fea_one_len.shape[0]):
    #     fea_one_len[n] = patch_for_one_person(fea_one_len[n])
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    for fea, ans in zip(fea_one_len, ans_one_len):
        fea_all.append(fea[-1, :])
        ans_all.append(ans)
        last_all.append(fea[-1, -3:-1])
        mh_all.append(int(np.max(fea[:, -3]) >= 130 or np.max(fea[:, -2]) >= 80))

fea_all = np.vstack(fea_all)
ans_all = np.vstack(ans_all)
last_all = np.vstack(last_all)
mh_all = np.array(mh_all)

hybp_ans_all = np.array([int(sbp >= 130 or dbp >= 80) for sbp, dbp in ans_all])
print('Hybp ratio: {:.4f}'.format(np.mean(hybp_ans_all)))

print('Overall shapes:', fea_all.shape, ans_all.shape, last_all.shape, mh_all.shape, hybp_ans_all.shape)

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.hist(ans_all[:, 0])
# plt.title('SBP')
# plt.subplot(2, 1, 2)
# plt.hist(ans_all[:, 1])
# plt.title('DBP')

# model = PCA(n_components=2)
# model.fit(fea_all)
# fea_reduced_all = model.transform(fea_all)
# idx_normal = np.where(hybp_ans_all == 0)[0]
# idx_hybp = np.where(hybp_ans_all == 1)[0]
# plt.figure()
# plt.plot(fea_reduced_all[idx_normal, 0], fea_reduced_all[idx_normal, 1], '.', label='Normal', markersize=1)
# plt.plot(fea_reduced_all[idx_hybp, 0], fea_reduced_all[idx_hybp, 1], '.', label='Hybp', markersize=1)
# plt.xlabel('Dim 0')
# plt.ylabel('Dim 1')
# plt.title('PCA')
# plt.legend()

model = LDA()
y = mh_all
model.fit(fea_all, y)
fea_reduced_all = model.transform(fea_all)
bins = np.linspace(fea_reduced_all.min(), fea_reduced_all.max(), 80)
idx_normal = np.where(y == 0)[0]
idx_hybp = np.where(y == 1)[0]
plt.figure()
plt.hist(fea_reduced_all[idx_normal], bins=bins, alpha=0.5, label='Normal')
plt.hist(fea_reduced_all[idx_hybp], bins=bins, alpha=0.5, label='Hybp')
plt.xlabel('Projected Value')
plt.ylabel('Count')
plt.title('LDA')
plt.legend()

# plt.figure()
# plt.hist(np.max(ans_all-last_all, axis=1), bins=30)

# model = LDA()
# # hybp_ans_2_all = np.array([int(ans[0] - last[0] >= 15) + int(ans[1] - last[1] >= 15) for last, ans in zip(last_all, ans_all)])
# hybp_ans_2_all = mh_all * 2 + hybp_ans_all
# model.fit(fea_all, hybp_ans_2_all)
# fea_reduced_all = model.transform(fea_all)
# plt.figure()
# for i in range(len(set(hybp_ans_2_all))):
#     idx = np.where(hybp_ans_2_all == i)[0]
#     plt.plot(fea_reduced_all[idx, 0], fea_reduced_all[idx, 1], '.', label='Class {}'.format(i), markersize=3)
# plt.xlabel('Dim 0')
# plt.ylabel('Dim 1')
# plt.title('LDA (2+ class)')
# plt.legend()

plt.show()
