import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, ttest_rel
from sklearn.metrics import det_curve, matthews_corrcoef, precision_recall_curve

import fea_names
import util

np.set_printoptions(linewidth=150)
plt.rcParams["savefig.directory"] = '.'

parser = argparse.ArgumentParser()
parser.add_argument('result_dir_1')
parser.add_argument('result_dir_2')
parser.add_argument('--n_fold_train', '-n', type=int, default=5)
args = parser.parse_args()

HYP_THS = [130, 80]

print('Loading npy files...')
results_1 = []
results_2 = []
for i in range(args.n_fold_train):
    results_1.append(np.load(os.path.join(args.result_dir_1, 'result_{}.npy'.format(i))))
    results_2.append(np.load(os.path.join(args.result_dir_2, 'result_{}.npy'.format(i))))
print('2 * {} mats finished, shapes: {}, {}'.format(args.n_fold_train, results_1[0].shape, results_2[0].shape))

print('Loading idx_to_chid files...')
idx_to_chid_1 = util.load_pkl(os.path.join(args.result_dir_1, 'idx_to_chid.pkl'))
idx_to_chid_2 = util.load_pkl(os.path.join(args.result_dir_2, 'idx_to_chid.pkl'))
print('Lengths: idx_to_chid_1: {}, idx_to_chid_2: {}'.format(len(idx_to_chid_1), len(idx_to_chid_2)))

print('Loading chid_to_fea file...')
chid_to_fea_1 = util.load_pkl(os.path.join(args.result_dir_1, 'chid_to_fea.pkl'))
print('Lengths: chid_to_fea: {}'.format(len(chid_to_fea_1)))

print('Extracing essential data from chid_to_fea...')
idx_age = fea_names.FEA_NAMES_APAMI_TO_IDX['age']
idx_sbp = fea_names.FEA_NAMES_APAMI_TO_IDX['sbp']
idx_dbp = fea_names.FEA_NAMES_APAMI_TO_IDX['dbp']
chid_to_age = {}
chid_to_sbp = {}
chid_to_dbp = {}
chid_to_ans = {}
for chid in chid_to_fea_1:
    if chid_to_fea_1[chid]['use']:
        fea_mat = np.reshape(chid_to_fea_1[chid]['fea'], (5, -1))
        is_pad = fea_mat[:, -1]
        fea_mat = fea_mat[np.where(is_pad>0)[0], :-1]
        # chid_to_age[chid] = fea_mat[-1, idx_age]
        # chid_to_sbp[chid] = fea_mat[-1, idx_sbp]
        # chid_to_dbp[chid] = fea_mat[-1, idx_dbp]
        chid_to_ans[chid] = chid_to_fea_1[chid]['ans']

print('Determining thresholds...')
beta = 1
thresholds = {
    'Best F-1': [0.5 for _ in range(args.n_fold_train)],
    'Min FPR + FNR': [0.5 for _ in range(args.n_fold_train)],
    'Min FPR + 2 * FNR': [0.5 for _ in range(args.n_fold_train)],
    'Min FPR + 4 * FNR': [0.5 for _ in range(args.n_fold_train)],
}
det_va_all = []
va_best_f_all = []
for fold in range(args.n_fold_train):
    va_ans = np.load(os.path.join(args.result_dir_1, 'va_ans_{}.npy'.format(fold)))
    va_pred = np.load(os.path.join(args.result_dir_1, 'va_pred_{}.npy'.format(fold)))
    p, r, ths_pr = precision_recall_curve(va_ans, va_pred)
    fpr, fnr, ths_det = det_curve(va_ans, va_pred)
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-6)
    thresholds['Best F-1'][fold] = ths_pr[np.argmax(f)]
    thresholds['Min FPR + FNR'][fold] = ths_det[np.argmin(fpr+fnr)]
    thresholds['Min FPR + 2 * FNR'][fold] = ths_det[np.argmin(fpr+2*fnr)]
    thresholds['Min FPR + 4 * FNR'][fold] = ths_det[np.argmin(fpr+4*fnr)]
    det_va_all.append([fpr, fnr])
    va_best_f_all.append('{:.6f}'.format(np.max(f)))
    print('Dir 1 fold {} ths: best f: {:.6f} --> {:.6f}, Min FPR + FNR: {:.6f}'.format(
        fold, thresholds['Best F-1'][fold], np.max(f), thresholds['Min FPR + FNR'][fold]
    ))
print('Dir 1 all va F:', va_best_f_all)

chid_to_result = {}

# Fetch results: first fold
for row in results_1[0]:
    idx = row[0]
    chid = idx_to_chid_1[idx]
    if chid not in chid_to_result:
        chid_to_result[chid] = {
            'raw_recog_1': [row[3]],
        }
for row in results_2[0]:
    idx = row[0]
    chid = idx_to_chid_2[idx]
    chid_to_result[chid]['raw_recog_2'] = [row[3]]

# Fetch results: remaining folds
for result in results_1[1:]:
    for row in result:
        idx = row[0]
        chid = idx_to_chid_1[idx]
        chid_to_result[chid]['raw_recog_1'].append(row[3])
for result in results_2[1:]:
    for row in result:
        idx = row[0]
        chid = idx_to_chid_2[idx]
        chid_to_result[chid]['raw_recog_2'].append(row[3])

# Stack results and essential information
x = []
y = []
pred_1 = []
pred_2 = []
age_all = []
sbp_all = []
dbp_all = []
ans_all = []
for chid in chid_to_result:
    x.append(np.mean(chid_to_result[chid]['raw_recog_1']))
    y.append(np.mean(chid_to_result[chid]['raw_recog_2']))
    pred_1.append(np.median([
        chid_to_result[chid]['raw_recog_1'][f] >= thresholds['Best F-1'][f] for f in range(args.n_fold_train)
    ]))
    pred_2.append(np.median([
        chid_to_result[chid]['raw_recog_2'][f] >= thresholds['Best F-1'][f] for f in range(args.n_fold_train)
    ]))
    # age_all.append(chid_to_age[chid])
    # sbp_all.append(chid_to_sbp[chid])
    # dbp_all.append(chid_to_dbp[chid])
    ans_all.append(chid_to_ans[chid][0])
x = np.array(x)
y = np.array(y)
pred_1 = np.array(pred_1)
pred_2 = np.array(pred_2)
# age_all = np.array(age_all)
# sbp_all = np.array(sbp_all)
# dbp_all = np.array(dbp_all)
ans_all = np.array(ans_all)

print('matthews_corrcoef:', matthews_corrcoef(pred_1, pred_2), matthews_corrcoef(pred_2, pred_1))
print('T-test between two results:', ttest_rel(pred_1, pred_2))

# contingency_table = np.array([
#     [np.sum(pred_1 == 0), np.sum(pred_1 == 1)],
#     [np.sum(pred_2 == 0), np.sum(pred_2 == 1)],
# ])
# print('Chi2_contingency between two results:', chi2_contingency(contingency_table))

# Use pred pos only
# idx_pred_1_pos = np.where(pred_1 == 1)[0]
# x = x[idx_pred_1_pos]
# y = y[idx_pred_1_pos]
# pred_1 = pred_1[idx_pred_1_pos]
# pred_2 = pred_2[idx_pred_1_pos]
# age_all = age_all[idx_pred_1_pos]
# sbp_all = sbp_all[idx_pred_1_pos]
# dbp_all = dbp_all[idx_pred_1_pos]
# ans_all = ans_all[idx_pred_1_pos]

out_prob_diff = y - x
out_prob_ratio = y / (x + 1e-6)
idx_change = np.where(out_prob_diff != 0)[0]
idx_up = np.where(out_prob_diff > 0)[0]
idx_down = np.where(out_prob_diff < 0)[0]

# print('Num prob down:', np.sum(out_prob_diff < 0))
# print('Num prob up:', np.sum(out_prob_diff > 0))
# print('Mean of nonzero diff:', np.mean(out_prob_diff[idx_change]))
# print('Mean of nonone ratio:', np.mean(out_prob_ratio[idx_change]))
# print('Mean age of prob-up/down:', np.mean(age_all[idx_up]), np.mean(age_all[idx_down]))
# print('Mean sbp of prob-up/down:', np.mean(sbp_all[idx_up]), np.mean(sbp_all[idx_down]))
# print('Mean dbp of prob-up/down:', np.mean(dbp_all[idx_up]), np.mean(dbp_all[idx_down]))

# plt.figure()
# plt.subplot(1, 3, 1)
# plt.boxplot([age_all[idx_down], age_all[idx_up]])
# plt.subplot(1, 3, 2)
# plt.boxplot([sbp_all[idx_down], sbp_all[idx_up]])
# plt.subplot(1, 3, 3)
# plt.boxplot([dbp_all[idx_down], dbp_all[idx_up]])

# num_n2n = np.sum(np.logical_and(pred_1 == 0, pred_2 == 0))
# num_n2p = np.sum(np.logical_and(pred_1 == 0, pred_2 == 1))
num_p2n = np.sum(np.logical_and(pred_1 == 1, pred_2 == 0))
num_p2p = np.sum(np.logical_and(pred_1 == 1, pred_2 == 1))
# num_n = num_n2n + num_n2p
num_p = num_p2n + num_p2p
# print('N -> N: {} ({:.2f}%)'.format(num_n2n, 100 * num_n2n / (num_n+1e-6)) )
# print('N -> P: {} ({:.2f}%)'.format(num_n2p, 100 * num_n2p / (num_n+1e-6)) )
print('P -> N: {} ({:.2f}%)'.format(num_p2n, 100 * num_p2n / num_p) )
print('P -> P: {} ({:.2f}%)'.format(num_p2p, 100 * num_p2p / num_p) )

num_tp = np.sum(np.logical_and(ans_all == 1, pred_1 == 1))
num_fp = np.sum(np.logical_and(ans_all == 0, pred_1 == 1))
num_tpdiff = np.sum(np.logical_and(np.logical_and(ans_all == 1, pred_1 == 1), out_prob_diff != 0))
num_fpdiff = np.sum(np.logical_and(np.logical_and(ans_all == 0, pred_1 == 1), out_prob_diff != 0))
num_tp2n = np.sum(np.logical_and(np.logical_and(ans_all == 1, pred_1 == 1), pred_2 == 0))
num_fp2n = np.sum(np.logical_and(np.logical_and(ans_all == 0, pred_1 == 1), pred_2 == 0))
print('TP: {}'.format(num_tp))
print('FP: {}'.format(num_fp))
print('TP has diff: {}, ratio: {:.2f}%'.format(num_tpdiff, num_tpdiff/num_tp*100))
print('FP has diff: {}, ratio: {:.2f}%'.format(num_fpdiff, num_fpdiff/num_fp*100))
print('TP -> N: {}, ratio (of has diff): {:.2f}%'.format(num_tp2n, num_tp2n/num_tpdiff*100))
print('FP -> N: {}, ratio (of has diff): {:.2f}%'.format(num_fp2n, num_fp2n/num_fpdiff*100))
exit()
plt.figure()
n, bins, _ = plt.hist(out_prob_diff, bins=np.linspace(np.min(out_prob_diff), np.max(out_prob_diff), 51))
plt.yscale('log')
for c, b in zip(n, bins):
    print(b, c)
plt.xlabel('Diffeence of Output Probabilities')
plt.ylabel('Count')

# plt.figure()
# n, bins, _ = plt.hist(out_prob_ratio, bins=np.linspace(0, np.max(out_prob_ratio), 31))
# plt.yscale('log')
# for c, b in zip(n, bins):
#     print(b, c)
# plt.xlabel('Ratio of Output Probabilities')
# plt.ylabel('Count')

plt.show()
