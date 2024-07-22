import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, ttest_rel
from sklearn.metrics import det_curve, precision_recall_curve, roc_auc_score, roc_curve

import result_util
import util
from fea_names import FEA_NAMES_APAMI as FEA_NAMES

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('result_dir')
parser.add_argument('--n_fold_train', '-n', type=int, default=5)
parser.add_argument('--output_detail_file', '-d')
parser.add_argument('--highlight_ths', '-ht', type=int, nargs=2, default=[140, 90])
parser.add_argument('--no_plot', action='store_true')
args = parser.parse_args()

HYP_THS = [130, 80]
plt.rcParams["savefig.directory"] = '.'

print('Loading npy files...')
results = []
for i in range(args.n_fold_train):
    results.append(np.load(os.path.join(args.result_dir, 'result_{}.npy'.format(i))))
print('{} mats finished, shape: {}'.format(args.n_fold_train, results[0].shape))

print('Loading pkl files...')
idx_to_chid = util.load_pkl(os.path.join(args.result_dir, 'idx_to_chid.pkl'))
chid_to_fea = util.load_pkl(os.path.join(args.result_dir, 'chid_to_fea.pkl'))
print('Lengths: idx_to_chid: {}, chid_to_fea: {}'.format(len(idx_to_chid), len(chid_to_fea)))

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
    va_ans = np.load(os.path.join(args.result_dir, 'va_ans_{}.npy'.format(fold)))
    va_pred = np.load(os.path.join(args.result_dir, 'va_pred_{}.npy'.format(fold)))
    p, r, ths_pr = precision_recall_curve(va_ans, va_pred)
    fpr, fnr, ths_det = det_curve(va_ans, va_pred)
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-6)
    thresholds['Best F-1'][fold] = ths_pr[np.argmax(f)]
    thresholds['Min FPR + FNR'][fold] = ths_det[np.argmin(fpr+fnr)]
    thresholds['Min FPR + 2 * FNR'][fold] = ths_det[np.argmin(fpr+2*fnr)]
    thresholds['Min FPR + 4 * FNR'][fold] = ths_det[np.argmin(fpr+4*fnr)]
    det_va_all.append([fpr, fnr])
    va_best_f_all.append('{:.6f}'.format(np.max(f)))
    print('Fold {} ths: best f: {:.6f} --> {:.6f}, Min FPR + FNR: {:.6f}'.format(
        fold, thresholds['Best F-1'][fold], np.max(f), thresholds['Min FPR + FNR'][fold]
    ))
print('All va F:', va_best_f_all)

idx_to_result = {}

# First fold
f_idx = 0
for row in results[0]:
    idx = row[0]
    if idx not in idx_to_result:
        idx_to_result[idx] = {
            'ans': row[1],
            'recog': {key: [row[3] >= thresholds[key][f_idx]] for key in thresholds},
            'raw_recog': [row[3]],
        }

# Remaining folds
for f_idx, result in enumerate(results[1:], 1):
    for row in result:
        idx = row[0]
        ans = row[1]
        assert np.sum(np.abs(ans - idx_to_result[idx]['ans'])) < 1e-6
        for key in thresholds:
            idx_to_result[idx]['recog'][key].append(row[3] >= thresholds[key][f_idx])
        idx_to_result[idx]['raw_recog'].append(row[3])

print('Calc AUROC for each fold...')
raw_recog = np.array([item['raw_recog'] for _, item in idx_to_result.items()])
recog = np.array([item['recog']['Best F-1'] for _, item in idx_to_result.items()])
ans = np.array([item['ans'] for _, item in idx_to_result.items()])
det_te_all = []
plt.figure()
for fold in range(raw_recog.shape[1]):
    fpr_det, fnr_det, ths_det = det_curve(ans, raw_recog[:, fold])
    fpr_roc, tpr_roc, _ = roc_curve(ans, raw_recog[:, fold])
    det_te_all.append([fpr_det, fnr_det])
    plt.plot(fpr_roc, tpr_roc, label='Fold {}'.format(fold+1))
    # result_util.print_results('Recog (fold {})'.format(fold), recog[:, fold].astype('int'), ans.astype('int'))
    # print(fold, '{:.6f}'.format(roc_auc_score(ans, raw_recog[:, fold])))
plt.legend()
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')

hybp_last_all = []
hybp_recog_all = []
hybp_recog_w1_all = []
hybp_recog_w2_all = []
hybp_recog_w4_all = []
mh_all = []
sex_all = []
sick_drug_r_all = []
hybp_ans_all = []
hybp_raw_ans_all = []
visit_count_all = []
visit_interval_all = []
counter = {
    'all': 0,
    'normal': 0,
    'only sbp high': 0,
    'only dbp high': 0,
    'both high': 0,
    'highlight': {
        'only sbp high': 0,
        'only dbp high': 0,
        'both': 0,
        'has_mh': 0,
        'no_mh': 0,
        'no_mh_len_count': [0, 0, 0, 0, 0, ],
    }
}
chid_to_error_type = {}

data_to_plot = []
if args.output_detail_file is not None:
    fout = open(args.output_detail_file, 'w', encoding='utf-8')
    fout.write(','.join(FEA_NAMES)+'\n')

for idx in idx_to_result:
    sick_drug_r = False
    fea_mat = np.reshape(chid_to_fea[idx_to_chid[idx]]['fea'], (5, -1))
    is_pad = fea_mat[:, -1]
    fea_mat = fea_mat[np.where(is_pad>0)[0], :-1]
    # for fea in fea_mat:
    #     if fea[3] == 1:
    #         sick_drug_r = True
    recog = np.median(np.vstack(idx_to_result[idx]['recog']['Best F-1'])).astype('int')
    ans = idx_to_result[idx]['ans']
    assert ans == chid_to_fea[idx_to_chid[idx]]['ans'][0]
    raw_ans = chid_to_fea[idx_to_chid[idx]]['raw_ans']
    last_sbp = chid_to_fea[idx_to_chid[idx]]['sbp_all'][-1]
    last_dbp = chid_to_fea[idx_to_chid[idx]]['dbp_all'][-1]
    has_mh = any([
        sbp >= HYP_THS[0] or dbp >= HYP_THS[1] \
            for sbp, dbp in zip(chid_to_fea[idx_to_chid[idx]]['sbp_all'], chid_to_fea[idx_to_chid[idx]]['dbp_all'])
    ])
    fea_next = chid_to_fea[idx_to_chid[idx]]['fea_next']
    counter['all'] += 1
    if ans == 0:
        counter['normal'] += 1
    elif ans == 1:
        data_to_plot.append(raw_ans)
        if raw_ans[0] >= 130 and raw_ans[1] < 80:
            counter['only sbp high'] += 1
        elif raw_ans[0] < 130 and raw_ans[1] >= 80:
            counter['only dbp high'] += 1
        else:
            assert raw_ans[0] >= 130 and raw_ans[1] >= 80
            counter['both high'] += 1
        sbp_hl_ths = args.highlight_ths[0]
        dbp_hl_ths = args.highlight_ths[1]
        if args.output_detail_file and recog == 0 and (raw_ans[0] >= sbp_hl_ths or raw_ans[1] >= dbp_hl_ths) and not has_mh:
            # if has_mh:
            #     counter['highlight']['has_mh'] += 1
            # else:
            #     fea_len = len(fea_mat)
            #     counter['highlight']['no_mh'] += 1
            #     counter['highlight']['no_mh_len_count'][fea_len-1] += 1
            # if raw_ans[0] >= sbp_hl_ths and raw_ans[1] < dbp_hl_ths:
            #     counter['highlight']['only sbp high'] += 1
            # elif raw_ans[0] < sbp_hl_ths and raw_ans[1] >= dbp_hl_ths:
            #     counter['highlight']['only dbp high'] += 1
            # else:
            #     assert raw_ans[0] >= sbp_hl_ths and raw_ans[1] >= dbp_hl_ths
            #     counter['highlight']['both'] += 1
            fout.write('idx: {},true id: {},mh: {},ans: {},next sbp: {}, next dbp: {},{}\n'.format(
                chid, idx_to_chid[idx], has_mh, ans, raw_ans[0], raw_ans[1], ',' * (len(FEA_NAMES)-5),
            ))
            for fea in fea_mat:
                fout.write(','.join(['{:.2f}'.format(f) for f in fea])+'\n')
            fout.write(','.join(['{:.2f}'.format(f) for f in fea_next])+'\n')
    hybp_last_all.append(int(last_sbp >= HYP_THS[0] or last_dbp >= HYP_THS[1]))
    hybp_recog_all.append(recog)
    hybp_recog_w1_all.append(np.median(np.vstack(idx_to_result[idx]['recog']['Min FPR + FNR'])).astype('int'))
    hybp_recog_w2_all.append(np.median(np.vstack(idx_to_result[idx]['recog']['Min FPR + 2 * FNR'])).astype('int'))
    hybp_recog_w4_all.append(np.median(np.vstack(idx_to_result[idx]['recog']['Min FPR + 4 * FNR'])).astype('int'))
    hybp_ans_all.append(ans)
    hybp_raw_ans_all.append(raw_ans)
    mh_all.append(has_mh)
    sex_all.append(fea_mat[-1, 0])
    sick_drug_r_all.append(sick_drug_r)
    visit_count_all.append(len(fea_mat))
    visit_interval_all.append(fea_mat[-1, -1])
    chid_to_error_type[idx_to_chid[idx]] = result_util.get_error_type(recog, ans)

if args.output_detail_file is not None:
    fout.close()

util.save_pkl(os.path.join(args.result_dir, 'chid_to_error_type.pkl'), chid_to_error_type)

# for key, val in counter.items():
#     if key == 'highlight':
#         for k, v in val.items():
#             print('  ', k, v)
#     else:
#         print(key, val)

# data_to_plot = np.array(data_to_plot)
# plt.figure()
# plt.plot(data_to_plot[:, 0], data_to_plot[:, 1], '.')
# plt.xlabel('SBP')
# plt.ylabel('DBP')
# plt.show()

hybp_last_all = np.array(hybp_last_all).astype('int')
hybp_recog_all = np.array(hybp_recog_all)
hybp_ans_all = np.array(hybp_ans_all).astype('int')
hybp_raw_ans_all = np.array(hybp_raw_ans_all)
mh_all = np.array(mh_all).astype('int')
sex_all = np.array(sex_all)
sick_drug_r_all = np.array(sick_drug_r_all)
visit_count_all = np.array(visit_count_all).astype('int')
visit_interval_all = np.array(visit_interval_all)

print('HYBP ratio:', np.mean(hybp_ans_all))
print('PPD >= 50 ratio:', np.mean(hybp_raw_ans_all[:,0]-hybp_raw_ans_all[:,1] >= 50))

print('Shapes:', hybp_last_all.shape, hybp_recog_all.shape, hybp_ans_all.shape)

result_util.print_results('Naive LAST', hybp_last_all, hybp_ans_all)
# result_util.print_results('Naive ALL', mh_all, hybp_ans_all)
m_ov, p_overall, r_overall, f_overall, a_overall = result_util.print_results('Recog (overall)', hybp_recog_all, hybp_ans_all)
# m_w1, _, _, _, _ = result_util.print_results('Recog (w1)', hybp_recog_w1_all, hybp_ans_all)
# m_w2, _, _, _, _ = result_util.print_results('Recog (w2)', hybp_recog_w2_all, hybp_ans_all)
# m_w4, _, _, _, _ = result_util.print_results('Recog (w4)', hybp_recog_w4_all, hybp_ans_all)

print('Overall ttest_rel results:', ttest_rel(hybp_last_all, hybp_recog_all))

# contingency_table = np.array([
#     [np.sum(hybp_last_all == 0), np.sum(hybp_last_all == 1)],
#     [np.sum(hybp_recog_all == 0), np.sum(hybp_recog_all == 1)],
# ])
# print('Overall chi2_contingency results:', chi2_contingency(contingency_table))

# ----- 不同錯誤權重的 DET curves

# plt.figure()

# for fold in range(args.n_fold_train):
#     plt.plot(det_va_all[fold][0], det_va_all[fold][1], '.-', label='VA {}'.format(fold), markersize=1)
# for fold in range(args.n_fold_train):
#     plt.plot(det_te_all[fold][0], det_te_all[fold][1], '.-', label='TE {}'.format(fold), markersize=1)

# plt.plot(m_ov[0, 1] / (m_ov[0, 0] + m_ov[0, 1]), m_ov[1, 0] / (m_ov[1, 1] + m_ov[1, 0]), '.', label='TE voted by best VA F-1', markersize=9)
# plt.plot(m_w1[0, 1] / (m_w1[0, 0] + m_w1[0, 1]), m_w1[1, 0] / (m_w1[1, 1] + m_w1[1, 0]), '.', label='TE voted by min VA FPR + FNR', markersize=9)
# plt.plot(m_w2[0, 1] / (m_w2[0, 0] + m_w2[0, 1]), m_w2[1, 0] / (m_w2[1, 1] + m_w2[1, 0]), '.', label='TE voted by min VA FPR + 2 * FNR', markersize=9)
# plt.plot(m_w4[0, 1] / (m_w4[0, 0] + m_w4[0, 1]), m_w4[1, 0] / (m_w4[1, 1] + m_w4[1, 0]), '.', label='TE voted by min VA FPR + 4 * FNR', markersize=9)

# plt.xlabel('FPR')
# plt.ylabel('FNR')
# plt.legend()

# ----- 脈壓差分布

# plt.hist(hybp_raw_ans_all[:,0]-hybp_raw_ans_all[:,1], bins=50)

# ----- 分來訪次數看結果

m, p, r, f, a = result_util.get_prfa(
    visit_count_all,
    sorted(list(set(visit_count_all))),
    'Recog CNT {}',
    hybp_recog_all,
    hybp_last_all,
    hybp_ans_all
)
print(sorted(list(set(visit_count_all))), m, p, r, f, a)
result_util.plot_4plot(
    m,
    {'recog': p['recog'], 'overall': p_overall,'naive': p['naive'],},
    {'recog': r['recog'], 'overall': r_overall,'naive': r['naive'],},
    {'recog': f['recog'], 'overall': f_overall,'naive': f['naive'],},
    {'recog': a['recog'], 'overall': a_overall,'naive': a['naive'],},
    [i+1 for i in range(5)],
    [1, 5],
    'Visit Counts',
)

# ----- 分有無病史看結果

m, p, r, f, a = result_util.get_prfa(
    mh_all,
    (1, 0),
    'Recog MH {}',
    hybp_recog_all,
    hybp_last_all,
    hybp_ans_all
)
result_util.plot_1plot_2bar(
    m,
    {'recog': p['recog'], 'overall': p_overall, 'naive': p['naive']},
    {'recog': r['recog'], 'overall': r_overall, 'naive': r['naive']},
    {'recog': f['recog'], 'overall': f_overall, 'naive': f['naive']},
    {'recog': a['recog'], 'overall': a_overall, 'naive': a['naive']},
    ['Has MH', 'No MH'],
)

# ----- 分性別看結果

# m, p, r, f, a = result_util.get_prfa(
#     sex_all,
#     (1, 0),
#     'Recog SEX {}',
#     hybp_recog_all,
#     hybp_last_all,
#     hybp_ans_all
# )
# result_util.plot_1plot_2bar(
#     m,
#     {'recog': p['recog'], 'overall': p_overall, 'naive': p['naive']},
#     {'recog': r['recog'], 'overall': r_overall, 'naive': r['naive']},
#     {'recog': f['recog'], 'overall': f_overall, 'naive': f['naive']},
#     {'recog': a['recog'], 'overall': a_overall, 'naive': a['naive']},
#     ['Male', 'Female'],
# )

# ----- 分服藥與否看結果

# m, p, r, f, a = result_util.get_prfa(
#     sick_drug_r_all,
#     (1, 0),
#     'Recog SDR {}',
#     hybp_recog_all,
#     hybp_last_all,
#     hybp_ans_all
# )
# result_util.plot_1plot_2bar(
#     m,
#     {'recog': p['recog'], 'overall': p_overall, 'naive': p['naive']},
#     {'recog': r['recog'], 'overall': r_overall, 'naive': r['naive']},
#     {'recog': f['recog'], 'overall': f_overall, 'naive': f['naive']},
#     {'recog': a['recog'], 'overall': a_overall, 'naive': a['naive']},
#     ['Yes', 'No'],
# )


# ----- 分最後兩次來訪間隔看結果

# visit_interval_count = {}
# for vi in visit_interval_all:
#     if vi not in visit_interval_count:
#         visit_interval_count[vi] = 0
#     visit_interval_count[vi] += 1
# y = []
# for vi in sorted(visit_interval_count.keys()):
#     print(vi, visit_interval_count[vi])
#     y.append(visit_interval_count[vi])
# plt.figure()
# plt.plot(sorted(visit_interval_count.keys()), y, '.-')

# bins = np.linspace(visit_interval_all.min(), visit_interval_all.max(), 7)
# idx_tn = np.where(np.logical_and(hybp_ans_all == 0, hybp_recog_all == 0))[0]
# idx_fp = np.where(np.logical_and(hybp_ans_all == 0, hybp_recog_all == 1))[0]
# idx_fn = np.where(np.logical_and(hybp_ans_all == 1, hybp_recog_all == 0))[0]
# idx_tp = np.where(np.logical_and(hybp_ans_all == 1, hybp_recog_all == 1))[0]
# idx_tn_naive = np.where(np.logical_and(hybp_ans_all == 0, hybp_last_all == 0))[0]
# idx_fp_naive = np.where(np.logical_and(hybp_ans_all == 0, hybp_last_all == 1))[0]
# idx_fn_naive = np.where(np.logical_and(hybp_ans_all == 1, hybp_last_all == 0))[0]
# idx_tp_naive = np.where(np.logical_and(hybp_ans_all == 1, hybp_last_all == 1))[0]
# hist_tn = np.histogram(visit_interval_all[idx_tn], bins=bins)[0]
# hist_fp = np.histogram(visit_interval_all[idx_fp], bins=bins)[0]
# hist_fn = np.histogram(visit_interval_all[idx_fn], bins=bins)[0]
# hist_tp = np.histogram(visit_interval_all[idx_tp], bins=bins)[0]
# hist_tn_naive = np.histogram(visit_interval_all[idx_tn_naive], bins=bins)[0]
# hist_fp_naive = np.histogram(visit_interval_all[idx_fp_naive], bins=bins)[0]
# hist_fn_naive = np.histogram(visit_interval_all[idx_fn_naive], bins=bins)[0]
# hist_tp_naive = np.histogram(visit_interval_all[idx_tp_naive], bins=bins)[0]
# beta = 1
# hist_all = hist_tn + hist_fp + hist_fn + hist_tp
# precision = hist_tp / (hist_tp + hist_fp) + 1e-6
# recall = hist_tp / (hist_tp + hist_fn) + 1e-6
# f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
# accuracy = (hist_tn + hist_tp) / hist_all
# hist_all_naive = hist_tn_naive + hist_fp_naive + hist_fn_naive + hist_tp_naive
# precision_naive = hist_tp_naive / (hist_tp_naive + hist_fp_naive) + 1e-6
# recall_naive = hist_tp_naive / (hist_tp_naive + hist_fn_naive) + 1e-6
# f_score_naive = (1 + beta ** 2) * precision_naive * recall_naive / (beta ** 2 * precision_naive + recall_naive)
# accuracy_naive = (hist_tn_naive + hist_tp_naive) / hist_all_naive

# print('People counts:', hist_all)
# result_util.plot_4plot(
#     hist_all,
#     {'recog': precision, 'overall': p_overall,'naive': precision_naive,},
#     {'recog': recall, 'overall': r_overall,'naive': recall_naive,},
#     {'recog': f_score, 'overall': f_overall,'naive': f_score_naive,},
#     {'recog': accuracy, 'overall': a_overall,'naive': accuracy_naive,},
#     bins[:-1],
#     [-3, -0.5],
#     'Duration of last 2 visits',
# )

if not args.no_plot:
    plt.show()
