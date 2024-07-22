import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import util

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('result_dir')
parser.add_argument('--n_fold_train', '-n', type=int, default=5)
args = parser.parse_args()

print('Loading npy files...')
results = []
for i in range(args.n_fold_train):
    results.append(np.load(os.path.join(args.result_dir, 'result_{}.npy'.format(i))))
print('{} mats finished, shape: {}'.format(args.n_fold_train, results[0].shape))

print('Loading pkl files...')
idx_to_chid = util.load_pkl(os.path.join(args.result_dir, 'idx_to_chid.pkl'))
chid_to_fea = util.load_pkl(os.path.join(args.result_dir, 'chid_to_fea.pkl'))

chid_to_result = {}

# First fold
for row in results[0]:
    chid = row[0]
    ans = row[1:3]
    recog = row[3:]
    if chid not in chid_to_result:
        chid_to_result[chid] = {
            'ans': ans,
            'recog': [recog],
        }

# Remaining folds
for result in results[1:]:
    for row in result:
        chid = row[0]
        ans = row[1:3]
        recog = row[3:]
        assert np.sum(np.abs(ans - chid_to_result[chid]['ans'])) < 1e-6
        chid_to_result[chid]['recog'].append(recog)

sbp_last_all = []
dbp_last_all = []
sbp_recog_all = []
dbp_recog_all = []
sbp_ans_all = []
dbp_ans_all = []
for chid in chid_to_result:
    recog = np.vstack(chid_to_result[chid]['recog'])
    ans = chid_to_result[chid]['ans']
    sbp_recog = np.mean(recog[:, 0])
    dbp_recog = np.mean(recog[:, 1])
    # if (abs(ans[0] - sbp_recog) >= 30 and ans[0] > 130 and sbp_recog < 130) or (abs(ans[1] - dbp_recog) >= 30 and ans[1] > 80 * dbp_recog < 80):
    #     print('=== chid: {} {}, ans: {}/{}, recog: {:.2f}/{:.2f} ==='.format(chid, idx_to_chid[chid], ans[0], ans[1], sbp_recog, dbp_recog))
    #     for fea in chid_to_fea[idx_to_chid[chid]]['fea']:
    #         print(' '.join(['{:.2f}'.format(f) for f in fea]))
    last_sbp = chid_to_fea[idx_to_chid[chid]]['fea'][-1, -3]
    last_dbp = chid_to_fea[idx_to_chid[chid]]['fea'][-1, -2]
    sbp_last_all.append(last_sbp)
    dbp_last_all.append(last_dbp)
    sbp_recog_all.append(sbp_recog)
    dbp_recog_all.append(dbp_recog)
    sbp_ans_all.append(ans[0])
    dbp_ans_all.append(ans[1])

sbp_last_all = np.array(sbp_last_all)
dbp_last_all = np.array(dbp_last_all)
sbp_ans_all = np.array(sbp_ans_all)
dbp_ans_all = np.array(dbp_ans_all)
sbp_recog_all = np.array(sbp_recog_all)
dbp_recog_all = np.array(dbp_recog_all)


def print_and_plot(method, sbp_result, dbp_result, sbp_ans, dbp_ans):
    if method == 'Naive':
        keep_idx = np.where(np.minimum(sbp_result, dbp_result) >= 0)[0]
        sbp_error = sbp_ans[keep_idx] - sbp_result[keep_idx]
        dbp_error = dbp_ans[keep_idx] - dbp_result[keep_idx]
    else:
        sbp_error = sbp_ans - sbp_result
        dbp_error = dbp_ans - dbp_result

    print(method, 'sbp mae: {:.2f}'.format(np.mean(np.abs(sbp_error))))
    print(method, 'dbp mae: {:.2f}'.format(np.mean(np.abs(dbp_error))))
    print(method, 'sbp min/max/std: {:.2f} {:.2f} {:.2f}'.format(np.min(sbp_error), np.max(sbp_error), np.std(sbp_error)))
    print(method, 'dbp min/max/std: {:.2f} {:.2f} {:.2f}'.format(np.min(dbp_error), np.max(dbp_error), np.std(dbp_error)))

    for s_ths in (130, 140):
        ans = (sbp_ans >= s_ths).astype('int')
        recog = (sbp_result >= s_ths).astype('int')
        conf_mat = util.get_conf_mat(recog, ans)
        print('===== S-Ths: {} ====='.format(s_ths))
        print(conf_mat)

    for d_ths in (80, 90):
        ans = (dbp_ans >= d_ths).astype('int')
        recog = (dbp_result >= d_ths).astype('int')
        conf_mat = util.get_conf_mat(recog, ans)
        print('===== D-Ths: {} ====='.format(d_ths))
        print(conf_mat)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.hist(sbp_error, bins=np.arange(-50, 85, 1))
    plt.title('SBP Error, min {:.2f}, max {:.2f}, std {:.2f}, MAE: {:.2f}'.format(
        np.min(sbp_error), np.max(sbp_error), np.std(sbp_error), np.mean(np.abs(sbp_error)),
    ))
    # plt.xlabel('Error')
    plt.ylabel('Count')

    plt.subplot(2, 2, 2)
    plt.plot(sbp_ans[::5], sbp_result[::5], '.')
    plt.title('SBP')
    # plt.xlabel('Groundtruth')
    plt.ylabel('Prediction')

    plt.subplot(2, 2, 3)
    plt.hist(dbp_error, bins=np.arange(-50, 85, 1))
    plt.title('DBP Error, min {:.2f}, max {:.2f}, std {:.2f}, MAE: {:.2f}'.format(
        np.min(dbp_error), np.max(dbp_error), np.std(dbp_error), np.mean(np.abs(dbp_error)),
    ))
    plt.xlabel('Error')
    plt.ylabel('Count')

    plt.subplot(2, 2, 4)
    plt.plot(dbp_ans[::5], dbp_result[::5], '.')
    plt.title('DBP')
    plt.xlabel('Groundtruth')
    plt.ylabel('Prediction')


print_and_plot('Naive', sbp_last_all, dbp_last_all, sbp_ans_all, dbp_ans_all)
print_and_plot('Recog', sbp_recog_all, dbp_recog_all, sbp_ans_all, dbp_ans_all)

plt.show()
