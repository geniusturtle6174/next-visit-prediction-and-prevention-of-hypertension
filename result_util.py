import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

import util


def get_error_type(recog, ans):
    '''
    0: TN, 1: FP, 2: FN, 3: TP
    '''
    return int(ans * 2 + recog)


def print_results(method, recog, ans, beta=1, calc_only=False):

    conf_mat = util.get_conf_mat(recog, ans)
    conf_mat_r = 100 * conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    precision = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1] + 1e-6) # P = TP / (TP + FP)
    recall = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0] + 1e-6) # R = TP / (TP + FN)
    f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-6)
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(conf_mat)
    assert np.abs(np.mean(recog == ans) - accuracy) <= 1e-6
    if not calc_only:
        print('=====', method, '=====')
        print('all_num:', len(ans))
        print('hybp_num:', sum(ans))
        print('TN: {:.0f} ({:2.2f}%)\nFP: {:.0f} ({:2.2f}%)\nFN: {:.0f} ({:2.2f}%)\nTP: {:.0f} ({:2.2f}%)'.format(
            conf_mat[0, 0],
            conf_mat_r[0, 0],
            conf_mat[0, 1],
            conf_mat_r[0, 1],
            conf_mat[1, 0],
            conf_mat_r[1, 0],
            conf_mat[1, 1],
            conf_mat_r[1, 1],
        ))
        print('P/R: {:.2f}, {:.2f}'.format(precision * 100, recall * 100))
        print('F/A: {:.2f}, {:.2f}'.format(f_score * 100, 100 * accuracy))
    return conf_mat, precision, recall, f_score, accuracy


def get_prfa(data_all, iter_data, fmt_str, hybp_recog_all, hybp_last_all, hybp_ans_all):
    m_all = []
    p_all = []
    r_all = []
    f_all = []
    a_all = []
    p_naive_all = []
    r_naive_all = []
    f_naive_all = []
    a_naive_all = []
    for d in iter_data:
        idx = np.where(data_all == d)[0]
        if len(idx) > 0:
            m, p, r, f, a = print_results(fmt_str.format(d), hybp_recog_all[idx], hybp_ans_all[idx])
            _, p_n, r_n, f_n, a_n = print_results(fmt_str.format(d), hybp_last_all[idx], hybp_ans_all[idx], calc_only=True)
            print('Local t-test results:', ttest_rel(hybp_last_all[idx], hybp_recog_all[idx]))
            m_all.append(np.sum(m))
            p_all.append(p)
            r_all.append(r)
            f_all.append(f)
            a_all.append(a)
            p_naive_all.append(p_n)
            r_naive_all.append(r_n)
            f_naive_all.append(f_n)
            a_naive_all.append(a_n)
    m_all = np.array(m_all)
    return m_all, \
        {'recog': p_all, 'naive': p_naive_all}, \
        {'recog': r_all, 'naive': r_naive_all}, \
        {'recog': f_all, 'naive': f_naive_all}, \
        {'recog': a_all, 'naive': a_naive_all},


def plot_4plot(people_cnts, p_curves, r_curves, f_curves, a_curves, x, x_oa, x_label):
    plt.figure()

    plt.subplot(2, 2, 1)
    # plt.plot(x, people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.plot(x, p_curves['recog'], '.-', label='Ours')
    plt.plot(x_oa, [p_curves['overall'], p_curves['overall'],], '--', label='P (recog overall)')
    plt.plot(x, p_curves['naive'], '.-', label='Baseline')
    plt.xlabel(x_label)
    plt.title('Precision')
    plt.legend()

    plt.subplot(2, 2, 2)
    # plt.plot(x, people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.plot(x, r_curves['recog'], '.-', label='Ours')
    plt.plot(x_oa, [r_curves['overall'], r_curves['overall'],], '--', label='R (recog overall)')
    plt.plot(x, r_curves['naive'], '.-', label='Baseline')
    plt.xlabel(x_label)
    plt.title('Recall')
    plt.legend()

    plt.subplot(2, 2, 3)
    # plt.plot(x, people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.plot(x, f_curves['recog'], '.-', label='Ours')
    plt.plot(x_oa, [f_curves['overall'], f_curves['overall'],], '--', label='F (recog overall)')
    plt.plot(x, f_curves['naive'], '.-', label='Baseline')
    plt.xlabel(x_label)
    plt.title('F1-score')
    plt.legend()

    plt.subplot(2, 2, 4)
    # plt.plot(x, people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.plot(x, a_curves['recog'], '.-', label='Ours')
    plt.plot(x_oa, [a_curves['overall'], a_curves['overall'],], '--', label='Acc (recog overall)')
    plt.plot(x, a_curves['naive'], '.-', label='Baseline')
    plt.xlabel(x_label)
    plt.title('Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(x, p_curves['recog'], '.-', label='Precision (Ours)')
    plt.plot(x, r_curves['recog'], '.-', label='Recall (Ours)')
    plt.plot(x, f_curves['recog'], '.-', label='F1-score (Ours)')
    plt.plot(x, p_curves['naive'], '.-', label='Precision (Baseline)')
    plt.plot(x, r_curves['naive'], '.-', label='Recall (Baseline)')
    plt.plot(x, f_curves['naive'], '.-', label='F-score (Baseline)')
    plt.xlabel(x_label)
    plt.legend()
    print('----- Plotted data -----')
    print(people_cnts)
    print(p_curves['recog'])
    print(r_curves['recog'])
    print(f_curves['recog'])
    print(p_curves['naive'])
    print(r_curves['naive'])
    print(f_curves['naive'])
    print('----------')


def plot_1plot_2bar(people_cnts, p_curves, r_curves, f_curves, a_curves, x_ticks):
    x = np.array([0, 1])

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.bar(x - 0.1, [p_curves['recog'][0], p_curves['recog'][1]], 0.2, label='P (recog)')
    plt.bar(x + 0.1, [p_curves['naive'][0], p_curves['naive'][1]], 0.2, label='P (naive)')
    plt.plot([0, 1], [p_curves['overall'], p_curves['overall']], '--', label='P (recog overall)')
    plt.xticks(x, x_ticks)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot([0, 1], people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.bar(x - 0.1, [r_curves['recog'][0], r_curves['recog'][1]], 0.2, label='R (recog)')
    plt.bar(x + 0.1, [r_curves['naive'][0], r_curves['naive'][1]], 0.2, label='R (naive)')
    plt.plot([0, 1], [r_curves['overall'], r_curves['overall']], '--', label='R (recog overall)')
    plt.xticks(x, x_ticks)
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot([0, 1], people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.bar(x - 0.1, [f_curves['recog'][0], f_curves['recog'][1]], 0.2, label='F (recog)')
    plt.bar(x + 0.1, [f_curves['naive'][0], f_curves['naive'][1]], 0.2, label='F (naive)')
    plt.plot([0, 1], [f_curves['overall'], f_curves['overall']], '--', label='F (recog overall)')
    plt.xticks(x, x_ticks)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot([0, 1], people_cnts/np.max(people_cnts), '.-', label='People Num (div by {})'.format(np.max(people_cnts)))
    plt.bar(x - 0.1, [a_curves['recog'][0], a_curves['recog'][1]], 0.2, label='Acc (recog)')
    plt.bar(x + 0.1, [a_curves['naive'][0], a_curves['naive'][1]], 0.2, label='Acc (naive)')
    plt.plot([0, 1], [a_curves['overall'], a_curves['overall']], '--', label='Acc (recog overall)')
    plt.xticks(x, x_ticks)
    plt.legend()

    print('----- Plotted data -----')
    print(people_cnts)
    print(p_curves['recog'])
    print(r_curves['recog'])
    print(f_curves['recog'])
    print(p_curves['naive'])
    print(r_curves['naive'])
    print(f_curves['naive'])
    print('----------')
