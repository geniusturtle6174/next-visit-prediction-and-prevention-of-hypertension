import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression

from fea_names import FEA_NAMES_APAMI as FEA_NAMES
from fea_names import FEA_NAMES_APAMI_TO_IDX as FEA_NAMES_TO_IDX
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)
plt.rcParams["savefig.directory"] = '.'

parser = argparse.ArgumentParser()
parser.add_argument('fea_dir')
parser.add_argument('--use_dims', '-d', type=int, nargs='+')
parser.add_argument('--no_patch', action='store_true')
args = parser.parse_args()

fea_all = []
total_count = 0
need_patch_count = 0
patch_count = 0
num_pos_when_patch = []
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
            fea_one_len[n], stat = patch_for_one_person(fea_one_len[n], return_stats=True)
            # total_count += fea_one_len[n].size
            # total_count += stat['total_idx']
            total_count += 104 * fea_one_len[n].shape[0]
            need_patch_count += stat['need_patch']
            patch_count += stat['num_patched']
            num_pos_when_patch.extend(stat['num_pos_when_patch'])
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    for fea, ans in zip(fea_one_len, ans_one_len):
        fea_all.append(fea)

fea_all = np.vstack(fea_all)
num_pos_when_patch = np.array(num_pos_when_patch)
print('Overall shapes:', fea_all.shape)
if not args.no_patch:
    print('Need patch ratio: {}/{} = {:.4f}%'.format(need_patch_count, total_count, need_patch_count/(total_count+1e-6)*100))
    print('Patched ratio: {}/{} = {:.4f}%'.format(patch_count, total_count, patch_count/(total_count+1e-6)*100))
    print('Num pos values when patching: list len {}, 1-num {}, mean {:.4f}, max {}'.format(
        num_pos_when_patch.shape,
        np.sum(num_pos_when_patch==1),
        np.mean(num_pos_when_patch),
        np.max(num_pos_when_patch),
    ))

idx_male = np.where(fea_all[:, 0] == 1)[0]
idx_female = np.where(fea_all[:, 1] == 1)[0]


def observe_and_get_slope(name_y, fea_all, idx_male, idx_female, row, col, nth, name_x='bmi'):
    dict_for_display = {
        'waist_circum': 'Waist Circumference (cm)',
        'hip_circum': 'Hip Circumference (cm)',
        'whr': 'Waist-hip Ratio',
        'g_fat': 'Body Fat Ratio (%)',
        'sbp': 'Systolic Blood Pressure (mmHg)',
        'dbp': 'Diastolic Blood Pressure (mmHg)',
        '(s+2d)/3': '(s+2d)/3',
        'rf_egfr': 'EGFR',
        'bmi': 'BMI',
    }
    name_y_to_display = dict_for_display.get(name_y, name_y)
    name_x_to_display = dict_for_display.get(name_x, name_x)

    x_m = fea_all[idx_male, FEA_NAMES_TO_IDX[name_x]]
    x_f = fea_all[idx_female, FEA_NAMES_TO_IDX[name_x]]
    y_m = fea_all[idx_male, FEA_NAMES_TO_IDX[name_y]]
    y_f = fea_all[idx_female, FEA_NAMES_TO_IDX[name_y]]

    keep_m = np.where((x_m > 0) & (y_m > 0))[0]
    keep_f = np.where((x_f > 0) & (y_f > 0))[0]

    x_m = x_m[keep_m]
    x_f = x_f[keep_f]
    y_m = y_m[keep_m]
    y_f = y_f[keep_f]

    model_m = LinearRegression()
    model_f = LinearRegression()
    model_m.fit(x_m[:, np.newaxis], y_m[:, np.newaxis])
    model_f.fit(x_f[:, np.newaxis], y_f[:, np.newaxis])
    coef_m = model_m.coef_[0][0]
    coef_f = model_f.coef_[0][0]
    intr_m = model_m.intercept_[0]
    intr_f = model_f.intercept_[0]
    corr_m = scipy.stats.pearsonr(x_m, y_m)[0]
    corr_f = scipy.stats.pearsonr(x_f, y_f)[0]
    print(name_y, 'male', coef_m, intr_m, corr_m)
    print(name_y, 'female', coef_f, intr_f, corr_f)

    plt.subplot(row, col, nth)
    plt.plot(x_m, y_m, '.', label='Male', markersize=3)
    plt.plot(x_f, y_f, '.', label='Female', markersize=3)
    plt.plot([10, 50], [10 * coef_m + intr_m, 50 * coef_m + intr_m], label='Fitting result (male)')
    plt.plot([10, 50], [10 * coef_f + intr_f, 50 * coef_f + intr_f], label='Fitting result (female)')
    plt.xlabel(name_x_to_display)
    plt.ylabel(name_y_to_display)
    plt.title('{} vs {}'.format(name_y_to_display, name_x_to_display))
    if nth == 1:
        plt.legend(loc='lower right', bbox_to_anchor=(1.05, 0), framealpha=0.99)


# observe_and_get_slope('au_r', fea_all, idx_male, idx_female, 1, 1, 1, name_x='age')

print('Checking highly correlated factors...')
for name in FEA_NAMES:
    x_m = fea_all[idx_male, FEA_NAMES_TO_IDX['bmi']]
    x_f = fea_all[idx_female, FEA_NAMES_TO_IDX['bmi']]
    y_m = fea_all[idx_male, FEA_NAMES_TO_IDX[name.split(' ')[0]]]
    y_f = fea_all[idx_female, FEA_NAMES_TO_IDX[name.split(' ')[0]]]

    keep_m = np.where((x_m > 0) & (y_m > 0))[0]
    keep_f = np.where((x_f > 0) & (y_f > 0))[0]

    if len(keep_m) <= 100 or len(keep_f) <= 100:
        continue

    x_m = x_m[keep_m]
    x_f = x_f[keep_f]
    y_m = y_m[keep_m]
    y_f = y_f[keep_f]

    corr_m = scipy.stats.pearsonr(x_m, y_m)[0]
    corr_f = scipy.stats.pearsonr(x_f, y_f)[0]
    if corr_m >= 0.5 or corr_f >= 0.5:
        print(name, corr_m, corr_f)

plt.figure()
observe_and_get_slope('waist_circum', fea_all, idx_male, idx_female, 2, 2, 1)
observe_and_get_slope('hip_circum', fea_all, idx_male, idx_female, 2, 2, 2)
observe_and_get_slope('whr', fea_all, idx_male, idx_female, 2, 2, 3)
observe_and_get_slope('g_fat', fea_all, idx_male, idx_female, 2, 2, 4)
# observe_and_get_slope('sbp-dbp', fea_all, idx_male, idx_female, 2, 2, 1)
# observe_and_get_slope('lf_alb/lf_glo', fea_all, idx_male, idx_female, 2, 2, 2)
# observe_and_get_slope('wbc_e', fea_all, idx_male, idx_female, 2, 2, 3)
# observe_and_get_slope('wbc_l', fea_all, idx_male, idx_female, 2, 2, 4)
plt.subplots_adjust(hspace=0.38)
plt.show()
exit()
if args.use_dims is not None:
    print('Use only dims:', args.use_dims)
    fea_all = fea_all[:, args.use_dims]
    print('New shape:', fea_all.shape)

fea_all = fea_all[::1, :]
corr_mat = np.zeros((fea_all.shape[1], fea_all.shape[1]))
for i in range(fea_all.shape[1]):
    print('Computing corr_mat {}/{}, has-val rate:\t{:.4f}'.format(
        i, FEA_NAMES[i], 100 * np.mean(fea_all[:, i] >= 0)
    ))
    for j in range(fea_all.shape[1]):
        corr_mat[i, j] = scipy.stats.pearsonr(fea_all[:, i], fea_all[:, j])[0]

plt.figure()
plt.imshow(corr_mat)
plt.yticks(
    ticks=np.arange(fea_all.shape[1]),
    labels=[fn.split(' ')[0] for fn in FEA_NAMES] if args.use_dims is None else [FEA_NAMES[d].split(' ')[0] for d in args.use_dims]
)
plt.colorbar()

plt.show()
