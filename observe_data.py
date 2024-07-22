import csv

import matplotlib.pyplot as plt
import numpy as np

import util

MAIN_NAME = 'mjdata20211225'

print('Loading files...')
with open('data/' + MAIN_NAME + '.csv', 'r', encoding='utf-8') as fin:
    cnt = list(csv.reader(fin))
print('Finished.')

print('Num rec:', len(cnt)-1)

header = cnt[0]
print('Num col:', len(header))

idx_pid = header.index('pid')
idx_year = header.index('yr')
idx_month = header.index('mon')
idx_age = header.index('age')
idx_gender = header.index('gender')
idx_g_sll = header.index('g_sll')
idx_g_slr = header.index('g_slr')
idx_g_ssl = header.index('g_ssl')
idx_g_ssr = header.index('g_ssr')
idx_g_dll = header.index('g_dll')
idx_g_dlr = header.index('g_dlr')
idx_g_dsl = header.index('g_dsl')
idx_g_dsr = header.index('g_dsr')

pid_to_data = {}
mat_all = []
for idx, arr in enumerate(cnt[1:]):
    if idx % 100000 == 0:
        print('Processing {}/{}...'.format(idx, len(cnt)-1))
    pid = arr[idx_pid]
    year = int(arr[idx_year])
    month = int(arr[idx_month])
    age = int(arr[idx_age]) if arr[idx_age] else 999
    gender = arr[idx_gender]
    yrmon = year * 100 + month
    sbp = max(list(map(util.parse_float, [
            arr[idx_g_sll], arr[idx_g_slr], arr[idx_g_ssl], arr[idx_g_ssr],
    ])))
    dbp = max(list(map(util.parse_float, [
            arr[idx_g_dll], arr[idx_g_dlr], arr[idx_g_dsl], arr[idx_g_dsr],
    ])))

    if year < 2010:
        continue

    if sbp <= 0:
        continue

    if dbp <= 0:
        continue

    if not pid in pid_to_data:
        pid_to_data[pid] = {
            'count': 0,
            'yrmon': set(),
            'age': 9999,
            'bp_s': [],
            'bp_d': [],
        }

    if yrmon in pid_to_data[pid]['yrmon']:
        continue

    pid_to_data[pid]['count'] += 1
    pid_to_data[pid]['yrmon'].add(yrmon)
    pid_to_data[pid]['age'] = min(pid_to_data[pid]['age'], age)
    pid_to_data[pid]['gender'] = gender
    pid_to_data[pid]['bp_s'].append(sbp)
    pid_to_data[pid]['bp_d'].append(dbp)
    arr_float = list(map(util.parse_float, arr[1:]))
    if len(arr_float) != len(header) - 1:
        print(arr)
        print(arr_float)
        exit()
    mat_all.append(arr_float)
    # if idx == 1000:
    #     break

print('Min yrmon:', min([min(pid_to_data[pid]['yrmon']) for pid in pid_to_data]))
print('Max yrmon:', max([max(pid_to_data[pid]['yrmon']) for pid in pid_to_data]))
print('Num people:', len(pid_to_data))

gender = [pid_to_data[pid]['gender'] for pid in pid_to_data]
print('Stat of gender: all {}, 1 {}, 2 {}'.format(
    len(gender),
    sum([g == '1' for g in gender]),
    sum([g == '2' for g in gender]),
))

age = [pid_to_data[pid]['age'] for pid in pid_to_data]
print('Min age:', min(age))

print('')


def print_distribution(title, x_global, cnt_global, x_male, cnt_male, x_female, cnt_female):
    xg_to_idx = {x: i for i, x in enumerate(x_global)}
    cnt_male_new = np.zeros_like(cnt_global)
    cnt_female_new = np.zeros_like(cnt_global)
    for x, c in zip(x_male, cnt_male):
        cnt_male_new[xg_to_idx[x]] = c
    for x, c in zip(x_female, cnt_female):
        cnt_female_new[xg_to_idx[x]] = c
    print(title)
    for x, c_g, c_m, c_f in zip(x_global, cnt_global, cnt_male_new, cnt_female_new):
        print(x, c_g, c_m, c_f)
    print('')


count = [pid_to_data[pid]['count'] for pid in pid_to_data]
count_male = [pid_to_data[pid]['count'] for pid in pid_to_data if pid_to_data[pid]['gender'] == '1']
count_female = [pid_to_data[pid]['count'] for pid in pid_to_data if pid_to_data[pid]['gender'] == '2']
visit, visit_cnt = np.unique(count, return_counts=True)
visit_male, visit_cnt_male = np.unique(count_male, return_counts=True)
visit_female, visit_cnt_female = np.unique(count_female, return_counts=True)
print('Stat of visit count: min {}, mean {}, median {}, q-75 {}, q-90 {}, max {}, sum {}'.format(
    np.min(count),
    np.mean(count),
    np.median(count),
    np.quantile(count, 0.75),
    np.quantile(count, 0.90),
    np.max(count),
    np.sum(count),
))
print_distribution(
    'Distribution of visit count:', visit, visit_cnt, visit_male, visit_cnt_male, visit_female, visit_cnt_female
)

interval_all = []
interval_all_male = []
interval_all_female = []
for pid in pid_to_data:
    this_yrmon = sorted(list(map(util.dt_to_float_ym, pid_to_data[pid]['yrmon'])))
    if len(this_yrmon) > 1:
        this_interval = [float('{:.4f}'.format(a - b)) for a, b in zip(this_yrmon[1:], this_yrmon[:-1])]
        interval_all.extend(this_interval)
        if pid_to_data[pid]['gender'] == '1':
            interval_all_male.extend(this_interval)
        elif pid_to_data[pid]['gender'] == '2':
            interval_all_female.extend(this_interval)
interval, interval_cnt = np.unique(interval_all, return_counts=True)
interval_male, interval_cnt_male = np.unique(interval_all_male, return_counts=True)
interval_female, interval_cnt_female = np.unique(interval_all_female, return_counts=True)
print_distribution(
    'Distribution of interval:',
    interval,
    interval_cnt,
    interval_male,
    interval_cnt_male,
    interval_female,
    interval_cnt_female,
)

hybp = [sbp >= 130 or dbp >= 80 for pid in pid_to_data for sbp, dbp in zip(pid_to_data[pid]['bp_s'], pid_to_data[pid]['bp_d'])]
print('HYBP ratio: {:.2f}%'.format(np.mean(hybp)*100))

bp_s = [pid_to_data[pid]['bp_s'][-1] for pid in pid_to_data]
print('Stat of last BP-S: count {}, min {}, mean {:.2f}, median {}, max {}'.format(
    len(bp_s), np.min(bp_s), np.mean(bp_s), np.median(bp_s), np.max(bp_s),
))

bp_s = [pid_to_data[pid]['bp_s'][-1] for pid in pid_to_data if pid_to_data[pid]['bp_s'][-1] > 0]
print('Stat of last BP-S (w/o missing): count {}, min {}, mean {:.2f}, median {}, max {}'.format(
    len(bp_s), np.min(bp_s), np.mean(bp_s), np.median(bp_s), np.max(bp_s),
))

bp_d = [pid_to_data[pid]['bp_d'][-1] for pid in pid_to_data]
print('Stat of last BP-D: count {}, min {}, mean {:.2f}, median {}, max {}'.format(
    len(bp_d), np.min(bp_d), np.mean(bp_d), np.median(bp_d), np.max(bp_d),
))

bp_d = [pid_to_data[pid]['bp_d'][-1] for pid in pid_to_data if pid_to_data[pid]['bp_d'][-1] > 0]
print('Stat of last BP-D (w/o missing): count {}, min {}, mean {:.2f}, median {}, max {}'.format(
    len(bp_d), np.min(bp_d), np.mean(bp_d), np.median(bp_d), np.max(bp_d),
))

# mat_all = np.array(mat_all)
# rec_has_value = mat_all > util.VAL_OF_EMPTY
# col_has_value_ratio = np.sum(rec_has_value, axis=0) / mat_all.shape[0]
# idx = np.argsort(col_has_value_ratio)
# for i in idx[:10]:
#     print(header[i+1], col_has_value_ratio[i])

# plt.figure()
# plt.plot(col_has_value_ratio[idx])
# plt.xlabel('Sorted Index')
# plt.ylabel('Has-Value Ratio')
# plt.show()

# bp_s_any_x = []
# bp_s_any_y = []
# bp_d_any_x = []
# bp_d_any_y = []
# bp_s_last_x = []
# bp_s_last_y = []
# bp_d_last_x = []
# bp_d_last_y = []
# for pid in pid_to_data:
#     bp_s_last_x.append(pid_to_data[pid]['bp_s'][-2])
#     bp_s_last_y.append(pid_to_data[pid]['bp_s'][-1])
#     bp_d_last_x.append(pid_to_data[pid]['bp_d'][-2])
#     bp_d_last_y.append(pid_to_data[pid]['bp_d'][-1])
    # for x, y in zip(pid_to_data[pid]['bp_s'][:-1], pid_to_data[pid]['bp_s'][1:]):
    #     bp_s_any_x.append(x)
    #     bp_s_any_y.append(y)
    # for x, y in zip(pid_to_data[pid]['bp_d'][:-1], pid_to_data[pid]['bp_d'][1:]):
    #     bp_d_any_x.append(x)
    #     bp_d_any_y.append(y)

# bp_s_last_diff = np.array(bp_s_last_y) - np.array(bp_s_last_x)
# plt.subplot(2, 1, 1)
# plt.hist(bp_s_last_diff, np.arange(-100, 100, 5))
# plt.title('Systolic BP Difference (last 2)')
# plt.ylabel('Count')

# bp_d_last_diff = np.array(bp_d_last_y) - np.array(bp_d_last_x)
# plt.subplot(2, 1, 2)
# plt.hist(bp_d_last_diff, np.arange(-100, 100, 5))
# plt.title('Diastolic BP Difference (last 2)')
# plt.xlabel('Difference')
# plt.ylabel('Count')

# plt.show()
