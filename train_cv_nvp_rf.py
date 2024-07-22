import argparse
import os
import warnings
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import util
from fea_util import get_time_delta_fea
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('fea_dir')
parser.add_argument('save_model_dir_name')
parser.add_argument('--n_fold', type=int, default=5)
parser.add_argument('--run_classification_prob', '-classify', action='store_true')
parser.add_argument('--hyp_ths', type=int, nargs=2, default=[130, 80])
parser.add_argument('--use_dims', '-d', type=int, nargs='+')
parser.add_argument('--no_use_dims', '-no_d', type=int, nargs='+')
parser.add_argument('--no_patch', action='store_true')
parser.add_argument('--delta', action='store_true')
parser.add_argument('--n_est', type=int, default=1000)
parser.add_argument('--max_depth', type=int, default=5)
parser.add_argument('--use_mh_type', type=int, default=-1, help='0: no, 1: yes, otherwise: all')
args = parser.parse_args()
print(args)

if args.use_dims is not None and args.no_use_dims is not None:
    raise Exception('`use_dims` and `no_use_dims` cannot be used at the same time!')

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
    fea_bp_min = np.min(fea_one_len, axis=1)[:, -2:]
    fea_bp_min = np.min(fea_bp_min, axis=1)
    keep_idx = np.where(fea_bp_min > 0)[0]
    fea_one_len = fea_one_len[keep_idx]
    ans_one_len = ans_one_len[keep_idx]
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    if not args.no_patch:
        print('\tPatching values...')
        for n in range(fea_one_len.shape[0]):
            fea_one_len[n] = patch_for_one_person(fea_one_len[n])
    if args.delta:
        print('\tGetting time delta...')
        fea_one_len_new = np.concatenate((fea_one_len, fea_one_len), axis=-1)
        for n in range(fea_one_len.shape[0]):
            fea_one_len_new[n] = get_time_delta_fea(fea_one_len[n])
        fea_one_len = np.copy(fea_one_len_new)
        print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tComputing pos ratio...')
    is_hybp = [sbp >= args.hyp_ths[0] or dbp >= args.hyp_ths[1] for sbp, dbp in zip(ans_one_len[:, 0], ans_one_len[:, 1])]
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
sbp_high_idx = np.where(np.max(fea_all[:, :, -2], axis=1) >= args.hyp_ths[0])[0]
dbp_high_idx = np.where(np.max(fea_all[:, :, -1], axis=1) >= args.hyp_ths[1])[0]
has_mh[sbp_high_idx] = 1
has_mh[dbp_high_idx] = 1
print('Finished, mh ratio: {:.4f}%'.format(100*np.mean(has_mh)))

if args.use_mh_type in (0, 1):
    keep_idx = np.where(has_mh == args.use_mh_type)[0]
    fea_all = fea_all[keep_idx]
    ans_all = ans_all[keep_idx]
    print('Use only mh type', args.use_mh_type, 'for training, shape:', fea_all.shape)

if args.use_dims is not None:
    print('Use only dims:', args.use_dims)
    fea_all = fea_all[:, :, args.use_dims]
    use_dims = deepcopy(args.use_dims)
    print('New shape:', fea_all.shape)
elif args.no_use_dims is not None:
    print('NO use this dims:', args.no_use_dims)
    use_dims = [i for i in range(fea_all.shape[-1]) if i not in args.no_use_dims]
    fea_all = fea_all[:, :, use_dims]
    print('New shape:', fea_all.shape)
else:
    use_dims = None

print('Adding is_pad feature and reshape')
is_pad = np.max(fea_all, axis=-1, keepdims=True)
fea_all = np.concatenate((fea_all, is_pad), axis=-1)
fea_all = np.reshape(fea_all, (fea_all.shape[0], -1))
print('Finished, shape:', fea_all.shape)

# print('Adding has_mh feature')
# fea_all = np.concatenate((fea_all, has_mh), axis=1)
# print('Finished, shape:', fea_all.shape)

sort_idx = np.lexsort((ans_all[:, 0]-args.hyp_ths[0], ans_all[:, 1]-args.hyp_ths[1]))
fea_all = fea_all[sort_idx]
ans_all = ans_all[sort_idx]
print('Sorted shapes:', sort_idx.shape, fea_all.shape, ans_all.shape)

tr_param = {
    'max_depth': args.max_depth,
    'run_classification_prob': args.run_classification_prob,
    'fea_dir': args.fea_dir,
    'fea_cfg': fea_cfg,
    'hyp_ths': args.hyp_ths,
    'many_to_many': False,
    'use_dims': use_dims,
    'no_patch': args.no_patch,
    'delta': args.delta,
    'use_mh_type': args.use_mh_type,
}

if tr_param['run_classification_prob']:
    raw_ans_all = np.copy(ans_all)
    sbp_high_idx = np.where(raw_ans_all[:, 0] >= tr_param['hyp_ths'][0])[0]
    dbp_high_idx = np.where(raw_ans_all[:, 1] >= tr_param['hyp_ths'][1])[0]
    ans_all = np.zeros(ans_all.shape[0])
    ans_all[sbp_high_idx] = 1
    ans_all[dbp_high_idx] = 1
    neg_ratio = np.mean(ans_all==0)
    pos_ratio = np.mean(ans_all==1)
    print('Use classification, ths: {}/{}, neg ratio: {:.4f}%, pos ratio: {:.4f}%'.format(
        tr_param['hyp_ths'][0],
        tr_param['hyp_ths'][1],
        100 * neg_ratio,
        100 * pos_ratio,
    ))

# --- Write config
if not os.path.exists(args.save_model_dir_name):
    os.makedirs(args.save_model_dir_name, 0o755)
    print('Model will be saved in {}'.format(args.save_model_dir_name))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.save_model_dir_name))
util.write_cfg(os.path.join(args.save_model_dir_name, 'config.yml'), tr_param)

models = {}
best_n_est_all = []
best_va_loss_all = []
data_num = fea_all.shape[0]
for fold in range(args.n_fold):

    valid_idx = np.where(np.arange(data_num)%args.n_fold==fold)[0]
    train_idx = np.where(np.arange(data_num)%args.n_fold!=fold)[0]

    print('Fold {}, train num {}, test num {}'.format(
        fold, len(train_idx), len(valid_idx),
    ))

    va_loss_fold = []
    tr_loss_fold = []
    best_va_loss_all.append(99999999)
    best_n_est_all.append(-1)
    for n_est in (600, 650, 700, 750, 800, 850, 900, 950, 1000):
        print('n_est:', n_est)
        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=tr_param['max_depth']
        )
        model.fit(fea_all[train_idx], ans_all[train_idx])
        va_pred = model.predict_proba(fea_all[valid_idx])
        tr_pred = model.predict_proba(fea_all[train_idx])
        va_loss = log_loss(ans_all[valid_idx], va_pred)
        tr_loss = log_loss(ans_all[train_idx], tr_pred)
        va_loss_fold.append(va_loss)
        tr_loss_fold.append(tr_loss)
        if va_loss < best_va_loss_all[-1]:
            print('Min loss found')
            best_va_loss_all[-1] = va_loss
            best_n_est_all[-1] = n_est

            util.save_pkl(os.path.join(args.save_model_dir_name, 'model_{}.pkl'.format(fold)), model)
            # np.save(os.path.join(args.save_model_dir_name, 'va_pred_{}.npy'.format(fold)), va_hybp_prob)
            # np.save(os.path.join(args.save_model_dir_name, 'va_ans_{}.npy'.format(fold)), ans_all[valid_idx])

    np.save(os.path.join(args.save_model_dir_name, 'va_loss_{}.npy'.format(fold)), va_loss_fold)
    np.save(os.path.join(args.save_model_dir_name, 'tr_loss_{}.npy'.format(fold)), tr_loss_fold)

util.write_cfg(os.path.join(args.save_model_dir_name, 'end.yml'), {'end': True})

best_va_loss_all = np.array(best_va_loss_all)
best_n_est_all = np.array(best_n_est_all)
print('Mean va loss: {:.6f}'.format(np.mean(best_va_loss_all)))
print('All va losses:', ['{:.4f}'.format(loss) for loss in best_va_loss_all])
print('All n_est:', best_n_est_all)
