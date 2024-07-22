import os
import json
import time
import argparse
import warnings
from shutil import copyfile

import torch
import numpy as np
from torch.autograd import Variable

import util, model_nvp, fea_util
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('output_result_dir', help='Results directory name')
parser.add_argument('--model_postfix', default='_bestVa', choices=('_bestVa', '_bestF'))
parser.add_argument('--n_fold_train', type=int, default=5)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--max_year_diff', type=float, default=-1, help='use non-neg value to overwrite config')
parser.add_argument('--max_len', type=int, default=-1, help='use non-neg value to overwrite config')
args = parser.parse_args()

nn_param = util.load_cfg(os.path.join(args.model_dir, 'config.yml'))

if 'many_to_many' not in nn_param:
    nn_param['many_to_many'] = False
if 'DELTA' not in nn_param['fea_cfg']:
    nn_param['fea_cfg']['DELTA'] = False
if 'ALLOW_SHORTER' not in nn_param['fea_cfg']:
    nn_param['fea_cfg']['ALLOW_SHORTER'] = True

if args.max_year_diff >= 0:
    nn_param['fea_cfg']['MAX_YEAR_DIFFERENCE'] = args.max_year_diff
if args.max_len >= 0:
    nn_param['fea_cfg']['MAX_NUM_FOR_ONE_CUSTOMER'] = args.max_len

if not os.path.exists(args.output_result_dir):
    os.makedirs(args.output_result_dir, 0o755)
    print('Results will be saved in {}'.format(args.output_result_dir))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.output_result_dir))

print('Loading network and copying P/R curve...')
networks = {}
for fold in range(args.n_fold_train):
    save_dic = torch.load(os.path.join(args.model_dir, 'model_{}{}'.format(fold, args.model_postfix)))
    if nn_param['many_to_many']:
        networks[fold] = model_nvp.TRANS_BILSTM_MTM(is_a_classifier=nn_param['run_classification_prob'])
    else:
        networks[fold] = model_nvp.TRANS_BILSTM_FLATTEN(num_out=1, is_a_classifier=nn_param['run_classification_prob'])
    networks[fold].load_state_dict(save_dic)
    networks[fold].eval()
    networks[fold].to(device)
    copyfile(
        os.path.join(args.model_dir, 'train_report_{}.txt'.format(fold)),
        os.path.join(args.output_result_dir, 'train_report_{}.txt'.format(fold))
    )
    if args.model_postfix == '_bestVa':
        bp_pr_curves_file = os.path.join(args.model_dir, 'bp_pr_curves_{}.npy'.format(fold))
    elif args.model_postfix == '_bestF':
        bp_pr_curves_file = os.path.join(args.model_dir, 'bp_pr_curves_f-score_{}.npy'.format(fold))
    if os.path.exists(bp_pr_curves_file):
        copyfile(
            bp_pr_curves_file,
            os.path.join(args.output_result_dir, 'bp_pr_curves_{}.npy'.format(fold))
        )

print('Loading chid_to_fea...')
tic = time.time()
chid_to_fea = util.load_pkl(os.path.join(nn_param['fea_dir'], 'chid_to_fea.pkl'))
toc = time.time()
print('Loading finished, time elapsed (s):', toc - tic)

chid_to_idx = {c: i for i, c in enumerate(chid_to_fea)}
idx_to_chid = {i: c for c, i in chid_to_idx.items()}
util.save_pkl(os.path.join(args.output_result_dir, 'idx_to_chid.pkl'), idx_to_chid)

for i, c in enumerate(chid_to_fea):
    if i % 1000 == 0:
        print('Feature check for {}/{}'.format(i, len(chid_to_fea)))
    fea = chid_to_fea[c]['fea']
    ans = chid_to_fea[c]['ans']
    chid_to_fea[c]['use'] = True
    # Skip no ans
    if np.min(ans) <= 0:
        chid_to_fea[c]['use'] = False
        continue
    # Skip no bp
    fea_bp_min = np.min(fea, axis=0)
    if fea_bp_min[-3] < 0 or fea_bp_min[-2] < 0:
        chid_to_fea[c]['use'] = False
        continue
    # Personal patch
    fea = patch_for_one_person(fea)
    # Answer and pkl
    raw_ans = None
    if nn_param['run_classification_prob']:
        raw_ans = np.copy(ans)
        ans = [1 if ans[0] >= nn_param['hyp_ths'][0] or ans[1] >= nn_param['hyp_ths'][1] else 0]
    chid_to_fea[c]['fea'] = fea
    chid_to_fea[c]['ans'] = ans
    chid_to_fea[c]['raw_ans'] = raw_ans

print('Test data num:', len(chid_to_fea))
util.save_pkl(os.path.join(args.output_result_dir, 'chid_to_fea.pkl'), chid_to_fea)

tic = time.time()
for fold in range(args.n_fold_train):
    idx_all = []
    ans_all = []
    recog_all = []
    idx_batch = []
    fea_batch = []
    ans_batch = []
    for i, c in enumerate(chid_to_fea):
        if i % 100000 == 0:
            print('Fold {}: processing customer {}/{}...'.format(fold, i, len(chid_to_fea)))
            toc = time.time()
            print('\tTime elapsed (s):', toc - tic)
        if chid_to_fea[c]['use'] == False:
            continue
        fea = chid_to_fea[c]['fea']
        ans = chid_to_fea[c]['ans']
        idx_batch.append(chid_to_idx[c])
        fea_batch.append(fea)
        ans_batch.append(ans)
        if len(fea_batch) == args.test_batch_size:
            data_loader = torch.utils.data.DataLoader(
                util.Data2Torch({
                    'fea': fea_batch,
                    'ans': np.array(ans_batch),
                    'idx': np.array(idx_batch)[:, np.newaxis],
                }),
                batch_size=args.test_batch_size,
                collate_fn=util.collate_fn_test,
            )
            for data, seq_len in data_loader:
                with torch.no_grad():
                    pred = networks[fold](
                        Variable(data['fea'].to(device)), seq_len,
                    ).detach().cpu().numpy()
                if nn_param['many_to_many']:
                    pred = pred[np.arange(pred.shape[0]), np.array(seq_len, dtype=int)-1, :]
                idx_all.append(data['idx'].numpy())
                ans_all.append(data['ans'].numpy())
                recog_all.append(pred)
            idx_batch = []
            fea_batch = []
            ans_batch = []
    if len(fea_batch) > 0:
        data_loader = torch.utils.data.DataLoader(
            util.Data2Torch({
                'fea': fea_batch,
                'ans': np.array(ans_batch),
                'idx': np.array(idx_batch)[:, np.newaxis],
            }),
            batch_size=len(fea_batch),
            collate_fn=util.collate_fn_test,
        )
        for data, seq_len in data_loader:
            with torch.no_grad():
                pred = networks[fold](
                    Variable(data['fea'].to(device)), seq_len,
                ).detach().cpu().numpy()
            if nn_param['many_to_many']:
                pred = pred[np.arange(pred.shape[0]), np.array(seq_len, dtype=int)-1, :]
            idx_all.append(data['idx'].numpy())
            ans_all.append(data['ans'].numpy())
            recog_all.append(pred)
    print(np.vstack(idx_all).shape, np.vstack(ans_all).shape, np.vstack(recog_all).shape)
    result_all = np.hstack([
        np.vstack(idx_all),
        np.vstack(ans_all),
        np.vstack(recog_all),
    ])
    print('Fold {} finished, shape: {}'.format(fold, result_all.shape))
    np.save(os.path.join(args.output_result_dir, 'result_{}.npy'.format(fold)), result_all)
    print('Results saved at:', args.output_result_dir)
