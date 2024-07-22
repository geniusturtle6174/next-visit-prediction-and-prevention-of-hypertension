import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn, optim
from torch.autograd import Variable

import loss
import model_nvp
import util
from patch_value import patch_for_one_person

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('fea_dir')
parser.add_argument('save_model_dir_name')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--va_not_imp_limit', '-va', type=int, default=100)
parser.add_argument('--n_fold', type=int, default=5)
parser.add_argument('--run_single_fold_only', type=int, default=-1)
parser.add_argument('--run_classification_prob', '-classify', action='store_true')
parser.add_argument('--hyp_ths', type=int, nargs=2, default=[130, 80])
args = parser.parse_args()

fea_cfg = util.load_cfg(os.path.join(args.fea_dir, 'config.yml'))

fea_all = []
ans_all = []
num_uq_lens = 0
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
    fea_bp_min = np.min(fea_one_len, axis=1)[:, -3:-1]
    fea_bp_min = np.min(fea_bp_min, axis=1)
    keep_idx = np.where(fea_bp_min>0)[0]
    fea_one_len = fea_one_len[keep_idx]
    ans_one_len = ans_one_len[keep_idx]
    print('\tFinished, shapes:', fea_one_len.shape, ans_one_len.shape)
    print('\tPatching values...')
    for n in range(fea_one_len.shape[0]):
        fea_one_len[n] = patch_for_one_person(fea_one_len[n])
    print('\tnp array to list...')
    for fea, ans in zip(fea_one_len, ans_one_len):
        fea_all.append(fea)
        ans_all.append(ans)
    num_uq_lens += 1

del fea_one_len, ans_one_len
fea_all = np.array(fea_all, dtype=object if num_uq_lens > 1 else 'float32')
ans_all = np.array(ans_all)
print('Overall shapes:', fea_all.shape, ans_all.shape)

# --- Setup network
nn_param = {
    'batch': 256,
    'optm_params': {
        'lr': 0.001
    },
    'run_classification_prob': args.run_classification_prob,
    'fea_dir': args.fea_dir,
    'fea_cfg': fea_cfg,
    'hyp_ths': args.hyp_ths,
    'many_to_many': False,
}

print('Setting network')
networks = {}
optimizers = {}
schedulers = {}
loss_func = nn.MSELoss()
for f in range(args.n_fold):
    networks[f] = model_nvp.TRANS_BILSTM_FLATTEN(num_out=1, is_a_classifier=nn_param['run_classification_prob'])
    networks[f].to(device)
    optimizers[f] = optim.Adam(list(networks[f].parameters()), lr=nn_param['optm_params']['lr'])
    schedulers[f] = torch.optim.lr_scheduler.StepLR(optimizers[f], step_size=1, gamma=0.95)

if nn_param['run_classification_prob']:
    raw_ans_all = np.copy(ans_all)
    sbp_high_idx = np.where(raw_ans_all[:, 0] >= nn_param['hyp_ths'][0])[0]
    dbp_high_idx = np.where(raw_ans_all[:, 1] >= nn_param['hyp_ths'][1])[0]
    ans_all = np.zeros((ans_all.shape[0], 1))
    ans_all[sbp_high_idx] = 1
    ans_all[dbp_high_idx] = 1
    neg_ratio = np.mean(ans_all==0)
    pos_ratio = np.mean(ans_all==1)
    loss_func = nn.BCELoss()
    print('Use classification, ths: {}/{}, neg ratio: {:.4f}%%, pos ratio: {:.4f}%%'.format(
        nn_param['hyp_ths'][0],
        nn_param['hyp_ths'][1],
        100 * neg_ratio,
        100 * pos_ratio,
    ))

# --- Write config
if not os.path.exists(args.save_model_dir_name):
    os.makedirs(args.save_model_dir_name, 0o755)
    print('Model will be saved in {}'.format(args.save_model_dir_name))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.save_model_dir_name))
util.write_cfg(os.path.join(args.save_model_dir_name, 'config.yml'), nn_param)

data_num = fea_all.shape[0]
for fold in range(args.n_fold):

    if args.run_single_fold_only >= 0 and args.run_single_fold_only != fold:
        continue

    best_va_loss = 9999
    best_f_score = 0

    valid_idx = np.where(np.arange(data_num)%args.n_fold==fold)[0]
    train_idx = np.where(np.arange(data_num)%args.n_fold!=fold)[0]

    print('Fold {}, train num {}, test num {}'.format(
        fold, len(train_idx), len(valid_idx),
    ))

    data_loader_train = torch.utils.data.DataLoader(
        util.Data2Torch({
            'fea': fea_all[train_idx],
            'ans': ans_all[train_idx],
        }),
        shuffle=True,
        batch_size=nn_param['batch'],
        collate_fn=util.collate_fn,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        util.Data2Torch({
            'fea': fea_all[valid_idx],
            'ans': ans_all[valid_idx],
        }),
        batch_size=nn_param['batch'],
        collate_fn=util.collate_fn,
    )

    va_not_imporved_continue_count = 0
    totalTime = 0
    fout = open(os.path.join(args.save_model_dir_name, 'train_report_{}.txt'.format(fold)), 'w')
    for epoch in range(args.epoch):
        util.print_and_write_file(fout, 'epoch {}/{}...'.format(epoch + 1, args.epoch))
        tic = time.time()
        # --- Batch training
        networks[fold].train()
        training_loss = 0
        n_batch = 0
        optimizers[fold].zero_grad()
        for idx, (data, seq_len) in enumerate(data_loader_train):
            pred = networks[fold](Variable(data['fea'].to(device)), seq_len)
            ans = Variable(data['ans'].to(device))
            if nn_param['run_classification_prob']:
                # ans = ans[:, 0].type(torch.long)
                pass
            loss = loss_func(pred, ans)
            optimizers[fold].zero_grad()
            loss.backward()
            optimizers[fold].step()
            training_loss += loss.data
            n_batch += 1
        # --- Training loss
        training_loss_avg = training_loss / n_batch
        util.print_and_write_file(
            fout, '\tTraining loss (avg over batch): {}, {}, {}'.format(
                training_loss_avg, training_loss, n_batch
            )
        )
        # --- Batch validation
        networks[fold].eval()
        va_loss = 0
        n_batch = 0
        va_pred_all = []
        va_ans_all = []
        for idx, (data, seq_len) in enumerate(data_loader_valid):
            ans = Variable(data['ans'].to(device)).float()
            if nn_param['run_classification_prob']:
                # ans = ans[:, 0].type(torch.long)
                pass
            with torch.no_grad():
                pred = networks[fold](Variable(data['fea'].to(device)), seq_len)
                loss = loss_func(pred, ans)
            va_pred_all.append(pred.detach().cpu().numpy())
            va_ans_all.append(ans.detach().cpu().numpy())
            va_loss += loss.data
            n_batch += 1
        # --- Validation loss
        va_loss_avg = va_loss / n_batch
        util.print_and_write_file(
            fout, '\tValidation loss (avg over batch): {}, {}, {}'.format(
                va_loss_avg, va_loss, n_batch
            )
        )
        # --- Stack pred results and save by F-score
        if nn_param['run_classification_prob']:
            beta = 1
            va_pred_all = np.vstack(va_pred_all)
            va_ans_all = np.vstack(va_ans_all)
            bp_p, bp_r, bp_ths = precision_recall_curve(va_ans_all, va_pred_all)
            bp_f = (1 + beta ** 2) * (bp_p+1e-6) * (bp_r+1e-6) / (beta ** 2 * (bp_p+1e-6) + (bp_r+1e-6))
            max_f = np.max(bp_f)
            util.print_and_write_file(fout, '\tF-score: {}'.format(max_f))
            if max_f > best_f_score:
                best_f_score = max_f
                # --- Save model
                util.print_and_write_file(fout, '\tWill save best F-score model')
                torch.save(
                    networks[fold].state_dict(),
                    os.path.join(args.save_model_dir_name, 'model_{}_bestF'.format(fold))
                )
                # --- Save PR curves
                bp_pr_curves = np.vstack([bp_p[:-1], bp_r[:-1], bp_ths])
                print('\tSaving P/R curves for best F-score, shapes:', bp_pr_curves.shape) # Shape: (p/r/ths, n_point)
                np.save(os.path.join(args.save_model_dir_name, 'bp_pr_curves_f-score_{}.npy'.format(fold)), bp_pr_curves)
        # --- Save and early stop by VA loss
        if va_loss_avg < best_va_loss:
            best_va_loss = va_loss_avg
            va_not_imporved_continue_count = 0
            util.print_and_write_file(fout, '\tWill save bestVa model')
            torch.save(
                networks[fold].state_dict(),
                os.path.join(args.save_model_dir_name, 'model_{}_bestVa'.format(fold))
            )
            if nn_param['run_classification_prob']:
                bp_pr_curves = np.vstack([bp_p[:-1], bp_r[:-1], bp_ths])
                print('\tSaving P/R curves for best VA loss, shapes:', bp_pr_curves.shape) # Shape: (p/r/ths, n_point)
                np.save(os.path.join(args.save_model_dir_name, 'bp_pr_curves_{}.npy'.format(fold)), bp_pr_curves)
        else:
            va_not_imporved_continue_count += 1
            util.print_and_write_file(fout, '\tva_not_imporved_continue_count: {}'.format(va_not_imporved_continue_count))
            if va_not_imporved_continue_count >= args.va_not_imp_limit:
                break
        util.print_and_write_file(fout, '\tLearning rate used for this epoch: {}'.format(schedulers[fold].get_last_lr()[0]))
        if schedulers[fold].get_last_lr()[0] >= 1e-4:
            schedulers[fold].step()
        # --- Time
        toc = time.time()
        totalTime += toc - tic
        util.print_and_write_file(fout, '\tTime: {:.3f} sec, estimated remaining: {:.3} hr'.format(
            toc - tic,
            1.0 * totalTime / (epoch + 1) * (args.epoch - (epoch + 1)) / 3600
        ))
        fout.flush()
    fout.close()
    # Save model
    torch.save(
        networks[fold].state_dict(),
        os.path.join(args.save_model_dir_name, 'model_{}_final'.format(fold))
    )
    print('Model saved in {}'.format(args.save_model_dir_name))
