import argparse
import os

import numpy as np

import util
from fea_names import FEA_NAMES_APAMI as FEA_NAMES

MAX_LEN = 5
np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('--n_fold_train', type=int, default=5)
parser.add_argument('--aggregate', '-agg', action='store_true')
args = parser.parse_args()

imp_all = []
for fold in range(args.n_fold_train):
    print('Getting fold', fold)
    model = util.load_pkl(os.path.join(args.model_dir, 'model_{}.pkl'.format(fold)))
    fea_dim = model.feature_importances_.shape[0]
    imp_raw = model.get_booster().get_score(importance_type='gain')
    imp = np.array([imp_raw.get('f'+str(d), 0) for d in range(fea_dim)]).astype('float32')
    imp /= np.sum(imp)
    imp_all.append(imp)
imp_all = np.array(imp_all)
print('Feature shape:', imp_all.shape)

if args.aggregate:
    imp_all = np.reshape(imp_all, (imp_all.shape[0], MAX_LEN, -1))
    imp_all = np.sum(imp_all, axis=1)
    print('Time x feature shape:', imp_all.shape)

if args.aggregate:
    xgb_fea_names = np.array(FEA_NAMES + ['is_pad'])
else:
    xgb_fea_names = np.array([['v-last-{}_{}'.format(v, n) for n in FEA_NAMES] for v in range(MAX_LEN, 0, -1)])
    xgb_fea_is_pad = np.array([['is_pad_last-{}'.format(v)] for v in range(MAX_LEN, 0, -1)])
    xgb_fea_names = np.concatenate((xgb_fea_names, xgb_fea_is_pad), axis=-1)
    xgb_fea_names = np.reshape(xgb_fea_names, -1)
print('Shape:', imp_all.shape, xgb_fea_names.shape)

for idx, name in enumerate(xgb_fea_names):
    print('{}: {}\t{}\t{}\t{}\t{}\t{}'.format(
        idx, name, imp_all[0, idx], imp_all[1, idx], imp_all[2, idx], imp_all[3, idx], imp_all[4, idx]
    ))
