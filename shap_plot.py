import os
import argparse

import numpy as np
import shap
from matplotlib.pyplot import subplots_adjust

import util
from fea_names import FEA_NAMES_APAMI as FEA_NAMES

MAX_LEN = 5
np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('--n_fold_train', type=int, default=5)
args = parser.parse_args()

tr_param = util.load_cfg(os.path.join(args.model_dir, 'config.yml'))

xgb_fea_names = np.array([['v-last-{}_{}'.format(v, n) for n in FEA_NAMES] for v in range(MAX_LEN, 0, -1)])
xgb_fea_is_pad = np.array([['is_pad_last-{}'.format(v)] for v in range(MAX_LEN, 0, -1)])
xgb_fea_names = np.concatenate((xgb_fea_names, xgb_fea_is_pad), axis=-1)
xgb_fea_names = np.reshape(xgb_fea_names, -1)
print('Shape:', xgb_fea_names.shape)
# print(xgb_fea_names[0:267][:5])
# print(xgb_fea_names[267:267*2][:5])
# print(xgb_fea_names[267*2:267*3][:5])
# print(xgb_fea_names[267*3:267*4][:5])
# print(xgb_fea_names[267*4:267*5][:5])
# exit()

shap_values = []
for fold in range(args.n_fold_train):
    print('Loading SHAP values {}...'.format(fold))
    shap_value = util.load_pkl(os.path.join(args.model_dir, 'shap_{}.pkl'.format(fold)))
    shap_value.feature_names = xgb_fea_names
    shap_values.append(shap_value)
    print(shap_value.shape, shap_value.values.shape)
    # shap.plots.beeswarm(shap_values[fold], max_display=20)
    # break

shap_value_fold_avg = sum(shap_values)
shap_value_fold_avg /= args.n_fold_train
# shap.plots.beeswarm(shap_value_fold_avg, max_display=20)

n_visit = 5
raw_n_dim = shap_value_fold_avg.shape[1]
n_dim = int(raw_n_dim / n_visit)
shap_value_visit_avg = sum([shap_value_fold_avg[:, n_dim*i:n_dim*(i+1)] for i in range(n_visit)]) / n_visit
shap_value_visit_avg.feature_names = FEA_NAMES + ['is_pad']
shap.plots.beeswarm(shap_value_visit_avg, max_display=20)
# shap_value_visit_avg = sum([shap_value_fold_avg[:, np.arange(0, raw_n_dim, n_dim)+i] for i in range(n_dim)]) / n_dim
# shap_value_visit_avg.feature_names = ['5', '4', '3', '2', '1']
# shap.plots.beeswarm(shap_value_visit_avg)
