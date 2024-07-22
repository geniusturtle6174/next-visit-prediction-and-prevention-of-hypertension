import numpy as np

import util

chid_to_error_type_model_a = util.load_pkl('results/220518_1_f/chid_to_error_type.pkl')
chid_to_error_type_model_b = util.load_pkl('results/220518_x3/chid_to_error_type.pkl')

error_type_mat = np.zeros((4, 4))
for chid in chid_to_error_type_model_a.keys():
    if chid not in chid_to_error_type_model_b:
        print('Chid {} not found in another pkl'.format(chid))
    error_type_mat[chid_to_error_type_model_a[chid], chid_to_error_type_model_b[chid]] += 1
diag_ratio = np.sum(error_type_mat[np.arange(4), np.arange(4)]) / np.sum(error_type_mat)

print(error_type_mat)
print('Off-diag ratio: {:.2f}%'.format((1-diag_ratio) * 100))
