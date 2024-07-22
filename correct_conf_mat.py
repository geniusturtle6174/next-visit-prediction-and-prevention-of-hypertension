import numpy as np

conf_mat = np.array([[42438, 8781], [5745, 20248]])
conf_mat_r = 100 * conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
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
print('ACC: {:.2f}'.format(100 * (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(conf_mat)))
