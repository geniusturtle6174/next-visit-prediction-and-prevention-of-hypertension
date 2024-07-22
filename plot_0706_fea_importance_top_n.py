import matplotlib.pyplot as plt
import numpy as np

naive_p = 71.98
naive_r = 69.35
naive_f = 70.64
naive_a = 80.59

all_fea_p = 70.11
all_fea_r = 77.83
all_fea_f = 73.77
all_fea_a = 81.37

fea_top_n = [5, 10, 15, 20, 25, 30]
fea_top_n_p = [68.20, 70.53, 70.25, 69.88, 69.29, 70.09]
fea_top_n_r = [79.11, 76.82, 77.13, 77.74, 78.57, 77.69]
fea_top_n_f = [73.25, 73.54, 73.53, 73.60, 73.64, 73.70]
fea_top_n_a = [80.55, 81.39, 81.30, 81.23, 81.06, 81.33]

plt.subplot(2, 2, 1)
plt.title('Precision')
plt.plot([0, fea_top_n[-1]], [naive_p, naive_p], label='Naive')
plt.plot([0, fea_top_n[-1]], [all_fea_p, all_fea_p], label='All features')
plt.plot(fea_top_n, fea_top_n_p, '.-', label='Top-n')
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Recall')
plt.plot([0, fea_top_n[-1]], [naive_r, naive_r], label='Naive')
plt.plot([0, fea_top_n[-1]], [all_fea_r, all_fea_r], label='All features')
plt.plot(fea_top_n, fea_top_n_r, '.-', label='Top-n')

plt.subplot(2, 2, 3)
plt.title('F-score')
plt.plot([0, fea_top_n[-1]], [naive_f, naive_f], label='Naive')
plt.plot([0, fea_top_n[-1]], [all_fea_f, all_fea_f], label='All features')
plt.plot(fea_top_n, fea_top_n_f, '.-', label='Top-n')
plt.xlabel('n')

plt.subplot(2, 2, 4)
plt.title('Accuracy')
plt.plot([0, fea_top_n[-1]], [naive_a, naive_a], label='Naive')
plt.plot([0, fea_top_n[-1]], [all_fea_a, all_fea_a], label='All features')
plt.plot(fea_top_n, fea_top_n_a, '.-', label='Top-n')
plt.xlabel('n')

plt.show()
