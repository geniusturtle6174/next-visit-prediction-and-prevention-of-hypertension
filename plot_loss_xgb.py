import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.directory"] = '.'

DIR_1 = 'models/240711_xgb'
DIR_2 = 'models/240711_lgbm'
DIR_3 = 'models/240712_rf'

# plt.subplot(3, 1, 1)
plt.subplot(1, 3, 1)
x = np.arange(1, 1001)
for i in range(5):
    va_loss = np.load(os.path.join(DIR_1, 'va_loss_{}.npy'.format(i)))
    plt.plot(x, va_loss, '--', label='Fold {} Validation'.format(i+1))
    min_idx = np.argmin(va_loss)
    plt.plot(x[min_idx], va_loss[min_idx], 'o')
for i in range(5):
    tr_loss = np.load(os.path.join(DIR_1, 'tr_loss_{}.npy'.format(i)))
    plt.plot(x, tr_loss, label='Fold {} Training'.format(i+1))
plt.title('XGBoost')
plt.xlabel('Number of Estimators')
plt.ylabel('Loss')
plt.ylim(0.24, 0.7)

# plt.subplot(3, 1, 2)
plt.subplot(1, 3, 2)
x = np.arange(1, 1001)
for i in range(5):
    va_loss = np.load(os.path.join(DIR_2, 'va_loss_{}.npy'.format(i)))
    plt.plot(x, va_loss, '--', label='Fold {} Validation'.format(i+1))
    min_idx = np.argmin(va_loss)
    plt.plot(x[min_idx], va_loss[min_idx], 'o')
for i in range(5):
    tr_loss = np.load(os.path.join(DIR_2, 'tr_loss_{}.npy'.format(i)))
    plt.plot(x, tr_loss, label='Fold {} Training'.format(i+1))
plt.title('LightGBM')
plt.xlabel('Number of Estimators')
plt.ylim(0.24, 0.7)
plt.legend()

plt.subplot(1, 3, 3)
x = [600, 650, 700, 750, 800, 850, 900, 950, 1000]
for i in range(5):
    va_loss = np.load(os.path.join(DIR_3, 'va_loss_{}.npy'.format(i)))
    plt.plot(x, va_loss, '--')
    min_idx = np.argmin(va_loss)
    plt.plot(x[min_idx], va_loss[min_idx], 'o')
for i in range(5):
    tr_loss = np.load(os.path.join(DIR_3, 'tr_loss_{}.npy'.format(i)))
    plt.plot(x, tr_loss)
plt.title('Random Forest')
plt.xlabel('Number of Estimators')
plt.ylim(0.24, 0.7)

plt.show()
