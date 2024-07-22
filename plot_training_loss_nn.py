import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_dir')
args = parser.parse_args()

train_loss_all = []
valid_loss_all = []
best_va_before_not_imp_10_at = []
for fold in range(5):
    report_path = os.path.join(args.model_dir, 'train_report_{}.txt'.format(fold))
    with open(report_path, 'r') as fin:
        cnt = fin.read().splitlines()
    train_loss = []
    valid_loss = []
    for idx, line in enumerate(cnt):
        if line.startswith('\tTraining loss'):
            train_loss.append(float(line.split(': ')[1].split(',')[0]))
        elif line.startswith('\tValidation loss'):
            valid_loss.append(float(line.split(': ')[1].split(',')[0]))
        elif line.startswith('\tva_not_imporved_continue_count: 10') and len(best_va_before_not_imp_10_at) == fold:
            best_va_before_not_imp_10_at.append(idx//6-9)
    train_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)

train_loss_all = np.array(train_loss_all)
valid_loss_all = np.array(valid_loss_all)
print('Shapes:', train_loss_all.shape, valid_loss_all.shape, best_va_before_not_imp_10_at)

# for fold in range(5):
#     plt.subplot(2, 3, fold+1)
#     plt.plot(train_loss_all[fold], label='Tr')
#     plt.plot(valid_loss_all[fold], label='Va')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Fold {}'.format(fold))

for fold in range(5):
    min_va_idx = np.argmin(valid_loss_all[fold])
    early_va_idx = best_va_before_not_imp_10_at[fold]
    print('Fold {}, min_va at: {}, best_va_before_not_imp_10_at: {}'.format(
        fold, min_va_idx+1, early_va_idx
    ))
    plt.plot(train_loss_all[fold], label='Tr {}'.format(fold))
    plt.plot(valid_loss_all[fold], label='Va {}'.format(fold))
    plt.plot(min_va_idx, valid_loss_all[fold][min_va_idx], 'r.', ms=10)
    plt.plot(early_va_idx-1, valid_loss_all[fold][early_va_idx-1], 'bx', ms=7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
plt.legend()

plt.show()