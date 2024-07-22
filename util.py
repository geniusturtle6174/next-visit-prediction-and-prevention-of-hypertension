import pickle

import numpy as np
import pickle5
import torch
import torch.nn.utils.rnn as rnn_utils
import yaml
from torch.utils.data import Dataset

import patch_value


def save_pkl(path, pkl):
    with open(path, 'wb') as handle:   
        pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except:
            return pickle5.load(handle)


def write_cfg(path, cfg):
    with open(path, 'w') as fout:
        yaml.dump(cfg, fout, default_flow_style=False)


def load_cfg(path):
    with open(path, 'r') as fin:
        return yaml.safe_load(fin)


def print_and_write_file(fout, cnt, fout_end='\n'):
    print(cnt)
    if fout is not None:
        fout.write(cnt  + fout_end)
        fout.flush()


class Data2Torch(Dataset):
    def __init__(self, data):
        '''
        data: dict of np.array, should have key 'fea'
        '''
        self.data = {k: v for k, v in data.items()}

    def __getitem__(self, index):
        return {k: torch.from_numpy(v[index]).float() for k, v in self.data.items()}

    def __len__(self):
        return len(self.data['fea'])


def collate_fn(data):
    data.sort(key=lambda x: x['fea'].shape[0], reverse=True)
    seq_lens = [d['fea'].shape[0] for d in data]
    data = {
        'fea': rnn_utils.pad_sequence([d['fea'] for d in data], batch_first=True),
        'ans': torch.stack([d['ans'] for d in data]),
    }
    return data, seq_lens


def collate_fn_mtm(data):
    data.sort(key=lambda x: x['fea'].shape[0], reverse=True)
    seq_lens = [d['fea'].shape[0] for d in data]
    data = {
        'fea': rnn_utils.pad_sequence([d['fea'] for d in data], batch_first=True),
        'ans': rnn_utils.pad_sequence([d['ans'] for d in data], batch_first=True),
    }
    return data, seq_lens


def collate_fn_test(data):
    data.sort(key=lambda x: x['fea'].shape[0], reverse=True)
    seq_lens = [d['fea'].shape[0] for d in data]
    data = {
        'fea': rnn_utils.pad_sequence([d['fea'] for d in data], batch_first=True),
        'ans': torch.stack([d['ans'] for d in data]),
        'idx': torch.stack([d['idx'] for d in data]),
        'sep': torch.stack([d['sep'] if 'sep' in d else d['idx'] for d in data]),
    }
    return data, seq_lens


def get_conf_mat(recog, ans):
    conf_mat = np.zeros((2, 2))
    for r, a in zip(recog, ans):
        conf_mat[a, r] += 1
    return conf_mat


def get_conf_mat_4d(recog, ans):
    '''
    Shapes:
        recog: (n, 2)
        ans: (n, 2)
    '''
    conf_mat = np.zeros((4, 4))
    counter = 0
    for r, a in zip(recog, ans):
        i = a[0] * 2 + a[1]
        j = r[0] * 2 + r[1]
        conf_mat[i, j] += 1
        counter += 1
    return conf_mat


def parse_float(in_str, val_of_text=-9999, val_of_empty=-99999):
    if in_str != '':
        try:
            return float(in_str)
        except:
            return val_of_text
    return val_of_empty


def dt_to_float_ym(dt):
    year = dt // 100
    month = dt % 100
    return year + (month - 0.5) / 12
