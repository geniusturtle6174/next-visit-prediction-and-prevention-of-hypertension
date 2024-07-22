import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x


class TRANS_BILSTM_FLATTEN(nn.Module):
    def __init__(self, num_out=2, is_a_classifier=False, max_seq_len=5):
        super(TRANS_BILSTM_FLATTEN, self).__init__()
        DIM_FEA = 274
        DIM_PRE = 64
        DIM_TRN = 512
        TRN_N_HEAD = 8
        DIM_LSTM_HIDDEN = 64
        DIM_POST = 128
        self.max_seq_len = max_seq_len
        self.pre = nn.Sequential(
            nn.Linear(in_features=DIM_FEA, out_features=DIM_PRE),
        )
        self.positional_encoding = PositionalEncoding(DIM_PRE, self.max_seq_len)
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=DIM_PRE, nhead=TRN_N_HEAD, dim_feedforward=DIM_TRN, batch_first=True
        )
        self.rnn = nn.LSTM(
            input_size=DIM_PRE,
            hidden_size=DIM_LSTM_HIDDEN,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        self.post = nn.Sequential(
            nn.Linear(
                in_features=DIM_LSTM_HIDDEN*2*self.max_seq_len,
                out_features=DIM_POST
            ),
            nn.BatchNorm1d(DIM_POST),
            nn.ReLU(),
            nn.Linear(in_features=DIM_POST, out_features=DIM_POST),
            nn.BatchNorm1d(DIM_POST),
            nn.ReLU(),
            nn.Linear(in_features=DIM_POST, out_features=num_out),
            nn.Sigmoid() if is_a_classifier else nn.ReLU(),
            # nn.Identity() if is_a_classifier else nn.ReLU(),
        )

    def get_src_mask(self):
        mask = Variable(torch.zeros(self.max_seq_len, self.max_seq_len)).type(torch.bool).to(DEVICE)
        # allow past + curr
        # for i in range(self.max_seq_len):
        #     mask[i, i+1:] = True
        # allow past + curr + future 1
        # for i in range(self.max_seq_len):
        #     mask[i, i+2:] = True
        # allow curr + neighbor 2
        for i in range(self.max_seq_len):
            if i - 1 > 0:
                mask[i, :i-1] = True
            mask[i, i+2:] = True
        print(mask)
        return mask

    def get_src_key_padding_mask(self, batch_size, seq_len):
        mask = Variable(torch.zeros(batch_size, self.max_seq_len)).type(torch.bool).to(DEVICE)
        for i in range(batch_size):
            mask[i, seq_len[i]:] = True
        return mask

    def forward(self, x, seq_len):
        # Pre
        if x.shape[1] < self.max_seq_len:
            x = F.pad(x, (0, 0, 0, self.max_seq_len-x.shape[1]))
        # seq_len = np.array(seq_len)
        # x_left = torch.cat((x[:, :1, :], x[:, :-1, :]), dim=1)
        # x_right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
        # x_right[np.arange(x.shape[0]), seq_len-1, :] = x[np.arange(x.shape[0]), seq_len-1, :]
        # x = torch.cat((x_left, x, x_right), dim=-1)
        x = self.pre(x)
        # Transformer
        x = self.positional_encoding(x)
        x = self.trans_enc(
            x, src_key_padding_mask=self.get_src_key_padding_mask(x.shape[0], seq_len)
        )
        # RNN
        x = rnn_utils.pack_padded_sequence(x, lengths=seq_len, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        if x.shape[1] < self.max_seq_len:
            x = F.pad(x, (0, 0, 0, self.max_seq_len-x.shape[1]))
        # Flatten + POST
        h = torch.flatten(x, start_dim=1)
        h = self.post(h)
        # RETURN
        return h


class WTF(nn.Module):
    def __init__(self, num_out=2, is_a_classifier=False, max_seq_len=5):
        super(WTF, self).__init__()
        DIM_FEA = 250
        self.max_seq_len = max_seq_len
        self.m_1 = TRANS_BILSTM_FLATTEN(num_out=DIM_FEA)
        self.m_2 = TRANS_BILSTM_FLATTEN(num_out=num_out, is_a_classifier=is_a_classifier, max_seq_len=max_seq_len+1)

    def forward(self, x, seq_len):
        x_psudo_next = self.m_1(x, seq_len)
        if x.shape[1] < self.max_seq_len + 1:
            x = F.pad(x, (0, 0, 0, self.max_seq_len+1-x.shape[1]))
        x[torch.arange(x.shape[0]), seq_len, :] = x_psudo_next
        return self.m_2(x, [v + 1 for v in seq_len])


class CONV_FC(nn.Module):
    def __init__(self, num_out=2, is_a_classifier=False):
        super(CONV_FC, self).__init__()
        self.max_seq_len = 5
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (3, 250), stride=1, padding=(1, 0)), # (5, 250) to (5, 1)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 1), stride=1, padding=(1, 0)), # (5, 1) to (5, 1)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 1), stride=1, padding=0), # (5, 1) to (3, 1)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 1), stride=1, padding=(1, 0)), # (3, 1) to (3, 1)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 1), stride=1, padding=0), # (3, 1) to (1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_out),
            nn.Sigmoid() if is_a_classifier else nn.ReLU(),
        )

    def forward(self, x, dummy):
        if x.shape[1] < self.max_seq_len:
            x = F.pad(x, (0, 0, 0, self.max_seq_len-x.shape[1]))
        x = torch.unsqueeze(x, dim=1) # (n, time, freq) to (n, ch, time, freq)
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, num_out=2, is_a_classifier=False):
        super(MLP, self).__init__()
        DIM_FEA = 85
        DIM_HIDDEN = 32
        self.net = nn.Sequential(
            nn.Linear(in_features=DIM_FEA, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
            nn.Linear(in_features=DIM_HIDDEN, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
            nn.Linear(in_features=DIM_HIDDEN, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
            nn.Linear(in_features=DIM_HIDDEN, out_features=num_out),
            nn.Identity() if is_a_classifier else nn.SiLU(),
        )

    def forward(self, x, dummy):
        if x.ndim == 3 and x.shape[1] == 1:
            x = torch.squeeze(x, dim=1)
        else:
            raise ValueError('x.ndim should 3 and x.shape[1] should be 1, but got {} and {}'.format(
                x.ndim, x.shape[1]
            ))
        return self.net(x)
