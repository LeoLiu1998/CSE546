import torch
from math import sin, cos, sqrt
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import Linear, Dropout, Parameter


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], \
                         requires_grad=False).cuda()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.fc_q = Linear(d_model, d_model)
        self.fc_v = Linear(d_model, d_model)
        self.fc_k = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)
        self.out = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        k = self.fc_k(k).view(batch_size, -1, self.h, self.d_k)
        q = self.fc_q(q).view(batch_size, -1, self.h, self.d_k)
        v = self.fc_v(v).view(batch_size, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):

    score = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -1e9)
    score = F.softmax(score, dim=-1)

    if dropout is not None:
        score = dropout(score)

    output = torch.matmul(score, v)
    return output


class Norm(nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = Parameter(torch.ones(self.size))
        self.bias = Parameter(torch.zeros(self.size))
        self.epsilon = epsilon

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.epsilon) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.dropout = Dropout(dropout)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

