import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel, TemporalAttentionLayer,
)
from my_model import SelfAttention


class Graph_Constructor(nn.Module):
    def __init__(self, nnodes, dim, alpha=3, static_feat=None):
        super(Graph_Constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.nnodes = nnodes
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self):
        idx = torch.arange(self.nnodes).cuda()
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).cuda()
        mask.fill_(float('0'))
        top_values, top_indices = torch.topk(torch.abs(adj).view(-1), math.ceil(0.3 * self.nnodes * self.nnodes))
        mask.view(-1)[top_indices] = 1
        adj = adj * mask + torch.eye(adj.size(0)).cuda()

        return adj


class GCN(nn.Module):
    def __init__(self, c_in, c_out, hid_dim, gdep=2, dropout=0.2, alpha=0.2):
        super(GCN, self).__init__()
        self.lin1 = nn.Linear(c_in * (gdep + 1), hid_dim)
        self.recon = nn.Sequential(nn.Linear(hid_dim, 32), nn.LeakyReLU(),
                                   nn.Linear(32, c_out), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * h + \
                (1 - self.alpha) * torch.bmm(h, a.unsqueeze(0).expand(h.size(0), *a.size()))
            out.append(h)
        h1 = torch.relu(self.lin1(torch.cat(out, dim=2)))
        rec = self.recon(h1)
        return h1, rec


class MTAD_Transform(nn.Module):

    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            gcn_hd_dim=64,
            gru_hid_dim=64,
            dropout=0.2,
            alpha=0.2
    ):
        super(MTAD_Transform, self).__init__()
        self.graph_constructor = Graph_Constructor(n_features, 16)
        self.gcn = GCN(n_features, n_features, gcn_hd_dim)
        self.temporal_feature = TemporalAttentionLayer(n_features, window_size, dropout, alpha, None, True)
        self.gru = GRULayer(n_features+gcn_hd_dim, gru_hid_dim, 1, dropout)
        self.forcast = nn.Sequential(nn.Linear(gru_hid_dim, 32), nn.ReLU(),
                                     nn.Linear(32, out_dim), nn.Sigmoid())
        # self.forcast = Forcast_Layer(n_features + gcn_hd_dim, n_features, window_size)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        adj = self.graph_constructor()
        _, rec = self.gcn(x, adj)
        x_a = x + (rec - x)**2
        hf, _ = self.gcn(x_a, adj)
        ht = self.temporal_feature(x_a)
        _, h = self.gru(torch.cat((ht, hf), 2))
        # forcast = self.forcast(torch.cat((ht, hf), 2)[:, -1, :])
        forcast = self.forcast(h)
        return rec, forcast.unsqueeze(1)

class Forcast_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, window_size, hd_dim=128):
        super(Forcast_Layer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim * window_size, hd_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hd_dim, out_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x