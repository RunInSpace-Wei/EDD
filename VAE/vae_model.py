import torch
from torch import nn
# from AE.ae_model import Encoder
import torch.nn.functional as F

from TranAD.dlutils import TransformerEncoderLayer
from modules import ConvLayer, FeatureAttentionLayer, TemporalAttentionLayer, GRULayer
from my_model import PositionalEncoding


class VAE_MODEL(nn.Module):
    def __init__(self, n_features, window_size, out_dim, h_dim=128, gru_hid_dim=160, z_dim=16, dropout=0.2):
        super(VAE_MODEL, self).__init__()
        self.window_size = window_size
        self.emb = nn.Embedding(n_features, 10)
        self.feature_extract = Encoder(n_features, window_size, gru_hid_dim=gru_hid_dim)
        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(gru_hid_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)

        # 解码器
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, out_dim)

        # 编别器
        self.disc = nn.Sequential(nn.Linear(z_dim, h_dim), nn.Dropout(dropout), nn.ReLU(),
                                  nn.Linear(h_dim, n_features), nn.Sigmoid())
        # self.disc = Discriminator(z_dim, 32)

    def forward(self, x_pos, x_neg=None):
        batch, window_size, n_features = x_pos.shape
        # weights = self.emb(torch.arange(x_pos.shape[2]).cuda())
        if x_neg is not None:
            # x_pos_f = self.feature_extract(x_pos, weights)[:, -1, :]
            # x_neg_f = self.feature_extract(x_neg, weights)[:, -1, :]
            x_pos_f = self.feature_extract(x_pos)
            x_neg_f = self.feature_extract(x_neg)

            mu_pos, log_var_pos = self.encode(x_pos_f)
            sampled_pos_z = self.reparameterization(mu_pos, log_var_pos)

            mu_neg, log_var_neg = self.encode(x_neg_f)
            sampled_neg_z = self.reparameterization(mu_neg.repeat(batch, 1), log_var_neg.repeat(batch, 1))

            x_pos_rec = self.decode(sampled_pos_z)
            # x_pos_rec = None

            x_disc_pos = self.disc(sampled_pos_z)
            x_disc_neg = self.disc(sampled_neg_z)
            # x_disc_neg = None

            return mu_pos, log_var_pos, x_pos_rec, x_disc_pos, x_disc_neg, mu_neg, log_var_neg
        else:
            # x_pos_f = self.feature_extract(x_pos, weights)[:, -1, :]
            x_pos_f = self.feature_extract(x_pos)

            mu_pos, log_var_pos = self.encode(x_pos_f)
            sampled_pos_z = self.reparameterization(mu_pos, log_var_pos)

            x_pos_rec = self.decode(sampled_pos_z)

            x_disc_pos = self.disc(sampled_pos_z)

            return mu_pos, log_var_pos, x_pos_rec, x_disc_pos, None

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = torch.tanh(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        z = z.repeat_interleave(self.window_size, dim=1).view(z.size(0), self.window_size, -1)
        h = F.relu(self.fc4(z))
        x_hat = torch.tanh(self.fc5(h))
        return x_hat

class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(Discriminator, self).__init__()
        self.att = nn.MultiheadAttention(in_dim, 16, batch_first=True)
        self.lin = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Dropout(dropout),  nn.Sigmoid())

    def forward(self, x):
        x = self.att(x, x, x, need_weights=False)[0]
        return self.lin(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(Encoder, self).__init__()
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3*n_features, gru_hid_dim, gru_n_layers, dropout)
        self.lstm = nn.LSTM(2*n_features, gru_hid_dim, gru_n_layers, batch_first=True)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        # h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat], dim=2)  # (b, n, 3k)
        #
        # _, h_end = self.gru(h_cat)
        _, (h_end, _) = self.lstm(h_cat)
        h_end = h_end.view(x.shape[0], -1)
        return h_end