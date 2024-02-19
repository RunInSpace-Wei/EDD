import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv,GCNConv
import torch.nn.functional as F
import scipy.sparse as sp

from modules import (
    ConvLayer, ReconstructionModel, Forecasting_Model, ReconstructionModel2, GRULayer,
)

class MTAD_MODEL(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
            self,
            n_features,
            window_size,
            heads,
            target_dim,
            kernel_size=7,
            feat_gat_embed_dim=None,
            use_gatv2=True,
            dropout=0.2,
            alpha=0.2,
            embed_dim=8
    ):
        super(MTAD_MODEL, self).__init__()
        self.n_features = n_features
        self.target_dim = target_dim
        self.conv = ConvLayer(n_features, kernel_size)
        self.pos_encoder = PositionalEncoding(n_features, dropout, window_size)
        self.temporal_feature = TemporalAttentionLayer(n_features, window_size, dropout, alpha, None, use_gatv2)
        self.embedding = nn.Embedding(n_features, embed_dim)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.forecasting_model = nn.Sequential(nn.Linear(window_size, 1), nn.Tanh())


    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = self.pos_encoder(x)
        x = self.conv(x)
        x = self.temporal_feature(x)
        x = self.feature_gat(x)

        weights = self.embedding(torch.arange(self.n_features).cuda())

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        x = torch.matmul(x, cos_ji_mat.T)
        # x = x + h / 2

        x = x.permute(0, 2, 1)
        predictions = self.forecasting_model(x)
        # recons = self.recon_model(x)
        return predictions.permute(0, 2, 1)


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.tanh(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.tanh(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.2)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x.permute(1, 0, 2)
        x = x + self.pe[pos:pos+x.size(0), :]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class MTAD_MODEL2(nn.Module):

    def __init__(
            self,
            n_features,
            window_size,
            heads,
            target_dim,
            kernel_size=7,
            feat_gat_embed_dim=None,
            use_gatv2=True,
            dropout=0.2,
            alpha=0.2,
            embed_dim=8
    ):
        super(MTAD_MODEL2, self).__init__()
        self.n_features = n_features
        self.target_dim = target_dim
        self.conv = ConvLayer(n_features, kernel_size)
        self.pos_encoder = PositionalEncoding(n_features, dropout, window_size)
        self.temporal_feature = TemporalAttentionLayer(n_features, window_size, dropout, alpha, None, use_gatv2)
        self.embedding = nn.Embedding(n_features, embed_dim)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.forecasting_model = nn.Sequential(nn.Linear(window_size, 1), nn.Tanh())


    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = self.pos_encoder(x)
        x = self.conv(x)
        x = self.temporal_feature(x)
        x = self.feature_gat(x)

        weights = self.embedding(torch.arange(self.n_features).cuda())

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        x = torch.matmul(x, cos_ji_mat.T)

        x = x.permute(0, 2, 1)
        predictions = self.forecasting_model(x)
        # recons = self.recon_model(x)
        return predictions.permute(0, 2, 1)

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

    def prune_tensor(self, A_pruned, n):
        """
        对邻接tensor矩阵A进行n次剪枝操作，返回剪枝后的邻接tensor矩阵
        """

        # 获取邻接矩阵的大小
        num_nodes = A_pruned.size(0)
        # 对邻接矩阵进行n次剪枝操作
        i = 0
        while i < n:
            # 获取邻接矩阵中最小值的行列索引
            min_idx = torch.argmin(A_pruned)
            min_row, min_col = min_idx // num_nodes, min_idx % num_nodes
            # 判断对应节点的度是否大于1，如果是则将该值置为0
            if torch.sum(A_pruned[min_row]) - A_pruned[min_row][min_col] > 0 \
                    or torch.sum(A_pruned[:, min_col]) - A_pruned[min_row][min_col] > 0:
                A_pruned[min_row][min_col] = 0
                i = i + 1
            else:
                A_pruned[min_row][min_col].fill_(torch.max(A_pruned))
        return A_pruned

    def edge_transform(self, adj_matrix):
        A = adj_matrix.detach().clone()
        edge_index_temp = sp.coo_matrix(A.cpu())
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index = torch.tensor(indices.astype(dtype=np.float64), requires_grad=True).cuda()  # 我们真正需要的coo形式

        return edge_index

    def forward(self):
        idx = torch.arange(self.nnodes).cuda()
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).cuda()
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(20, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask

        return self.edge_transform(adj)


class GraphLearning(nn.Module):
    def __init__(self, n_features, window_size, embed_dim=12):
        super(GraphLearning, self).__init__()
        hidden = math.ceil(0.6*window_size)

        self.embed = nn.Embedding(n_features, embed_dim)
        self.gat = GCNConv(window_size+embed_dim, window_size, dropout=0.2)
        self.mlp = nn.Sequential(nn.Linear(window_size, hidden), nn.Tanh(),
                                 nn.Linear(hidden, window_size), nn.Tanh())

        self.n_features = n_features

    def forward(self, x, batch_size):
        embeddings = self.embed(torch.arange(self.n_features).cuda())
        all_embeddings = embeddings.repeat(batch_size, 1)
        weights = embeddings.detach().clone()
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        cos_ji_mat = self.prune_tensor(cos_ji_mat, math.ceil(0.6*self.n_features*self.n_features))
        cos_ji_mat[cos_ji_mat != 0] = 1
        edges = self.edge_transform(cos_ji_mat)
        batch_edge_index = get_batch_edge_index(edges, batch_size, self.n_features)

        x = torch.cat((x, all_embeddings), 1)
        x = self.gat(x, batch_edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        return x, batch_edge_index

    def prune_tensor(self, A_pruned, n):
        # 获取邻接矩阵的大小
        num_nodes = A_pruned.size(0)
        # 对邻接矩阵进行n次剪枝操作
        i = 0
        while i < n:
            # 获取邻接矩阵中最小值的行列索引
            min_idx = torch.argmin(A_pruned)
            min_row, min_col = min_idx // num_nodes, min_idx % num_nodes
            # 判断对应节点的度是否大于1，如果是则将该值置为0
            if torch.sum(A_pruned[min_row]) - A_pruned[min_row][min_col] > 0 \
                    or torch.sum(A_pruned[:, min_col]) - A_pruned[min_row][min_col] > 0:
                A_pruned[min_row][min_col] = 0
                i = i + 1
            else:
                A_pruned[min_row][min_col].fill_(torch.max(A_pruned))

        return A_pruned

    def edge_transform(self, adj_matrix):
        A = adj_matrix.detach().clone()
        edge_index_temp = sp.coo_matrix(A.cpu())
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.tensor(indices).cuda()

        return edge_index


class GraphLearning2(nn.Module):
    def __init__(self, n_features, window_size, embed_dim=12):
        super(GraphLearning2, self).__init__()
        hidden = math.ceil(0.6*window_size)

        self.gat = GCNConv(window_size, window_size, dropout=0.2)
        self.mlp = nn.Sequential(nn.Linear(window_size, hidden), nn.Tanh(),
                                 nn.Linear(hidden, window_size), nn.Tanh())

        self.n_features = n_features

    def forward(self, x, batch_edge_index):
        x = self.gat(x, batch_edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        return x


class Feature_extractor(nn.Module):
    def __init__(self, n_features, window_size, use_gatv2=True, dropout=0.2, alpha=0.2,):
        super(Feature_extractor, self).__init__()
        self.temporal_feature = TemporalAttentionLayer(n_features, window_size, dropout, alpha, None, use_gatv2)
        self.feature_gat = GATConv(window_size, window_size, dropout=dropout, static=True)
        self.forecasting_model = nn.Sequential(nn.Linear(window_size, 1), nn.Tanh())
        self.window_size = window_size

    def forward(self, x, edges):
        x = self.temporal_feature(x)
        x = x.permute(0, 2, 1).contiguous().view(-1, self.window_size).contiguous()
        x = self.feature_gat(x, edges)
        x = self.forecasting_model(x)
        return x

class MTAD_MODEL3(nn.Module):
    def __init__(self, n_features, window_size):
        super(MTAD_MODEL3, self).__init__()
        self.graph_constructor = Graph_Constructor(n_features, 8)
        self.graph_learning = GraphLearning(n_features, window_size)
        self.anomaly_detect = Feature_extractor(n_features, window_size)

        self.n_features = n_features
        self.window_size = window_size

    def forward(self, x):
        # x(batch, window, n_features)
        x = x.permute(0, 2, 1).contiguous()
        batch_num, node_num, all_feature = x.shape
        x_batch = x.view(-1, all_feature).contiguous()

        edge_index = self.graph_constructor()
        batch_edge_index = get_batch_edge_index(edge_index, batch_num, node_num)
        # x_rec, batch_edge_index = self.graph_learning(x_batch, batch_num)


        # *****************
        x_rec = self.graph_learning(x, edge_index).permute(0, 2, 1)
        x = x_batch + x_rec
        x_pre = self.anomaly_detect(x.view(batch_num, node_num, all_feature).permute(0, 2, 1), batch_edge_index.clone().detach())
        return x_rec.view(batch_num, node_num, -1).contiguous().permute(0, 2, 1), \
            x_pre.view(batch_num, -1, node_num)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_num = org_edge_index.shape[1]
    org_edge_index = org_edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        org_edge_index[:, i*edge_num:(i+1)*edge_num] = org_edge_index[:, i*edge_num:(i+1)*edge_num] + i*node_num

    return org_edge_index



class MTAD_MODEL4(nn.Module):
    def __init__(self, n_features, window_size,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2):
        super(MTAD_MODEL4, self).__init__()
        self.embed = nn.Embedding(n_features, 12)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_feature = TemporalAttentionLayer(n_features, window_size, dropout, alpha, None, use_gatv2)
        self.forcast = Forecasting_Model(window_size, forecast_hid_dim, 1, forecast_n_layers, dropout)
        self.recon = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, n_features, recon_n_layers,dropout)
        self.w = nn.Parameter(torch.rand((n_features, n_features)))
        self.gru = GRULayer(2 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        self.n_features = n_features
        self.window_size = window_size

    def forward(self, x):
        # x(batch, window, n_features)
        # batch_num, all_feature, node_num = x.shape
        # embeddings = self.embed(torch.arange(self.n_features).cuda())
        x_t = self.temporal_feature(x)
        # x_f = self.feature_gat(torch.cat([x, embeddings.unsqueeze(0).repeat(batch_num, 1, 1).permute(0, 2, 1)], 1))
        x_f = self.feature_gat(x)
        # weights = embeddings.detach().clone()
        # weights = embeddings
        # cos_ji_mat = torch.matmul(weights, weights.T)
        # normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        # cos_ji_mat = cos_ji_mat / normed_mat
        # cos_ji_mat = self.soft(self.relu(cos_ji_mat))
        # cos_ji_mat = self.prune_tensor(cos_ji_mat, math.ceil(0.6 * self.n_features * self.n_features))

        # x_f = torch.matmul(x_f, cos_ji_mat.T)
        x_f = torch.matmul(x_f, self.w)
        _, x_g = self.gru(torch.cat([x_f, x_t], 2))
        recons = self.recon(x_g)
        res = self.forcast((x_t + x_f).permute(0, 2, 1))

        return recons, res.permute(0, 2, 1)



    def prune_tensor(self, A_pruned, n):
        # 获取邻接矩阵的大小
        num_nodes = A_pruned.size(0)
        # 对邻接矩阵进行n次剪枝操作
        i = 0
        k = 0
        while i < n:
            k += 1
            # 获取邻接矩阵中最小值的行列索引
            min_idx = torch.argmin(A_pruned)
            min_row, min_col = min_idx // num_nodes, min_idx % num_nodes
            # 判断对应节点的度是否大于1，如果是则将该值置为0
            if torch.sum(A_pruned[min_row]) - A_pruned[min_row][min_col] > 0 \
                    or torch.sum(A_pruned[:, min_col]) - A_pruned[min_row][min_col] > 0:
                A_pruned[min_row][min_col] = 0
                i = i + 1
            else:
                A_pruned[min_row][min_col].fill_(torch.max(A_pruned))
        return A_pruned


