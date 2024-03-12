import sys, os

sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_mean

from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GINConv, global_mean_pool ,ResGatedGraphConv,TransformerConv , GCNConv, GatedGraphConv, global_add_pool, Set2Set
from torch.nn import Sequential, Linear, ReLU
from bayecon import BEConv
import copy
import math
import random
import torchbnn as bnn
from torch_geometric.utils import subgraph


class PriorDiscriminator(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = th.nn.Linear(input_dim, input_dim)
        self.l1 = th.nn.Linear(input_dim, input_dim)
        self.l2 = th.nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return th.sigmoid(self.l2(h))


class FF(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = th.nn.Sequential(
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU()
        )
        self.linear_shortcut = th.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # a = self.block(x)
        # print(a.shape)
        # b = self.linear_shortcut(x)
        # print(b.shape)
        c = self.block(x) + self.linear_shortcut(x)
        return c


class Encoder(th.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = th.nn.ModuleList()
        self.bns = th.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:

                conv = BEConv(dim, dim)
 
            else:
                conv = BEConv(num_features, dim)

            bn = th.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

        self.mlp = th.nn.Sequential(th.nn.Linear(384, 192),th.nn.ReLU(),th.nn.Linear(192,192))


        self.multi_en = th.nn.MultiheadAttention(embed_dim=num_features, num_heads=8, dropout=0.5)
        self.pool = Set2Set(64, processing_steps=2)
        self.ffn_en = FF(dim * 2)  # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = th.nn.LayerNorm(num_features)
        self.batchnorm = th.nn.BatchNorm1d(dim)
        self.layer_norm2 = th.nn.LayerNorm(dim * 2)
        self.dropout = th.nn.Dropout(0.5)
        self.lin0 = th.nn.Linear(dim * 2, dim)
        self.lin1 = th.nn.Linear(dim * 4, dim * 3)

        self.lstm1 = th.nn.LSTM(num_features, dim, num_layers=2, dropout=0, bidirectional=True)
        self.lin2 = th.nn.Linear(num_features, dim *2)

    def forward(self, x, edge_index, batch):

        x_one = copy.deepcopy(x)
        xs_one = []
        for i in range(self.num_gc_layers):
            x_one = F.relu(self.convs[i](x_one, edge_index))
            xs_one.append(x_one)

        # xpool_one = [self.pool(x_one, batch) for x_one in xs_one]
        xpool_one = [global_add_pool(x_one, batch) for x_one in xs_one]

        # xpool_one = [global_mean_pool(x_one, batch) for x_one in xs_one]

        x_one = th.cat(xpool_one, 1)
        # x_one = self.mlp(x_one)
        num_graphs = max(batch) + 1
        news = th.stack([x[(batch == idx).nonzero(as_tuple=False).squeeze()[0]] for idx in range(num_graphs)])
        out = self.layer_norm1(news)
        out = out.unsqueeze(1)
        skip_out = out
        out, attn_wt = self.multi_en(out[-1:, :, :], out, out)

        out = out + skip_out

        out, _ = self.lstm1(out) # seq_len, batch, input_size

        out = self.layer_norm2(out)  # Layer norm
        # skip_out = out
        out = self.ffn_en(out)
        out = out.squeeze(1)
        # out = self.dropout(out)
        news = F.relu(self.lin0(out))
        # news = F.dropout(news,p=0.5)
        x_one = th.cat([x_one, news], dim=1)
        x_one = self.lin1(x_one)


        b = th.cat(xs_one, 1)
        # print(x_one.shape)
        # print(b.shape)

        return x_one, b

    def get_embeddings(self, data):

        with th.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_embed, node_embed = self.forward(x, edge_index, batch)
        return node_embed


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)
    # Ep = log_2 + p_samples
    Ep = log_2 - F.softplus(- p_samples)
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)
    # Eq = q_samples

    Eq = F.softplus(-q_samples) + q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, edge_index, batch, measure, l_enc_pos, l_enc_dropped, l_enc_dropped_two):
    '''
    Args:
        l: Local feature map.
        g: Global features.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = th.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = th.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = th.mm(l_enc, g_enc.t())
    res_two = th.mm(l_enc_pos, g_enc.t())
    res_three = th.mm(l_enc_dropped, g_enc.t())
    res_four = th.mm(l_enc_dropped_two, g_enc.t())

    E_pos = get_positive_expectation((res + res_two + res_three + res_four) * pos_mask / 4, measure,
                                     average=False).sum()
    E_pos = E_pos / num_nodes

    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


class Net(th.nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(Net, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(5000, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)  # Feed forward layer
        self.global_d = FF(self.embedding_dim)  # Feed forward layer
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        # self.local_d = FF(192)
        # self.global_d = FF(192)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):

        x, edge_index, dropped_edge_index, batch, num_graphs, mask = data.x, data.edge_index, data.dropped_edge_index, data.batch, max(
            data.batch) + 1, data.mask
        edge_sub, _ = subgraph(mask, edge_index)  # generate subgraph's edge index
        node_mask = th.ones((x.size(0), 1)).cuda()
        for i in range(x.size(0)):
            if random.random() >= 0.8:
                node_mask[i] = 0
        node_mask[data.rootindex] = 1
        x_pos_two = x * node_mask  # mask node

        y, M = self.encoder(x, edge_index, batch)  # y->num_graphs x dim; M->num_nodes x dim
        y_pos, M_pos = self.encoder(x, edge_sub, batch)  # subgraph
        y_dropped, M_dropped = self.encoder(x, dropped_edge_index, batch)  # drop edge
        y_dropped_two, M_dropped_two = self.encoder(x_pos_two, edge_index, batch)  # mask_node

        g_enc = self.global_d(y)  # feed forward
        l_enc = self.local_d(M)  # feed forward

        l_enc_pos = self.local_d(M_pos)
        l_enc_dropped = self.local_d(M_dropped)
        l_enc_dropped_two = self.local_d(M_dropped_two)

        # kl = self.kl_loss(self.encoder).cuda()

        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure, l_enc_pos, l_enc_dropped,
                                               l_enc_dropped_two)
                            # +0.1*kl

        return local_global_loss


class Classfier(th.nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes):
        super(Classfier, self).__init__()
        self.linear_one = th.nn.Linear(5000 * 2, 2 * hid_feats)
        self.linear_two = th.nn.Linear(2 * hid_feats, hid_feats)
        self.linear_three = th.nn.Linear(in_feats, hid_feats)
        # self.linear_three = th.nn.Linear(in_feats+768, hid_feats)

        self.linear_transform = th.nn.Linear(hid_feats * 2, 4)
        self.prelu = th.nn.PReLU()


        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embed, data):
        ori = scatter_mean(data.x, data.batch, dim=0)
        root = data.x[data.rootindex]
        ori = th.cat((ori, root), dim=1)


        ori = self.linear_one(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)
        ori = self.linear_two(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)

        x = scatter_mean(embed, data.batch, dim=0)
        x = self.linear_three(x)
        x = F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        out = th.cat((x, ori), dim=1)
        out = self.linear_transform(out)
        x = F.log_softmax(out, dim=1)
        return x

