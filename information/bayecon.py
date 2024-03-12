from typing import Union, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch import Tensor
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torchbnn as bnn
from blitz.modules import BayesianLinear
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score

class BEConv(MessagePassing):
    r"""

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(BEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.mlp = torch.nn.Sequential(
        #     bnn.BayesLinear(
        #         prior_mu=0,
        #         prior_sigma=0.1,
        #         in_features=in_channels,
        #         out_features=in_channels,
        #     ),
        #     torch.nn.BatchNorm1d(in_channels),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.5),
        #     bnn.BayesLinear(
        #         prior_mu=0,
        #         prior_sigma=0.1,
        #         in_features=in_channels,
        #         out_features=out_channels,
        #     ),
        # )
        self.mlp = torch.nn.Sequential(
            BayesianLinear(
                in_features=out_channels,
                out_features=out_channels,
            ),

            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            BayesianLinear(
                in_features=out_channels,
                out_features=out_channels,
            ),
            # torch.nn.BatchNorm1d(out_channels),
            # torch.nn.ReLU(),
            # # torch.nn.Dropout(p=0.5),
            # BayesianLinear(
            #     in_features=out_channels,
            #     out_features=out_channels,
            # ),
        )


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
        self.lin2 = Linear(in_channels[1], out_channels, bias=False)
        self.lin3 = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

        self.eps = torch.nn.Parameter(torch.Tensor([0]))


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        a = self.lin1(x[0])
        b = self.lin2(x[1])

        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        # out = self.mlp(a + self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight,
        #                                                    size=None))
        out = self.mlp((1 + self.eps) * a + self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight,
                             size=None))

        return out + self.lin3(x[1])

    def message(self, a_j: Tensor, b_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = a_j - b_i
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class BCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(BCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = BEConv(nfeat, nhid)
        self.gc2 = BEConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # self.device = device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)



