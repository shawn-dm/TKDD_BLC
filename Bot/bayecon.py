from typing import Union, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch import Tensor
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torchbnn as bnn
from blitz.modules import BayesianLinear

class BEConv(MessagePassing):
    r"""
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
        # m = x.num_nodes
        # feat_dim = data.graph['node_feat']
        row, col = edge_index
        row = row - row.min()

        A = SparseTensor(row=row, col=col,
                         sparse_sizes=(5000,5000)
                         ).to_torch_sparse_coo_tensor()


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
