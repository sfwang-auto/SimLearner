import torch
from torch import nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    r"""This class applies a graph convolutional layer

    Args:
        input_size (int): The size of input node feature vector
        output_size (int): The size of output node feature vector
        dropout (float): a dropout layer after MLP. Default=0.0
        activation (str, optional): an activation function (``'relu'`` or ``'leaky_relu'``) after MLP. Default=None.

    Shape:
        - x: :math:`[\sum^B_i{L_i}, N, 1, 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - Output: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector,
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0, activation: str = None):
        super(GraphConvLayer, self).__init__()

        self.nonlinear = nn.Sequential(
            nn.Conv2d(input_size, output_size, 1),
            nn.BatchNorm2d(output_size)
        )

        if activation == 'relu':
            self.nonlinear.add_module('Relu', nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            self.nonlinear.add_module('Leaky_Relu', nn.LeakyReLU(inplace=True))
        if dropout > 0:
            self.nonlinear.add_module('Dropout', nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, ld: list, am: torch.Tensor) -> torch.Tensor:
        x = aggregate_node(x, ld, am)
        x = self.nonlinear(x)
        return x


class GraphConvResBlock(nn.Module):
    r"""This class applies a residual block containing two GraphConvLayers

    Args:
        input_size (int): The size of input node feature vector
        output_size (int): The size of output node feature vector

    Shape:
        - x: :math:`[\sum^B_i{L_i}, N, 1, 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - Output: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector,
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Sequential(
            nn.Conv2d(self.input_size, self.output_size, 1),
            nn.BatchNorm2d(self.output_size)
        )

        self.graph_conv_layer_1 = GraphConvLayer(input_size, input_size, 0.2, 'relu')
        self.graph_conv_layer_2 = GraphConvLayer(input_size, output_size)
        self.activate = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, x, ld, am):
        # linear transformation for input
        residual = self.linear(x) if self.input_size != self.output_size else x
        # two GCN layers
        x = self.graph_conv_layer_1(x, ld, am)
        x = self.graph_conv_layer_2(x, ld, am)
        x = self.activate(x + residual)
        return x


class Encoder(nn.Module):
    r"""This class encode the raw features to the descriptors

    Args:
        n_ref_points (int): the number of reference points

    Shape:
        - x_o: :math:`[B, N, n_ref + 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - Output: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector, n_ref is the number of reference points
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """
    def __init__(self, n_ref_points):
        super().__init__()
        mlp1_dim = 64
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, mlp1_dim, (1, n_ref_points + 1)),
            nn.BatchNorm2d(mlp1_dim),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        cell_size = 64
        self.bilstm = nn.LSTM(mlp1_dim, cell_size, batch_first=True, bidirectional=True)

        mlp2_dim = 256
        self.mlp2 = nn.Sequential(
            nn.Conv2d(cell_size * 2, mlp2_dim, 1),
            nn.BatchNorm2d(mlp2_dim),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        gcrb_dim = [256, 512]
        self.gcl = GraphConvLayer(mlp2_dim, gcrb_dim[0], activation='leaky_relu')
        self.gcrb_1 = GraphConvResBlock(gcrb_dim[0], gcrb_dim[0])
        self.gcrb_2 = GraphConvResBlock(gcrb_dim[0], gcrb_dim[1])

        self.fc = nn.Sequential(
            nn.Conv2d(gcrb_dim[-1], 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(512, 400, 1),
        )

    def rnn_module(self, x, ld):
        x_1 = x.unsqueeze(1)  # [B, 1, N, 32]
        x_2 = self.mlp1(x_1)  # [B, 64, N, 1]
        x_2 = x_2.squeeze(3).permute(0, 2, 1)  # [B, N, 64]

        px = nn.utils.rnn.pack_padded_sequence(x_2, ld, enforce_sorted=False, batch_first=True)
        packed_out, (ht, ct) = self.bilstm(px)
        padded_out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, padding_value=-float('inf'))

        group_1 = []
        for i, len_seq in enumerate(ld):
            y = padded_out[0][i, :len_seq, :]
            group_1.append(y)
        x_3 = torch.cat(group_1, dim=0)
        x_3 = x_3.unsqueeze(2).unsqueeze(3)
        x_4 = self.mlp2(x_3)  # [B, 256, 1, 1]
        return x_4

    def forward(self, x_o, ld, am):
        x = self.rnn_module(x_o, ld)

        x = self.gcl(x, ld, am)
        x = self.gcrb_1(x, ld, am)
        x = self.gcrb_2(x, ld, am)

        group_2 = []
        prev = 0
        for i, len_seq in enumerate(ld):
            y = x[prev:prev + len_seq, :]
            g_max, _ = torch.max(y, 0)
            group_2.append(g_max)
            prev += len_seq
        x_1 = torch.stack(group_2, 0)  # [B, 512, 1, 1]
        x_2 = self.fc(x_1)
        x_2 = x_2.squeeze(-1).squeeze(-1)

        return x_2


class GraSR(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, n_key_points=31, dim=400, K=1024, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = Encoder(n_key_points)
        self.encoder_k = Encoder(n_key_points)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.id_queue = [""] * K

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pos_ids):
        # gather keys before updating queue
        bsz = keys.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + bsz <= self.K:
            self.queue[:, ptr:ptr + bsz] = keys.t()
            self.id_queue[ptr:ptr + bsz] = list(pos_ids)
        else:
            bsz0 = self.K - ptr
            self.queue[:, ptr:] = keys.t()[:, :bsz0]
            self.id_queue[ptr:] = list(pos_ids)[:bsz0]
            self.queue[:, :bsz - bsz0] = keys.t()[:, bsz0:]
            self.id_queue[:bsz - bsz0] = list(pos_ids)[bsz0:]
        ptr = (ptr + bsz) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, fea, adj, n_nodes, tmscores, pos_fea, pos_adj, pos_n_nodes, pos_ids):
        # compute query features
        q = self.encoder_q(fea, n_nodes, adj)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(pos_fea, pos_n_nodes, pos_adj)  # keys: NxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        pos_logits = (torch.einsum('nc,nc->n', [q, k]) / self.T).exp()
        # negative logits: NxK
        neg_logits = (torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) / self.T).exp()

        bsz = q.shape[0]
        device = q.device
        pos_tmscores = torch.zeros((bsz, 1), device=device)
        neg_tmscores = torch.zeros((bsz, self.K), device=device)
        for i in range(bsz):
            pos_tmscores[i] = tmscores[pos_ids[i]][i]
            for j, neg_id in enumerate(self.id_queue):
                neg_tmscores[i, j] = tmscores[neg_id][i] if neg_id != "" else 0
        mask = neg_tmscores < pos_tmscores

        contrast_loss = -(pos_logits / (neg_logits * mask).sum(-1)).log().mean()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pos_ids)
        
        return contrast_loss

    def graph_forward(self, fea, adj, n_nodes):
        q = self.encoder_q(fea, n_nodes, adj)  # queries: NxC
        q = F.normalize(q, dim=1)
        return q


def aggregate_node(x: torch.Tensor, ld: list, am: torch.Tensor) -> torch.Tensor:
    r"""
    This function is used to aggregate node features in the graph.

    :param x:  node feature matrix
    :param ld:  sequence length list
    :param am:  adjacency matrix
    :return:  updated node feature matrix

    Shape:
        - x: :math:`[\sum^B_i{L_i}, N, 1, 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - return: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector,
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """
    x = x.squeeze()
    prev, ba_group = 0, []
    for j, len_seq in enumerate(ld):
        y = x[prev:prev + len_seq, :]  # [l, input_size]
        single_am = am[j, :len_seq, :len_seq]
        y = single_am @ y  # [l, input_size]
        ba_group.append(y)
        prev += len_seq
    x = torch.cat(ba_group, 0)  # [B, input_size]
    x = x.unsqueeze(2).unsqueeze(3)  # [B, input_size, 1, 1]
    return x
