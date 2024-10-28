import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from modules.layers import MPNNLayer


class Featurizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.central_idx = args.bb_atoms.index(args.central_atom)

        node_in_dim = 4
        edge_in_dim = 23
        hidden_dim = args.hidden_dim

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
    
    def cal_dihedral(self, X, batch, chain_id, eps=1e-6):
        X = X.reshape(-1, 3)

        dX = X[1:] - X[:-1]
        U = F.normalize(dX, dim=-1)

        cross1 = torch.cross(U[:-2], U[1:-1])
        cross2 = torch.cross(U[1:-1], U[2:])
        cross1 = F.normalize(cross1, dim=-1)
        cross2 = F.normalize(cross2, dim=-1)

        dihedral = torch.arccos(
            (cross1 * cross2).sum(-1).clip(-1 + eps, 1 - eps)
        ) * torch.sign((cross2 * U[:-2]).sum(-1))
        dihedral = F.pad(dihedral, (1, 2), 'constant', torch.nan)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        dihedral[idx] = torch.nan
        dihedral[idx - 1] = torch.nan
        dihedral[idx - 2] = torch.nan

        idx = np.where(chain_id[:-1] != chain_id[1:])[0] + 1
        dihedral[idx] = torch.nan
        dihedral[idx - 1] = torch.nan
        dihedral[idx - 2] = torch.nan
        return torch.cat([torch.sin(dihedral)[:, None], torch.cos(dihedral)[:, None]], dim=-1)
    
    def cal_angle(self, X, batch, chain_id, eps=1e-6):
        dX0 = F.normalize(X[:-2] - X[1:-1], dim=-1)
        dX1 = F.normalize(X[2:] - X[1:-1], dim=-1)

        cosine = (dX0 * dX1).sum(-1)
        sine = torch.sqrt(1 - cosine.pow(2) + eps)
        sine = F.pad(sine, (1, 1), 'constant', torch.nan)
        cosine = F.pad(cosine, (1, 1), 'constant', torch.nan)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        sine[idx] = torch.nan
        cosine[idx] = torch.nan
        sine[idx - 1] = torch.nan
        cosine[idx - 1] = torch.nan

        idx = np.where(chain_id[:-1] != chain_id[1:])[0] + 1
        sine[idx] = torch.nan
        cosine[idx] = torch.nan
        sine[idx - 1] = torch.nan
        cosine[idx - 1] = torch.nan
        return torch.cat([sine[:, None], cosine[:, None]], dim=-1)

    def rbf(self, D, D_min=0., D_max=20., num_rbf=16):
        D_mu = torch.linspace(D_min, D_max, num_rbf, device=D.device)
        for _ in range(len(D.shape)):
            D_mu = D_mu[None]
        D_sigma = (D_max - D_min) / num_rbf
        return torch.exp(
            -((D[..., None] - D_mu) / D_sigma) ** 2
        )
    
    def cal_dist(self, central_X, src_idx, tgt_idx, eps=1e-6):
        dist = torch.sqrt(
            (central_X[src_idx] - central_X[tgt_idx]).pow(2).sum(-1) + eps
        )
        dist = self.rbf(dist)
        return dist

    def cal_local_system(self, X, batch, chain_id):
        dX = X[1:] - X[:-1]
        u = F.normalize(dX, dim=-1)
        b = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        # Q = [b, n, b ✖️ n]
        Q = torch.stack((b, n, torch.cross(b, n)), dim=-1)
        Q = F.pad(Q, (0, 0, 0, 0, 1, 1), 'constant', torch.nan)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        Q[idx] = torch.nan
        Q[idx - 1] = torch.nan

        idx = np.where(chain_id[:-1] != chain_id[1:])[0] + 1
        Q[idx] = torch.nan
        Q[idx - 1] = torch.nan
        return Q
    
    def quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        signs = torch.sign(torch.stack([
            R[:, 2,1] - R[:, 1,2],
            R[:, 0,2] - R[:, 2,0],
            R[:, 1,0] - R[:, 0,1]
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        q = torch.cat((xyz, w), -1)
        q = F.normalize(q, dim=-1)
        return q
    
    def cal_orient(self, Q, src_idx, tgt_idx):
        src_Q = Q[src_idx]
        tgt_Q = Q[tgt_idx]
        R = torch.matmul(tgt_Q.transpose(-1, -2), src_Q)
        return self.quaternions(R)
    
    def cal_direct(self, Q, central_X, src_idx, tgt_idx):
        tgt_Q = Q[tgt_idx]
        direct = F.normalize(central_X[src_idx] - central_X[tgt_idx], dim=-1)
        direct = torch.matmul(tgt_Q.transpose(-1, -2), direct[..., None]).squeeze(-1)
        return direct

    def forward(self, data):
        batch = data.batch
        chain_id = np.concatenate(data.chain_id)
        central_coords = data.central_coords
        src_idx, tgt_idx = data.edge_index[0], data.edge_index[1]

        dihedral = self.cal_dihedral(central_coords, batch, chain_id)
        angle = self.cal_angle(central_coords, batch, chain_id)
        Q = self.cal_local_system(central_coords, batch, chain_id)
        orient = self.cal_orient(Q, src_idx, tgt_idx)
        dist = self.cal_dist(central_coords, src_idx, tgt_idx)
        direct = self.cal_direct(Q, central_coords, src_idx, tgt_idx)  

        h_V = torch.cat([dihedral, angle], -1)
        h_E = torch.cat([orient, dist, direct], -1)

        h_V = torch.nan_to_num(h_V)
        h_E = torch.nan_to_num(h_E)
        h_V = self.node_norm(self.node_embedding(h_V))
        h_E = self.edge_norm(self.edge_embedding(h_E))
        return h_V, h_E
    

class SimLearner(nn.Module):
    def __init__(self, args):
        super().__init__()
        drop_rate = args.drop_rate
        hidden_dim = args.hidden_dim
        self.temperature = args.temperature
        
        self.featurizer = Featurizer(args)

        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, drop_rate=drop_rate) for _ in range(args.n_layers)]
        )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def graph_forward(self, data):
        mask = data.mask
        batch = data.batch
        edge_idx = data.edge_index 
        h_V, h_E = self.featurizer(data)

        for layer in self.layers:
            h_V = layer(h_V, h_E, edge_idx)
        
        h_G = scatter_sum(
            h_V * mask[:, None], batch, dim=0
        ) / scatter_sum(torch.ones_like(h_V) * mask[:, None], batch, dim=0)
        h_G = F.normalize(h_G, dim=-1)

        return h_G
    
    def forward(self, data, positive_data, negative_data):
        h_G = self.graph_forward(data)
        h_G_pos = self.graph_forward(positive_data)
        h_G_neg = self.graph_forward(negative_data)

        sim_pos = (h_G * h_G_pos / self.temperature).sum(-1).exp()
        sim_neg = (h_G * h_G_neg / self.temperature).sum(-1).exp()
        return (sim_neg / sim_pos * positive_data.tmscore / negative_data.tmscore).log().pow(2).mean()