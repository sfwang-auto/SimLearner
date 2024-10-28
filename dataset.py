import os
import math
import torch
import random
import torch.utils.data as data
import torch.nn.functional as F
from torch_geometric.data import Data

BB_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N"]
LETTER_TO_NUM = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'N': 4, '_': 4}


class RNADataset(data.Dataset):
    def __init__(self, args, split, radius=20, read_all=False):
        super().__init__()
        self.radius = radius
        self.bb_atoms = args.bb_atoms
        self.central_idx = self.bb_atoms.index(args.central_atom)
        self.bb_idx = [BB_ATOMS.index(bb_atom) for bb_atom in self.bb_atoms]
        
        self.is_test = split == 'test'
        self.processed_dir = 'data/repre_aligned_processed/'
        self.split = torch.load('data/repre_split.pt')[split]

        if split == 'train':
            self.id_to_subset = torch.load('data/repre_train_id_to_subset.pt')
        elif split == 'val':
            self.id_to_subset = torch.load('data/repre_val_id_to_subset.pt')
        elif split == 'test':
            self.id_to_subset = torch.load('data/repre_test_id_to_subset.pt')

        self.read_all = read_all
        if read_all:
            self.data = {}
            for id in self.split:
                self.data[id] = torch.load(os.path.join(self.processed_dir, id + '.pt'))

    
    def radius_neighbor(self, X, eps=1e-6):
        dist = torch.sqrt(
            (X[:, None] - X[None]).pow(2).sum(-1) + eps
        )

        n = X.shape[0]
        dist[torch.arange(n), torch.arange(n)] = 0
        tgt_idx, src_idx = torch.where(dist < self.radius)
        return src_idx, tgt_idx

    def __len__(self): 
        return len(self.split)
    
    def process(self, id, align_id=None):
        data = self.data[id] if self.read_all else torch.load(os.path.join(self.processed_dir, id + '.pt'))

        coords = torch.tensor(data['coords'][:, self.bb_idx]).to(torch.float32)
        central_coords = coords[:, self.central_idx]

        src_idx, tgt_idx = self.radius_neighbor(central_coords)

        edge_index = torch.cat([src_idx[None], tgt_idx[None]], dim=0)

        mask = ~torch.isnan(central_coords[:, 0])

        if align_id is None or id == align_id:
            tmscore = 1
        else:
            tmscore = data['tmscore'][align_id]

        num_nodes = coords.shape[0]
        return Data(
            mask=mask, id=id, chain_id=data['chain_id'],  
            num_nodes=num_nodes, edge_index=edge_index, 
            coords=coords, central_coords=central_coords, 
            tmscore=tmscore
        )
    
    def __getitem__(self, idx): 
        id = self.split[idx]
        data = self.process(id)

        if self.is_test:
            return data
        else:
            positive_ids = self.id_to_subset[id]
            random.shuffle(positive_ids)
            positive_data = self.process(positive_ids[0], id)

            negative_ids = list(self.id_to_subset.keys())
            random.shuffle(negative_ids)
            negative_data = self.process(negative_ids[0], id)

            return data, positive_data, negative_data


class DeepFoldDataset(data.Dataset):
    def __init__(self, args, split, max_length=512, read_all=False):
        super().__init__()
        self.max_length = max_length

        self.bb_atoms = args.bb_atoms
        self.central_idx = self.bb_atoms.index(args.central_atom)
        self.bb_idx = [BB_ATOMS.index(bb_atom) for bb_atom in self.bb_atoms]
        
        self.processed_dir = 'data/repre_aligned_processed/'
        self.split = torch.load('data/repre_split.pt')[split]

        if split == 'train':
            self.id_to_subset = torch.load('data/deepfold_repre_train_id_to_subset.pt')
        elif split == 'val':
            self.id_to_subset = torch.load('data/deepfold_repre_val_id_to_subset.pt')
        self.is_test = split == 'test'

        self.read_all = read_all
        if read_all:
            self.data = {}
            for id in self.split:
                self.data[id] = torch.load(os.path.join(self.processed_dir, id + '.pt'))

    def __len__(self): 
        return len(self.split)
    
    def process(self, id):
        data = self.data[id] if self.read_all else torch.load(os.path.join(self.processed_dir, id + '.pt'))

        coords = torch.tensor(data['coords'][:, self.bb_idx]).to(torch.float32)
        central_coords = coords[:, self.central_idx]

        n_res = central_coords.shape[0]
        n_pad = self.max_length - n_res
        dist = (central_coords[None] - central_coords[:, None]).pow(2).sum(-1).sqrt().pow(-2)
        dist[torch.arange(n_res), torch.arange(n_res)] = 0
        dist = torch.cat([dist, torch.ones(n_res, n_pad) * torch.nan], dim=-1)
        dist = torch.cat([dist, torch.ones(n_pad, self.max_length) * torch.nan], dim=0)

        mask = ~torch.isnan(dist)
        dist = torch.nan_to_num(dist)

        return dist, mask
    
    def __getitem__(self, idx): 
        id = self.split[idx]
        dist, mask = self.process(id)

        if self.is_test:
            return data
        else:
            positive_ids = self.id_to_subset[id]
            random.shuffle(positive_ids)
            positive_id = positive_ids[0]
            pos_dist, pos_mask = self.process(positive_id)
            return dist, mask, id, pos_dist, pos_mask, positive_id


class GraSRDataset(data.Dataset):
    def __init__(self, args, split, read_all=False, bsz=64, queue_size=1024, n_ref_points=31, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.bb_atoms = args.bb_atoms
        self.central_idx = self.bb_atoms.index(args.central_atom)
        self.bb_idx = [BB_ATOMS.index(bb_atom) for bb_atom in self.bb_atoms]

        self.bsz = bsz
        self.cur_epoch = 0
        self.queue_size = queue_size
        self.n_ref_points = n_ref_points
        
        self.processed_dir = 'data/repre_aligned_processed/'
        self.split = torch.load('data/repre_split.pt')[split]

        if split == 'train':
            self.id_to_subset = torch.load('data/grasr_repre_train_id_to_subset.pt')
        elif split == 'val':
            self.id_to_subset = torch.load('data/grasr_repre_val_id_to_subset.pt')
        self.is_test = split == 'test'

        self.read_all = read_all
        self.data = {}
        for id in self.split:
            if read_all:
                self.data[id] = self.extract_features(torch.load(os.path.join(self.processed_dir, id + '.pt')))
    
    def get_relative_dist(self, coords):
        mask = ~torch.isnan(coords).any(-1)

        group_num = int(math.log2(self.n_ref_points + 1))
        assert 2 ** group_num - 1 == self.n_ref_points,\
            "The number of anchor points is {} and should be 2^k - 1, " \
            "where k is an integer, but k is {}.".format(self.n_ref_points, group_num)
        n_points = coords.shape[0]
        ref_points = []
        for i in range(group_num):
            n_points_in_group = 2 ** i
            for j in range(n_points_in_group):
                beg, end = n_points * j // n_points_in_group, math.ceil(n_points * (j + 1) / n_points_in_group)
                if mask[beg:end, None].sum():
                    ref_point = torch.nan_to_num(coords[beg:end]).sum(0) / mask[beg:end, None].sum()
                else:
                    ref_point = None
                ref_points.append(ref_point)
        relative_dist = [(coords - rp[None]).pow(2).sum(-1).sqrt()[:, None] if rp is not None else torch.zeros_like(coords[:, :1]) for rp in ref_points]
        relative_dist = torch.nan_to_num(torch.cat(relative_dist, dim=-1))
        return relative_dist

    def cal_angle_cosine(self, coords):
        dX0 = F.normalize(coords[:-2] - coords[1:-1], dim=-1)
        dX1 = F.normalize(coords[2:] - coords[1:-1], dim=-1)
        cosine = (dX0 * dX1).sum(-1)
        cosine = F.pad(cosine, (1, 1), 'constant', torch.nan)
        cosine = torch.nan_to_num(cosine)
        return cosine.reshape(-1, 1)
    
    def extract_features(self, data):
        coords = torch.tensor(data['coords'][:, self.bb_idx]).to(torch.float32)[:, self.central_idx]
        relative_dist = self.get_relative_dist(coords)
        angle_cosine = self.cal_angle_cosine(coords)
        fea = torch.cat([relative_dist, angle_cosine], dim=-1)

        omega, epsilon = 4.0, torch.tensor(2.0)
        dist = (coords[None] - coords[:, None]).pow(2).sum(-1).sqrt()
        adj = omega / torch.maximum(dist, epsilon)
        adj = torch.nan_to_num(adj)

        n_nodes = fea.shape[0]
        n_pad = self.max_length - n_nodes
        fea = torch.cat([fea, torch.zeros(n_pad, fea.shape[1])], dim=0)
        adj = torch.cat([adj, torch.zeros(n_nodes, n_pad)], dim=-1)
        adj = torch.cat([adj, torch.zeros(n_pad, self.max_length)], dim=0)
        return (fea, adj, n_nodes, data['tmscore'])

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        id = self.split[idx]
        fea, adj, n_nodes, tmscores = self.data[id] if self.read_all else  self.extract_features(torch.load(os.path.join(self.processed_dir, id + '.pt')))
        tmscores[id] = 1

        if self.is_test:
            return fea, adj, n_nodes, id
        else:
            positive_ids = self.id_to_subset[id]
            random.shuffle(positive_ids)
            positive_id = positive_ids[0]
            pos_fea, pos_adj, pos_n_nodes, _ = self.data[positive_id] if self.read_all else self.extract_features(torch.load(os.path.join(self.processed_dir, positive_id + '.pt')))
            return fea, adj, n_nodes, tmscores, pos_fea, pos_adj, pos_n_nodes, positive_id
            