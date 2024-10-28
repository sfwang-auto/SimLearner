import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from utils import parse_args
from dataset import GraSRDataset
from modules.models import get_model


def process_grasr_data(fea, adj, n_nodes, device):
    max_n_nodes = n_nodes.max()
    fea = fea[:, :max_n_nodes].to(device)
    adj = adj[:, :max_n_nodes, :max_n_nodes].to(device)
    return fea, adj


def infer(model, loader, device, model_name):
    model.eval()
    ids, h_Gs = [], []
    with torch.no_grad():
        for fea, adj, n_nodes, id in loader:
            fea, adj = process_grasr_data(fea, adj, n_nodes, device)
            h_G = model.graph_forward(fea, adj, n_nodes)
            ids += id
            h_Gs.append(h_G)
    h_Gs = torch.cat(h_Gs, dim=0)
    sims = (h_Gs[None] * h_Gs[:, None] / 0.07).sum(-1).exp()

    n = len(ids)
    tmscores = np.zeros((n, n))
    for i in range(n):
        align = torch.load('data/repre_aligned_processed/' + ids[i] + '.pt')['tmscore']
        for j in range(i + 1, n):
            tmscores[i, j] = align[ids[j]]
    
    sims = sims[torch.triu(torch.ones((n, n), dtype=bool), diagonal=1)].cpu().numpy()
    tmscores = tmscores[np.triu(np.ones((n, n), dtype=bool), k=1)]
    print(np.corrcoef(sims, tmscores)[0, 1])
    plt.scatter(sims, tmscores, marker='.')
    plt.savefig('outputs/' + model_name + '.png')


def main():
    args = parse_args('grasr.yaml')
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    test_set = GraSRDataset(args, 'test')
    test_loader = DataLoader(test_set, args.bsz, shuffle=False)

    model = get_model(args, device, model_path=f'paras/{args.model_name}.h5')
    best_model = get_model(args, device, model_path=f'paras/{args.model_name}_best.h5')
    infer(model, test_loader, device, args.model_name)
    # infer(best_model, test_loader, device, args.model_name)

if __name__ == "__main__":
    main()