import torch
from torch_geometric.loader import DataLoader

from trainer import deepfold_train
from dataset import DeepFoldDataset
from modules.models import get_model
from utils import set_seed, parse_args


def main():
    set_seed()
    args = parse_args('deepfold.yaml')
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)

    train_set = DeepFoldDataset(args, 'train', read_all=False)
    val_set = DeepFoldDataset(args, 'val', read_all=False)
    train_loader = DataLoader(train_set, args.bsz // 2, shuffle=True)
    val_loader = DataLoader(val_set, args.bsz // 2, shuffle=False)

    deepfold_train(args, model, train_loader, val_loader, train_set.id_to_subset, val_set.id_to_subset, device)


if __name__ == "__main__":
    main()