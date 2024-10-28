import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils import LossRecorder


def train(args, model, train_loader, val_loader,  device):
    name = "simlearner"
    optimizer = Adam(model.parameters(), args.lr)

    os.makedirs('paras', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    records = []
    best_lr_loss = 1e6
    for epoch in range(args.n_epochs):
        model.train()
        t = tqdm(train_loader)
        train_recorder = LossRecorder(['lr'])
        for data, positive_data, negative_data in t:
            optimizer.zero_grad()
            
            data = data.to(device)
            positive_data = positive_data.to(device)
            negative_data = negative_data.to(device)
            lr_loss = model(data, positive_data, negative_data)
            
            lr_loss.backward()
            optimizer.step()

            train_recorder.update('lr', lr_loss.item(), data.batch[-1].item() + 1)

            t.set_description(
                "epoch: %d lr: %.3f" % (epoch, train_recorder.avg_loss['lr'])
            )
        
        model.eval()
        val_recorder = LossRecorder(['lr'])
        with torch.no_grad():
            for data, positive_data, negative_data in val_loader:
                data = data.to(device)
                positive_data = positive_data.to(device)
                negative_data = negative_data.to(device)
                lr_loss = model(data, positive_data, negative_data)
                val_recorder.update('lr', lr_loss.item(), data.batch[-1].item() + 1)

            val_lr_loss = val_recorder.avg_loss['lr']
        
        records.append(
            np.array([
                train_recorder.avg_loss['lr'], 
                val_lr_loss, 
            ])[None]
        )
        
        if val_lr_loss < best_lr_loss:
            best_lr_loss = val_lr_loss
            torch.save(model.state_dict(), f"paras/{name}_best.h5")
    
    records = np.concatenate(records, 0)
    torch.save(model.state_dict(), f"paras/{name}.h5")

    plt.plot(records[:, 0], label='train')
    plt.plot(records[:, 1], label='val')
    plt.legend()
    plt.savefig(f"outputs/{name}_loss.png")

def process_grasr_data(fea, adj, n_nodes, pos_fea, pos_adj, pos_n_nodes, device):
    max_n_nodes = n_nodes.max()
    max_pos_n_nodes = pos_n_nodes.max()

    fea = fea[:, :max_n_nodes].to(device)
    adj = adj[:, :max_n_nodes, :max_n_nodes].to(device)
    pos_fea = pos_fea[:, :max_pos_n_nodes].to(device)
    pos_adj = pos_adj[:, :max_pos_n_nodes].to(device)
    return fea, adj, pos_fea, pos_adj

def grasr_train(args, model, train_loader, val_loader,  device):
    name = "grasr"
    optimizer = Adam(model.parameters(), args.lr)

    os.makedirs('paras', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    records = []
    # best_contrast_loss = 1e6
    for epoch in range(args.n_epochs):
        model.train()
        t = tqdm(train_loader)
        train_recorder = LossRecorder(['contrast'])
        for fea, adj, n_nodes, tmscores, pos_fea, pos_adj, pos_n_nodes, pos_ids in t:
            optimizer.zero_grad()
            
            fea, adj, pos_fea, pos_adj = process_grasr_data(fea, adj, n_nodes, pos_fea, pos_adj, pos_n_nodes, device)
            contrast_loss = model(fea, adj, n_nodes, tmscores, pos_fea, pos_adj, pos_n_nodes, pos_ids)
            
            contrast_loss.backward()
            optimizer.step()

            train_recorder.update('contrast', contrast_loss.item(), fea.shape[0])

            t.set_description(
                "epoch: %d contrast: %.3f" % (epoch, train_recorder.avg_loss['contrast'])
            )

        records.append(train_recorder.avg_loss['contrast'])
        
        # model.eval()
        # val_recorder = LossRecorder(['contrast'])
        # with torch.no_grad():
        #     for fea, adj, n_nodes, tmscores, pos_fea, pos_adj, pos_n_nodes, pos_ids in val_loader:
        #         fea, adj, pos_fea, pos_adj = process_grasr_data(fea, adj, n_nodes, pos_fea, pos_adj, pos_n_nodes, device)
        #         contrast_loss = model(fea, adj, n_nodes, tmscores, pos_fea, pos_adj, pos_n_nodes, pos_ids)
        #         val_recorder.update('contrast', contrast_loss.item(), fea.shape[0])

        #     val_contrast_loss = val_recorder.avg_loss['contrast']
        
        # records.append(
        #     np.array([
        #         train_recorder.avg_loss['contrast'], 
        #         val_contrast_loss, 
        #     ])[None]
        # )
        
        # if val_contrast_loss < best_contrast_loss:
        #     best_contrast_loss = val_contrast_loss
        #     torch.save(model.state_dict(), f"paras/{name}_best.h5")
    
    torch.save(model.state_dict(), f"paras/{name}.h5")
    torch.save(optimizer.state_dict(), f"paras/{name}_optimizer.h5")

    plt.plot(records, label='train')
    plt.legend()
    plt.savefig(f"outputs/{name}_loss.png")


def process_deepfold_data(dist, mask, id, pos_dist, pos_mask, pos_id, device):
    mask = torch.cat([mask, pos_mask], dim=0)
    dist = torch.cat([dist, pos_dist], dim=0)
    id = id + pos_id

    max_length = torch.where(mask.any(0).any(0))[0].max().item()
    dist = dist[:, None, :max_length, :max_length].to(device)
    mask = mask[:, None, :max_length, :max_length].to(device)
    return dist, mask, id


def deepfold_train(args, model, train_loader, val_loader, train_id_to_subset, val_id_to_subset, device):
    optimizer = Adam(model.parameters(), args.lr)

    os.makedirs('paras', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    records = []
    best_lr_loss = 1e6
    for epoch in range(args.n_epochs):
        model.train()
        t = tqdm(train_loader)
        train_recorder = LossRecorder(['mm'])
        for dist, mask, id, pos_dist, pos_mask, pos_id in t:
            optimizer.zero_grad()
            
            dist, mask, id = process_deepfold_data(dist, mask, id, pos_dist, pos_mask, pos_id, device)
            lr_loss = model(dist, mask, id, train_id_to_subset)
            
            lr_loss.backward()
            optimizer.step()

            train_recorder.update('lr', lr_loss.item(), data.batch[-1].item() + 1)

            t.set_description(
                "epoch: %d lr: %.3f" % (epoch, train_recorder.avg_loss['lr'])
            )
        
        model.eval()
        val_recorder = LossRecorder(['lr'])
        with torch.no_grad():
            for data, positive_data, negative_data in val_loader:
                data = data.to(device)
                positive_data = positive_data.to(device)
                negative_data = negative_data.to(device)
                lr_loss = model(data, positive_data, negative_data)
                val_recorder.update('lr', lr_loss.item(), data.batch[-1].item() + 1)

            val_lr_loss = val_recorder.avg_loss['lr']
        
        records.append(
            np.array([
                train_recorder.avg_loss['lr'], 
                val_lr_loss, 
            ])[None]
        )
        
        if val_lr_loss < best_lr_loss:
            best_lr_loss = val_lr_loss
            torch.save(model.state_dict(), "paras/deepfold_best.h5")
    
    records = np.concatenate(records, 0)
    torch.save(model.state_dict(), "paras/deepfold.h5")

    plt.plot(records[:, 0], label='train')
    plt.plot(records[:, 1], label='val')
    plt.legend()
    plt.savefig("outputs/deepfold_loss.png")
