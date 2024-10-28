import torch
from torch import nn
import torch.nn.functional as F


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def __call__(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        
        if self.training:
            mean = (x.detach() * mask).sum(dim=(0, 2, 3)) / mask.sum(dim=(0, 2, 3))
            x_norm = (x - mean[None, :, None, None]) * mask
            var = x_norm.detach().pow(2).sum(dim=(0, 2, 3)) / mask.sum(dim=(0, 2, 3))

            device = x.device
            self.running_mean = (1 - self.momentum) * self.running_mean.to(device) + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var.to(device) + self.momentum * var

            x_norm = x_norm / torch.sqrt(var[None, :, None, None] + self.eps)
        else:
            x_norm = (x - self.running_mean[None, :, None, None]) * mask
            x_norm = x_norm / torch.sqrt(self.running_var[None, :, None, None] + self.eps)

        if self.affine:
            x_norm = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
        return x_norm


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=int(kernel_size / 2 - 1))
        self.norm = BatchNorm2d(out_channels)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x, mask):
        x = self.conv(x)

        kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size), device=x.device)
        mask = F.conv2d(mask.float(), kernel, stride=2, padding=int(self.kernel_size / 2 - 1)) > 0

        x = self.norm(x, mask)
        # x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x, mask


class DeepFold(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 1
        kernel_size_list = [12, 4, 4, 4, 4, 4]
        out_channels_list = [128, 256, 512, 512, 512, 398]

        self.blocks = nn.ModuleList([])
        for kernel_size, out_channels in zip(kernel_size_list, out_channels_list):
            self.blocks.append(Block(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask, id, train_id_to_subset):
        for block in self.blocks:
            x, mask = block(x, mask)
        x = x * torch.eye(x.shape[-1], device=x.device)

