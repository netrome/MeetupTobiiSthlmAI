import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, batch):
        return batch.view(batch.size(0), -1)

class Unflatten(nn.Module):
    def forward(self, batch):
        return batch.view(batch.size(0), 8, 5, 5)
