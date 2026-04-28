import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import SAGEConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_channels)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        return self.dropout(x)
