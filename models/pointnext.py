import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph


class PyGPointNextBlock(nn.Module):
    def __init__(self, in_channels, expansion=4, k=32, pos_dim=3):
        super().__init__()
        self.k = k
        self.pos_dim = pos_dim
        mid_channels = in_channels * expansion

        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(mid_channels + pos_dim, in_channels, 1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1))

    def forward(self, x, pos_trans):
        B, C, N = x.shape
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        pos_flat = pos_trans.transpose(1, 2).reshape(-1, self.pos_dim)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)

        edge_index = knn_graph(pos_flat, k=self.k, batch=batch_idx, loop=True)

        target_idx, source_idx = edge_index[0], edge_index[1]

        current_channels = x.shape[1]
        x_flat = x.transpose(1, 2).reshape(-1, current_channels)
        neighbor_feat = x_flat[source_idx]

        neighbor_pos = pos_flat[source_idx]
        center_pos = pos_flat[target_idx]
        relative_pos = neighbor_pos - center_pos

        grouped = torch.cat([relative_pos, neighbor_feat], dim=1)

        try:
            grouped = grouped.view(B, N, self.k, -1)
        except RuntimeError as e:
            raise RuntimeError(f"KNN shape mismatch! Ensure loop=True is set. Error: {e}")

        aggregated = grouped.max(dim=2)[0].transpose(1, 2)

        out = self.conv2(aggregated)
        out = self.bn2(out)

        return shortcut + self.gamma * out


class PyGPointNextEncoder(nn.Module):
    """PointNeXt encoder using PyTorch Geometric KNN for local feature aggregation.
    Input: [B, C, N] where C is (x, y, nx, ny) for 4D or (x, y, z) for 3D.
    """
    def __init__(self, in_channels=3, width=32, k=32, latent_dim=128):
        super().__init__()
        self.pos_dim = 2 if in_channels == 4 else 3

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 1),
            nn.BatchNorm1d(width),
            nn.GELU()
        )

        self.layer1 = PyGPointNextBlock(width, expansion=4, k=k, pos_dim=self.pos_dim)
        self.layer2 = PyGPointNextBlock(width, expansion=4, k=k, pos_dim=self.pos_dim)

        self.trans = nn.Sequential(
            nn.Conv1d(width, width * 2, 1),
            nn.BatchNorm1d(width * 2),
            nn.GELU()
        )

        self.layer3 = PyGPointNextBlock(width * 2, expansion=4, k=k, pos_dim=self.pos_dim)
        self.layer4 = PyGPointNextBlock(width * 2, expansion=4, k=k, pos_dim=self.pos_dim)

        self.final_proj = nn.Sequential(
            nn.Conv1d(width * 2, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        """x: [B, C, N]"""
        pos = x[:, :self.pos_dim, :]

        x = self.stem(x)

        x = self.layer1(x, pos)
        x = self.layer2(x, pos)

        x = self.trans(x)
        x = self.layer3(x, pos)
        x = self.layer4(x, pos)

        x = self.final_proj(x)
        return x
