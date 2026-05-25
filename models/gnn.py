import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax


class LocationEdgeUpdateLayer(nn.Module):
    """Edge update layer (Eq. 5 and Eq. 6 in the paper).
    Explicitly distinguishes Internal (type 0) and External (type 1) edges.
    Uses a one-layer MLP for feature update.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()

        # Eq. 5: U^k is a one-layer MLP
        self.U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Eq. 6: H^k_r is a one-layer MLP
        # Input: edge_feat (64) + node_v (64) + node_w (64) = 192
        self.H_internal = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU()
        )
        self.H_external = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, edge_types):
        # x: [N, 64], edge_attr: [E, 64]
        row, col = edge_index

        # 1. Transform node features (Eq. 5)
        h_v = self.U(x[row])
        h_w = self.U(x[col])

        # 2. Concat edge + node pair (Eq. 6 input)
        # he || U(hv) || U(hw) -> [E, 192]
        concat_feat = torch.cat([edge_attr, h_v, h_w], dim=-1)

        # 3. Apply type-specific update function
        mask_internal = (edge_types == 0).squeeze()
        mask_external = (edge_types == 1).squeeze()

        new_edge_attr = torch.zeros_like(edge_attr)

        if mask_internal.sum() > 0:
            res = self.H_internal(concat_feat[mask_internal])
            new_edge_attr[mask_internal] = res.to(new_edge_attr.dtype)

        if mask_external.sum() > 0:
            res = self.H_external(concat_feat[mask_external])
            new_edge_attr[mask_external] = res.to(new_edge_attr.dtype)

        return new_edge_attr


class LocationNodeUpdateLayer(MessagePassing):
    """Node update layer (Eq. 7, 8, 9 in the paper).
    Uses a two-layer MLP (S_r) with attention computed from edge features (Eq. 9).
    """
    def __init__(self, hidden_dim=64):
        super().__init__(aggr='add')

        self.hidden_dim = hidden_dim

        # Eq. 7: U^k (used inside node update)
        self.U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Eq. 7: S^k_r is a "two-layer MLP"
        # Input: U(h_w) (64) + h_edge (64) = 128
        self.S_internal = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.S_external = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Eq. 9: Attention vector 'a'
        self.att_vec = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.att_vec)

    def forward(self, x, edge_index, edge_attr, edge_types):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_types=edge_types)

    def message(self, x_j, edge_attr, edge_types, index):
        # x_j: neighbor features [E, 64]
        # edge_attr: UPDATED edge features (h^{k+1}_{ev,w}) [E, 64]

        # Part A: Compute Message Content (Eq. 7)
        u_xw = self.U(x_j)
        cat_feat = torch.cat([u_xw, edge_attr], dim=-1)  # [E, 128]

        mask_internal = (edge_types == 0).squeeze()
        mask_external = (edge_types == 1).squeeze()

        msg = torch.zeros_like(x_j)

        if mask_internal.sum() > 0:
            res = self.S_internal(cat_feat[mask_internal])
            msg[mask_internal] = res.to(msg.dtype)

        if mask_external.sum() > 0:
            res = self.S_external(cat_feat[mask_external])
            msg[mask_external] = res.to(msg.dtype)

        # Part B: Compute Attention Coefficients (Eq. 9)
        # alpha = exp( a^T * LeakyReLU( h_edge ) )
        edge_feat_activated = F.leaky_relu(edge_attr, 0.2)
        alpha_scores = (edge_feat_activated * self.att_vec).sum(dim=-1)
        alpha = softmax(alpha_scores, index)

        # Part C: Weighted Message
        return msg * alpha.view(-1, 1)

    def update(self, aggr_out, x):
        # Eq. 8: Residual Connection
        return x + aggr_out


class DATE25_GNN(nn.Module):
    """DATE 2025: Location is All You Need - Full Architecture.
    Default parameters set according to the paper.
    """
    def __init__(self,
                 node_in_dim=4,   # x1, y1, x2, y2
                 edge_in_dim=1,   # distance
                 hidden_dim=64,   # Fig 5
                 num_layers=4):   # Sec III.C
        super().__init__()

        # 1. Initial Embedding (Linear Projection)
        self.node_enc = nn.Linear(node_in_dim, hidden_dim)
        self.edge_enc = nn.Linear(edge_in_dim, hidden_dim)

        # 2. Stack of 4 Message-Passing Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'edge_update': LocationEdgeUpdateLayer(hidden_dim),
                'node_update': LocationNodeUpdateLayer(hidden_dim)
            }))

        # 3. Classifier Head (Fig. 5)
        # Input: 64 (Max) + 64 (Avg) = 128
        # Structure: 128 -> 64 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Handle edge_type
        if hasattr(data, 'edge_type'):
            edge_types = data.edge_type
        else:
            edge_types = torch.ones(edge_index.size(1), 1, device=x.device, dtype=torch.long)

        # A. Feature Initialization
        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)

        # B. Message Passing Loop
        for layer in self.layers:
            # Order: Edge Update First (Eq 5,6) -> Node Update (Eq 7,8,9)
            edge_attr = layer['edge_update'](x, edge_index, edge_attr, edge_types)
            x = layer['node_update'](x, edge_index, edge_attr, edge_types)

        # C. Readout (Fig. 5: Global Max + Global Avg)
        gmp = global_max_pool(x, batch)
        gap = global_mean_pool(x, batch)
        representation = torch.cat([gmp, gap], dim=1)

        # D. Classification
        logits = self.classifier(representation)

        return logits, representation
