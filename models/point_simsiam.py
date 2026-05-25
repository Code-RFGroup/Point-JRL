import torch.nn as nn


class PointSimSiam(nn.Module):
    """SimSiam wrapper for point cloud encoders (e.g., PointNeXt).
    Adds a 3-layer projector and 2-layer predictor on top of the base encoder.
    """
    def __init__(self, base_encoder, encoder_out_dim=128, dim=128, pred_dim=64):
        super(PointSimSiam, self).__init__()

        self.encoder = base_encoder

        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
