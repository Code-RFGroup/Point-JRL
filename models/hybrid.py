import torch.nn as nn


class HybridSimSiam(nn.Module):
    def __init__(self, base_encoder, encoder_out_dim, dim, pred_dim, num_classes=2):
        super(HybridSimSiam, self).__init__()

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

        self.classifier = nn.Linear(encoder_out_dim, num_classes)

    def forward(self, x1, x2):
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)

        z1 = self.projector(feat1)
        z2 = self.projector(feat2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        logits1 = self.classifier(feat1)
        logits2 = self.classifier(feat2)

        return p1, p2, z1.detach(), z2.detach(), logits1, logits2
