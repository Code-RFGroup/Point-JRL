import torch.nn as nn


class SimSiamLoss(nn.Module):
    """Negative cosine similarity loss for SimSiam."""
    def __init__(self):
        super(SimSiamLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, p1, p2, z1, z2):
        z1 = z1.detach()
        z2 = z2.detach()

        loss1 = -(self.criterion(p1, z2).mean())
        loss2 = -(self.criterion(p2, z1).mean())

        return 0.5 * (loss1 + loss2)


class HybridLoss(nn.Module):
    """Combined SimSiam SSL loss + cross-entropy classification loss."""
    def __init__(self, lambda_cls=1.0):
        super(HybridLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.ssl_loss = SimSiamLoss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, p1, p2, z1, z2, logits1, logits2, labels):
        loss_ssl = self.ssl_loss(p1, p2, z1, z2)

        loss_cls_1 = self.cls_loss(logits1, labels)
        loss_cls_2 = self.cls_loss(logits2, labels)
        loss_cls = 0.5 * (loss_cls_1 + loss_cls_2)

        total_loss = loss_ssl + self.lambda_cls * loss_cls

        loss_dict = {
            "total": total_loss,
            "ssl": loss_ssl,
            "cls": loss_cls
        }

        return total_loss, loss_dict


class LPALoss(nn.Module):
    """Triplet margin loss for Layout Pattern Analysis (LPA).
    L(z, p, n) = max(d(z, p) - d(z, n) + margin, 0)
    """
    def __init__(self, margin=1.5, p=2.0):
        super(LPALoss, self).__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p, reduction='mean')

    def forward(self, z_a, z_p, z_n):
        return self.criterion(z_a, z_p, z_n)


class GNNLoss(nn.Module):
    """Cross-entropy loss for GNN classification."""
    def __init__(self):
        super(GNNLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
