import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BlockA(nn.Module):
    def __init__(self, in_channels=32, out_channels=64, kernel_size=3, stride=2):
        super(BlockA, self).__init__()
        self.main_path = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=1)
        )
        self.shortcut_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        main_out = self.main_path(x)
        shortcut_out = self.shortcut_path(x)
        return main_out + shortcut_out


class BlockB(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(BlockB, self).__init__()
        self.main_path = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(channels, channels, kernel_size, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SeparableConv2d(channels, channels, kernel_size, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SeparableConv2d(channels, channels, kernel_size, stride=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.main_path(x)


class LPA(nn.Module):
    """Layout Pattern Analysis (LPA) feature extractor.
    Uses separable convolutions with residual blocks for layout image embedding.
    """
    def __init__(self, input_channels=1):
        super(LPA, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            SeparableConv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            SeparableConv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(kernel_size=3, stride=2),

            BlockA(32, 64, kernel_size=3, stride=2),
            BlockA(64, 64, kernel_size=3, stride=2),

            BlockB(64, kernel_size=3),
            BlockB(64, kernel_size=3),

            BlockA(64, 128, kernel_size=3, stride=2),
            BlockA(128, 256, kernel_size=3, stride=2),

            SeparableConv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            SeparableConv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        embedding = self.embedding_head(features)
        return embedding
