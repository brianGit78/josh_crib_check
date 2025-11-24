import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class CribMobileNet(nn.Module):
    """Lightweight transfer-learning model for crib presence detection.

    The model uses a MobileNetV3-Small backbone with a custom classifier head
    that outputs a single logit for binary classification. We keep the
    backbone trainable so the network can adapt to lighting conditions and the
    specific camera setup while still benefiting from ImageNet pretraining.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SimpleCNN(nn.Module):
    """Legacy CNN retained for reference and potential experimentation."""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
