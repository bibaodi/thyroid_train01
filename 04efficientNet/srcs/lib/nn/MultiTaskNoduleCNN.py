"""
MultiTaskNoduleCNN model definition
"""
import torch
import torch.nn as nn


class MultiTaskNoduleCNN(nn.Module):
    """EfficientNet-B0多任务模型"""
    def __init__(self, feature_mappings, dropout_rate=0.4):
        super().__init__()
        self.mappings = feature_mappings

        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone_features = 1280
        self.backbone.classifier = nn.Identity()

        self.shared_features = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.75)
        )

        self.heads = nn.ModuleDict()
        for task, mapping in self.mappings.items():
            num_classes = len(mapping)
            self.heads[task] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(shared)
        return outputs
