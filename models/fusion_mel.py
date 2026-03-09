import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import EarlyBranch, DeepEncoder

class DualMelFusion(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.branch_max = EarlyBranch("max")
        self.branch_avg = EarlyBranch("avg")

        self.fusion = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(128)

        self.deep = DeepEncoder()

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
        
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.4),
        
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
        
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        feat_max = self.branch_max(x)
        feat_avg = self.branch_avg(x)

        fused = torch.cat([feat_max, feat_avg], dim=1)
        
        fused = self.fusion_bn(self.fusion(fused))
        fused = F.relu(fused)

        deep_feat = self.deep(fused)

        return self.classifier(deep_feat)