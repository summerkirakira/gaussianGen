import torch.nn as nn
import torch


class SkeletonPointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=64, dtype=torch.float32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 128, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, output_dim, dtype=dtype)
        )

    def forward(self, x):
        # x shape: (batch_size, num_points, input_dim)
        point_features = self.mlp(x)
        # 最大池化得到全局特征
        global_features = torch.max(point_features, dim=1)[0]
        return global_features