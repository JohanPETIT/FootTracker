
import torch.nn as nn
import torch.nn.functional as F
class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        # Spatial feature extraction
        self.spatial_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.spatial_bn1 = nn.BatchNorm2d(32)
        self.spatial_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.spatial_bn2 = nn.BatchNorm2d(64)
        self.spatial_pool = nn.MaxPool2d(2)

        # Temporal feature extraction
        self.temporal_conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(128)
        self.temporal_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.temporal_bn2 = nn.BatchNorm1d(128)
        self.temporal_pool = nn.MaxPool1d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 30 * 30, 128)  # Adjusted size
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Apply spatial convolutions
        x = self.spatial_pool(self.spatial_bn1(F.relu(self.spatial_conv1(x))))
        x = self.spatial_pool(self.spatial_bn2(F.relu(self.spatial_conv2(x))))

        # Apply temporal convolutions
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(-1, C, H, W)  # Merge batch and time dimensions
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Rearrange dimensions for 1D conv
        x = x.view(-1, C, H)  # Reshape for temporal convolutions
        x = self.temporal_pool(self.temporal_bn1(F.relu(self.temporal_conv1(x))))
        x = self.temporal_pool(self.temporal_bn2(F.relu(self.temporal_conv2(x))))

        # Flatten and apply fully connected layers
        x = x.view(batch_size, -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
