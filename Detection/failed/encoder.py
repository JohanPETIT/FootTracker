import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2) 
        return x

encoder = Encoder()


input_tensor = torch.randn(1, 3, 224, 224)


output_tensor = encoder(input_tensor)

print(output_tensor.shape)
