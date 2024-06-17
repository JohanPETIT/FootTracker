import torchvision.models as models
import torch.nn as nn

class EventResNet(nn.Module):
    def __init__(self):
        super(EventResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4)

    def forward(self, x):
        x = x.to('cuda')
        x = self.resnet(x)
        return x
