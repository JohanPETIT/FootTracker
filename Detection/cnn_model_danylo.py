import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from collections import defaultdict, Counter

label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}
# Inverse mapping to obtain label in string format from the numeric
int_to_label = {value: key for key, value in label_to_int.items()}
# Initialize class_counts as a defaultdict
class_counts = defaultdict(int)

# Define transforms with normalization to prepare input data(images).
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244, 244)),
    transforms.ToTensor(), # Normalization is handled internally.
])

device = torch.device('cuda')# Replace by 'cpu' if not enough GPU. 



# Define the Focal Loss class. Focal loss will handle the disbalance issue in dataset.
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device) # Weights of classes (4 weights for 4 classes).
        self.gamma = gamma # Reduce the relative loss for well-classified examples. More is gamma more training is dedicated for low classified exemples.
        self.logits = logits # Multiclass prediction if logits is False.
        self.reduce = reduce # Average the loss over the batch(Yes) or return the loss for each example separately.

    def forward(self, inputs, targets):
        targets = targets.to(device) #Move to GPU for faster treatement.
        inputs = inputs.to(device)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none') # Multiclass prediction.
        pt = torch.exp(-BCE_loss) # Calculate the probability of the prediction for each example.
        F_loss = self.alpha[targets] * (1-pt)**self.gamma * BCE_loss # Adjust the BCE loss by a factor (1-pt)**self.gamma (equation from internet).

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class EventCNN(nn.Module):
    def __init__(self):
        super(EventCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.to('cuda')
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 4)  # Fully connected layer for classification
    
    def forward(self, x):
        x = x.to(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor before the fully connected layer
        x = self.fc(x)
        return x
