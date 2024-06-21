import os
import pandas as pd
import wandb
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import random
import pprint
from model_training_danylo import label_to_int

# Initialize WandB and login
wandb.login()

# Choose searching method (random,grid,bayes)
sweep_config_grid = {
    'method': 'grid'
    }
# Define our goal
metric_grid = {
    'name': 'accuracy',
    'goal': 'maximize'   
    } 
# Assigning the chosen metrics as values to their corresponding key in metric dictionnary
sweep_config_grid['metric'] = metric_grid
#  Set parameters and their range to test
parameters_dict_grid = {
    'optimizer': {
        'values': ['adam']
        },
    'fc_layer_size': {
        'values': [64,256,512]
        },
    'alpha': {
          'values': [0.25,0.5,0.75,1]
        },
    'gamma': {
          'values': [0.5,1,1.5,2,2.5]
        },
    'batch_size': {
          'values': [16,32,64]
        },
    'num_workers': {
          'values': [1,2,3]
        },
}
# Add dictionnary to sweep_config
sweep_config_grid['parameters'] = parameters_dict_grid

sweep_id_grid = wandb.sweep(sweep_config_grid, project="Adjusting hyperparameters Final Grid")

# Define the Focal Loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets = targets.to('cuda')
        inputs = inputs.to('cuda')
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# VideoDataset class
class VideoDataset(Dataset):
    def __init__(self, video_directory, frame_count=10, transform=None, test=False):
        self.video_directory = video_directory
        self.frame_count = frame_count
        self.transform = transform
        self.test = test
        self.batches = []

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_path, label = self.batches[idx]
        frames = []
        for i in range(self.frame_count):
            frame_path = os.path.join(batch_path, f'frame{i:03}.png')
            frame = cv2.imread(frame_path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        if self.test:
            return frames, os.path.basename(batch_path)
        else:
            label_int = torch.tensor(label, dtype=torch.long)
            return frames, label_int

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

# Load your dataset
video_directory = '/storage8To/student_projects/foottracker/detectionData/outputjerem'
video_ids = os.listdir(video_directory)
batches = []
labels = []

for video_id in video_ids:
    video_path = os.path.join(video_directory, video_id)
    for batch_folder in os.listdir(video_path):
        batch_path = os.path.join(video_path, batch_folder)
        if os.path.isdir(batch_path):
            label = batch_folder.split('_')[-1]
            batches.append(batch_path)
            labels.append(label_to_int[label])

# Randomly select 1000 batches
random.seed(42)
selected_batches = random.sample(list(zip(batches, labels)), 1000)

# Split the selected batches into train and test sets with ratio 70% and 30%
train_size = int(0.7 * len(selected_batches))
train_batches = selected_batches[:train_size]
test_batches = selected_batches[train_size:]

# Create datasets
train_dataset = VideoDataset(
    video_directory=video_directory,
    frame_count=10,
    transform=transform,
    test=False
)
train_dataset.batches = train_batches

test_dataset = VideoDataset(
    video_directory=video_directory,
    frame_count=10,
    transform=transform,
    test=True
)
test_dataset.batches = test_batches

# Define the CNN model
class EventCNN(nn.Module):
    def __init__(self, fc_layer_size):
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
        self.fc1 = nn.Linear(64 * 30 * 30, fc_layer_size)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(fc_layer_size, 4)

    def forward(self, x):
        x = x.to('cuda')
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        
        model = EventCNN(config.fc_layer_size).to('cuda')
        criterion = FocalLoss(alpha=config.alpha, gamma=config.gamma, logits=False, reduce=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        loss_history = []
        for epoch in range(10):  # Number of epochs can be adjusted
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                inputs = inputs.view(-1, 3, 244, 244)
                outputs = model(inputs)
                outputs = outputs.view(inputs.size(0) // 10, 10, -1).mean(1)
                loss = criterion(outputs, labels)
                loss.backward()
                 # Log gradients and weights
                wandb.watch(model, criterion, log="all", log_freq=10)

                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    average_loss = running_loss / 10
                    accuracy = 100 * correct / total
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
                    loss_history.append(average_loss)
                    running_loss = 0.0
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            wandb.log({
                'epoch': epoch + 1,
                'loss': average_loss,
                'accuracy': accuracy,
                'epoch_duration': epoch_duration
            })

        print("Training complete")

# Define the main function to run the sweep
def main():
    wandb.agent(sweep_id_grid, train_model)

if __name__ == "__main__":
    main()

print('\n Processing...')