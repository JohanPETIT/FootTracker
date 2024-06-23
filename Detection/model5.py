import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, ConfusionMatrixDisplay
import wandb
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import torch.nn.functional as F
from collections import defaultdict, Counter
from torchsummary import summary
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# DATASET AVAILABLE:

# Directory where data is located : /storage8To/student_projects/foottracker/detectionData/outputjerem

# labels(events) are 'noevent','play','challenge','throwin'.

# In this directory are 12 folders which is videos' id.
#1606b0e6_0  35bd9041_0  3c993bd2_0  407c5a9e_1  9a97dae4_1  cfbe2e94_1
#1606b0e6_1  35bd9041_1  3c993bd2_1  4ffd5986_0  cfbe2e94_0  ecf251d4_0
# In each folder are numerious batches.Each batch contains 10 frames(images) in png format.
# 
# exemple: Folder '1606b0e6_0'

# cd 1606b0e6_0
# pwd: /storage8To/student_projects/foottracker/detectionData/outputjerem/1606b0e6_0
# ls 
# batch00501_noevent batch00502_noevent batch00503_play batch00504_play batch00505_challenge batch00506_throwin  batch00507_noevent batch00508_noevent batch00509_noevent batch00510_noevent batch00511_play batch00512_noevent batch00513_noevent ….  batch08591_noevent
# 
# In folder batch00501_noevent(noevent is the label) (pwd: /storage8To/student_projects/foottracker/detectionData/outputjerem/1606b0e6_0/batch00501_noevent):
#frame000.png  frame001.png  frame002.png  frame003.png  frame004.png frame005.png  frame006.png  frame007.png  frame008.png  frame009.png

# Every batch contains exactly 10 frames. Names for images are the same for every batch (frame000.png, frame001.png ... frame009.png)
# Number of batches: noevent    play    challenge   throwin    total
#                    77582      13388   2293        578        93841


# Initialization general variables.

# Initialize WandB to log the results on the internet platform.
wandb.init(project="Adjusting hyperparameters Final Grid")  
# Replace by cuda if allocated space available
device = torch.device('cpu')
# Label to integer mapping. Each corresponding event(label) will be associate to its int class.
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
    'fc_layer_size': {
        'values': [64,128,256]
        },
    'gamma': {
          'values': [0.5,1,1.5,2,2.5]
        },
    'desired_number_of_batches': {
          'values': [200,300,600,500]
        },
        'oversample_rate_factor': {
            'values':[55000,40000,30000]
        }
}
# Add dictionnary to sweep_config
sweep_config_grid['parameters'] = parameters_dict_grid

sweep_id_grid = wandb.sweep(sweep_config_grid, project="Adjusting hyperparameters Final Grid")


# Initialize variables for VideoDataset data
video_directory = '/storage8To/student_projects/foottracker/detectionData/outputjerem'
video_ids = os.listdir(video_directory) # 12 videos' ids.
batches = []
labels = []
for video_id in video_ids: # Every video folder contains many batches(also folders) with 10 images(png format) per batch.
   # Next comments are the exemples of compilation.
    video_path = os.path.join(video_directory, video_id)# /storage8To/student_projects/foottracker/detectionData/outputjerem/1606b0e6_0
    for batch_folder in os.listdir(video_path): 
        batch_path = os.path.join(video_path, batch_folder)# /storage8To/student_projects/foottracker/detectionData/outputjerem/1606b0e6_0/batch00503_play 
        if os.path.isdir(batch_path):
            label = batch_folder.split('_')[-1] # play
            class_counts[label_to_int[label]] += 1 # class_counts [1,0,0,0] (because play is assigned to label 0).
            batches.append(batch_path) 
            labels.append(label_to_int[label])#*10

# VideoDataset class, which will download the data.
 
# Return tuple of 10 images converted to tensor and tensor label.
# Exemple :   10_images,label in train_dataset[0] <=> train_dataser[0] = (tensor([[[[0.2745, 0.1804, 0.1725,  ..., 0.4157, 0.3608, 0.3608],
#                                                                                   [0.2314, ... 0.1686,  ..., 0.1137, 0.1176, 0.1412]]]]),  
#                                                                         tensor(1))

# Define a function to load images from a batch path
def load_images_from_batch(batch_path):
    images = []
    for i in range(10):  # Assuming there are exactly 10 images named frame000.png, frame001.png, ..., frame009.png
        img_path = os.path.join(batch_path, f'frame{i:03d}.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = transform(image)
        images.append(image)
    return images


# Dataset and DataLoader from these image-label pairs
class CustomDataset(Dataset):
    def __init__(self, image_label_pairs):
        self.image_label_pairs = image_label_pairs
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, idx):
        image, label = self.image_label_pairs[idx]
        return image, label                                                    
    
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
    
class SimpleCNN(nn.Module):
    def __init__(self, fc_layer_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 61 * 61, fc_layer_size)  # Assuming input images are 244x244
        self.fc2 = nn.Linear(fc_layer_size, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 61 * 61)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare the data by balancing classes using SMOTE and creating data loaders.
def prepare_data(batches, labels, desired_number_of_batches, oversample_rate_factor):
    """
    Prepare the data by balancing classes using SMOTE and creating data loaders.
    """
    image_label_pairs = []

    for batch_path, label in zip(batches, labels):
        images = load_images_from_batch(batch_path)
        image_label_pairs.extend([(image, torch.tensor(label)) for image in images])

    # Splitting the data into training and validation sets.
    random.shuffle(image_label_pairs)
    train_size = int(0.8 * len(image_label_pairs))
    train_pairs = image_label_pairs[:train_size]
    val_pairs = image_label_pairs[train_size:]

    # Balance the training dataset using SMOTE
    train_images, train_labels = zip(*train_pairs)
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)

    # Use SMOTE to oversample the minority classes
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
    smote_images, smote_labels = smote.fit_resample(train_images.view(len(train_images), -1), train_labels)

    # Convert back to original shape
    smote_images = torch.tensor(smote_images).view(-1, 3, 244, 244)
    smote_labels = torch.tensor(smote_labels)

    # Creating balanced dataset pairs
    balanced_pairs = list(zip(smote_images, smote_labels))

    # Limit the number of batches to the desired number
    if len(balanced_pairs) > desired_number_of_batches * 10:
        balanced_pairs = balanced_pairs[:desired_number_of_batches * 10]

    # Creating the final datasets
    train_dataset = CustomDataset(balanced_pairs)
    val_dataset = CustomDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


# Training the model with hyperparameters from WandB.
def train_model(config=None, train_loader=None, val_loader=None, num_epochs=5):
    """
    Training the model with hyperparameters from WandB.
    """
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(config.fc_layer_size).to(device)
    class_weights = torch.tensor([0.24,0.1,0.3,0.43], dtype=torch.float).to(device)  # Placeholder for class weights
    criterion = FocalLoss(alpha=class_weights, gamma=config.gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        
        # Log metrics to WandB
        wandb.log({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'recall': recall,
        })
    
    return model

def train_model_wrapper(config=None):
    train_loader, val_loader = prepare_data(batches, labels, config.desired_number_of_batches, config.oversample_rate_factor)
    train_model(config=config, train_loader=train_loader, val_loader=val_loader)

wandb.agent(sweep_id_grid, function=train_model_wrapper)
print('ok')
print('bababoy')