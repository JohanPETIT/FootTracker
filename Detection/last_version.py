import os
import wandb
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import random
from collections import defaultdict, Counter


# Initialize WandB
wandb.init()  # Add graphs for accuracy and loss on a server

# Label to integer mapping
label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}
# Inverse mapping
int_to_label = {value: key for key, value in label_to_int.items()}

# Initialize class_counts as a defaultdict
class_counts = defaultdict(int)

# Define the Focal Loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, logits=False, reduce=True):
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
        F_loss = self.alpha[targets] * (1-pt)**self.gamma * BCE_loss

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
            class_counts[label_to_int[label]] += 1
            batches.append(batch_path)
            labels.append(label_to_int[label])

# Determine the maximum class count for oversampling 
max_class_count = max(class_counts.values())

# Oversample minority classes 
oversampled_batches = [] 
oversampled_labels = [] 
for batch_path, label in zip(batches, labels): 
     oversampled_batches.append(batch_path) 
     oversampled_labels.append(label) 
     if class_counts[label] < max_class_count: 
        oversampled_batches.extend([batch_path] * (max_class_count // class_counts[label])) 
        oversampled_labels.extend([label] * (max_class_count // class_counts[label]))

# Randomly select 1000 batches
#random.seed(22)  # Set seed for reproducibility
#selected_batches = random.sample(list(zip(oversampled_batches, oversampled_labels)), 75000)
selected_batches = list(zip(oversampled_batches, oversampled_labels))
# Split the selected batches into train and test sets with ratio 70% and 30%
random.shuffle(selected_batches)
random_selected_batches = random.sample(selected_batches,70000)
train_size = int(0.7 * len(random_selected_batches))
train_batches = random_selected_batches[:train_size]
test_batches = random_selected_batches[train_size:]

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


# Calculate class weights for Focal Loss
class_counts_samp = dict(Counter([label for _, label in train_batches]))
class_counts_samp_ordered = [value for key, value in sorted(class_counts_samp.items())]
total_counts = len(train_batches)
class_weights_samp_list = [count/total_counts for count in class_counts_samp_ordered]
class_weights = torch.tensor(class_weights_samp_list, dtype=torch.float32).to('cuda')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 244 * 244, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 4)  # 6 classes

    def forward(self, x):
        x = x.view(-1, 3 * 244 * 244)  # Flatten the input
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

model = Net()
model.to('cuda')
criterion = FocalLoss(alpha=class_weights, gamma=2.5, logits=False, reduce=True)
optimizer = optim.Adam(model.parameters())

# Training loop for model training, returns loss history
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        all_preds = []
        all_labels = []
        for batch_idx, (inputs, labels_temporary) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to('cuda')
            inputs = inputs.view(-1, 3, 244, 244)
            outputs = model(inputs)
            labels_temporary = labels_temporary.to('cuda')
            labels_temporary = labels_temporary.repeat_interleave(10)
            loss = criterion(outputs, labels_temporary)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels_temporary.size(0)
            correct += (predicted == labels_temporary).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_temporary.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:  # Print the statistics every 5th batch.
                average_loss = running_loss / 10
                accuracy = 100 * correct / total
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
                loss_history.append(average_loss)
                accuracy_history.append(accuracy)
                running_loss = 0.0

        # Calculate and log accuracy per class
        conf_matrix = confusion_matrix(all_labels, all_preds)
        class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        for i, acc in enumerate(class_accuracy):
            wandb.log({f'class_{i}_accuracy': acc})
          
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1} duration: {epoch_duration:.2f} seconds')

        wandb.log({
            'epoch': epoch + 1,
            'loss': average_loss,
            'accuracy': accuracy,
            'epoch_duration': epoch_duration
        })
    print('Finished Training')
    return loss_history, accuracy_history

# Training
loss_history, accuracy_history = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Save loss and accuracy history as images
plt.figure()
plt.plot(loss_history)
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss_history_last.png')
wandb.log({"loss_history_last": wandb.Image('loss_history_last.png')})

plt.figure()
plt.plot(accuracy_history)
plt.title('Accuracy History')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('accuracy_history_last.png')
wandb.log({"accuracy_history": wandb.Image('accuracy_history_last.png')})

# Test the model and evaluate accuracy
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels_temporary in test_loader:
            inputs = inputs.to('cuda')
            inputs = inputs.view(-1, 3, 244, 244)
            labels_temporary = labels_temporary.to('cuda')
            labels_temporary = labels_temporary.repeat_interleave(10)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels_temporary.size(0)
            correct += (predicted == labels_temporary).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_temporary.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[int_to_label[i] for i in range(len(label_to_int))])
    disp.plot()
    plt.savefig('confusion_matrix_last.png')
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix_last.png')})

test_model(model, test_loader)

# Save the model
torch.save(model.state_dict(), 'last_model.pth')
wandb.save('last_model.pth')
print('cool')