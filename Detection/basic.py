import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
import wandb
import cv2
import random

# Label to integer mapping
label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}
# Inverse mapping
int_to_label = {value: key for key, value in label_to_int.items()}

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert PIL images to tensor
])

def load_dataset(root_dir, transform=None,test=False):
    data = []
    frames = []
    class_counts = defaultdict(int)
    for video_id in os.listdir(root_dir):
        video_path = os.path.join(root_dir, video_id)
        if os.path.isdir(video_path):
            for batch_id in os.listdir(video_path):
                label = batch_id.split('_')[-1]
                batch_path = os.path.join(video_path, batch_id)
                for i in range(10):
                 frame_path = os.path.join(batch_path, f'frame{i:03}.png')
                 frame = cv2.imread(frame_path)
                if transform:
                    frame = transform(frame)
                    frames.append(frame)
                data.append((frame_path, label_to_int[label]))
                class_counts[label_to_int[label]] += 1
            frames = torch.stack(frames)
        if test:
            return frames, os.path.basename(batch_path)
        else:
           return data, class_counts
# Create the dataset
dataset, class_counts = load_dataset(root_dir='/storage8To/student_projects/foottracker/detectionData/outputjerem', transform=transform)

dataloader= DataLoader(dataset, batch_size=1, shuffle=False)

max_class_count = max(class_counts.values())
class_weights = torch.tensor([max_class_count / class_counts[i] for i in range(len(label_to_int))], dtype=torch.float)
print('loose')

for batch_idx, (images, labels) in enumerate(dataloader):
    images = images.view(-1, 3, 40, 40)
    labels = labels.repeat_interleave(10)
    print(images.shape, labels)
    if batch_idx == 1:
        break
print('hola')

def focal_loss(inputs, targets, alpha=1, gamma=2, reduction='mean'):
    BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha[targets] * (1 - pt) ** gamma * BCE_loss
    
    if reduction == 'mean':
        return torch.mean(F_loss)
    elif reduction == 'sum':
        return torch.sum(F_loss)
    else:
        return F_loss

def simple_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 4, kernel_size=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return model

# Instantiate the model, loss function, and optimizer
model = simple_cnn()
print(model)
criterion = lambda inputs, targets: focal_loss(inputs, targets, alpha=class_weights, gamma=2.5, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 1 == 0:
                average_loss = running_loss / 1
                accuracy = 100 * correct / total
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
                loss_history.append(average_loss)
                accuracy_history.append(accuracy)

                running_loss = 0.0
                correct = 0.0
                total = 0.0

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
loss_history, accuracy_history = train_model(model, dataloader, criterion, optimizer, num_epochs=10)

plt.figure()
plt.plot(loss_history)
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss_history_conv3d.png')
wandb.log({"loss_history": wandb.Image('loss_history_conv3d.png')})

plt.figure()
plt.plot(accuracy_history)
plt.title('Accuracy History')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('accuracy_history_conv3d.png')
wandb.log({"accuracy_history": wandb.Image('accuracy_history_conv3d.png')})

def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[int_to_label[i] for i in range(len(label_to_int))])
    disp.plot()
    plt.savefig('confusion_matrix_conv3d.png')
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix_conv3d.png')})

# Testing
test_model(model, dataloader)

# Save the model
torch.save(model.state_dict(), 'conv3d_model.pth')
wandb.save('conv3d_model.pth')
print('Model saved successfully.')
