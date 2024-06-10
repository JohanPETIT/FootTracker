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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

# Initialize WandB
run = wandb.init() # Ajouter les graphes pour accurancy et loss sur un serveur

# Label to integer mapping
label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}

# Define the Focal Loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
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

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(batches, labels, test_size=0.3, random_state=42)

# Create datasets
train_dataset = VideoDataset(
    video_directory=video_directory,
    frame_count=10,
    transform=transform,
    test=False
)
train_dataset.batches = list(zip(X_train, Y_train))

test_dataset = VideoDataset(
    video_directory=video_directory,
    frame_count=10,
    transform=transform,
    test=True
)
test_dataset.batches = list(zip(X_test, Y_test))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


# Define the CNN model
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
        self.fc1 = nn.Linear(64 * 30 * 30, 100)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, focal loss, and optimizer
model = EventCNN()
criterion = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for model training
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
          try:  
            optimizer.zero_grad()
            inputs = inputs.view(-1, 3, 244, 244)
            outputs = model(inputs)
            outputs = outputs.view(inputs.size(0) // 10, 10, -1).mean(1)
            loss = criterion(outputs, labels)
            loss.backward()
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

          except ValueError as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
          
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1} duration: {epoch_duration:.2f} seconds')

        wandb.log({
            'epoch': epoch + 1,
            'loss': average_loss,
            'accuracy': accuracy,
            'epoch_duration': epoch_duration
        })
    print("Training complete")
    return loss_history

# Prediction function
def predict_model(model, test_loader): #Use only for test prediction, do not use train folder
    model.eval()
    predictions = []
    true_labels = [] #Initialize true labels for focal loss
    with torch.no_grad():
        for inputs, video_files in test_loader:
            inputs = inputs.view(-1, 3, 244, 244)
            outputs = model(inputs)
            outputs = outputs.view(inputs.size(0) // 10, 10, -1).mean(1)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(video_files)
    return predictions, true_labels 

# Save predictions to a file
def save_predictions(predictions, true_labels, output_file='predictions.csv'):
    df = pd.DataFrame(list(zip(true_labels, predictions)), columns=['video_file', 'predicted_label'])
    df.to_csv(output_file, index=False)

# Plot training history
def plot_training_history(loss_history):
    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

# Plot confusion matrix
#def plot_confusion_matrix(true_labels, predictions):
    #cm = confusion_matrix(true_labels, predictions, labels=list(label_to_int.values()))
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_int.keys()))
    #disp.plot()
    #plt.show()



# Train the model and make predictions
loss_history = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
predictions, true_labels = predict_model(model, test_loader)
#We transform real string labels from test set to numeric format
true_labels_numerical = [label_to_int[label.split('_')[-1]] for label in true_labels]
#We save predictions to the csv format
save_predictions(predictions, true_labels)
#We display the loss history
plot_training_history(loss_history)
#We initialise the confusion mattrix to be displayed on wandb
cm = confusion_matrix(true_labels_numerical, predictions)
class_names = ['play', 'noevent', 'challenge', 'throwin']
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels_numerical, preds=predictions, class_names=class_names)})

#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#disp.plot(cmap=plt.cm.Blues)
#plt.show()
#plot_confusion_matrix([label_to_int[label.split('_')[-1]] for label in true_labels], predictions)
print('ok')