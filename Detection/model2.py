import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
# Initialize WandB
wandb.init()  

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
        self.alpha = alpha.to('cuda')
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
    def __init__(self, video_directory,  transform=None, test=False):
        self.video_directory = video_directory
        self.transform = transform
        self.test = test
        self.batches = []

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_path, label = self.batches[idx]
        frames = []
        for i in range(10):
            frame_path = os.path.join(batch_path, f'frame{i:03}.png')
            frame = cv2.imread(frame_path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        if self.test:
            return frames, os.path.basename(batch_path)#ten times
        else:
            label_int = torch.tensor(label, dtype=torch.long)#.tentimes
            return frames,label_int

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((40, 40)),
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

# Determine the maximum class count 
max_class_count = max(class_counts.values())
# Calculate class weights
#class_weights = torch.tensor([max_class_count / class_counts[i] for i in range(len(label_to_int))], dtype=torch.float)


batch_labels_seperate = list(zip(batches, labels)) 
oversampled_train_batches = []
#for batch0, label0 in batch_labels_seperate:
#    count = class_counts[label0]
 #   oversample_factor = max_class_count // count
  #  oversampled_train_batches.extend([(batch0, label0)] * oversample_factor)

# Randomly select batches

for labeld in set(class_counts.keys()):
    class_batch_paths = [batchd for batchd in batch_labels_seperate if batchd[1] == labeld]
    oversample_factor = max_class_count - class_counts[labeld] - 40000
    oversampled_train_batches.extend(class_batch_paths)
    if oversample_factor >0:
        oversampled_train_batches.extend(random.choices(class_batch_paths, k=oversample_factor))


#selected_batches = list(zip(batches, labels))
random.shuffle(oversampled_train_batches)
random_selected_batches = random.sample(oversampled_train_batches, 100)
labels_random = [labelb for _, labelb in random_selected_batches]
labels_random.sort()
# Count the occurrences of each label
class_counts_random = Counter(labels_random)

class_weights = torch.tensor([ (1 / (class_counts_random[i]/ sum(list(class_counts_random))))  for i in range(len(label_to_int))], dtype=torch.float)

train_size = int(0.7 * len(random_selected_batches))
train_batches = random_selected_batches[:train_size]
test_batches = random_selected_batches[train_size:]

# Extract features from video batches
#def extract_features(batch_path):
 #   frames = []
  #  for i in range(10):
   #     frame_path = os.path.join(batch_path, f'frame{i:03}.png')
    #    frame = cv2.imread(frame_path)
     #   frame = cv2.resize(frame, (40, 40))
      #  frames.append(frame.flatten())
    #return np.concatenate(frames)
# Oversample minority classes by duplicating their batches

#train_features = [extract_features(batch_path) for batch_path, _ in train_batches]
#train_labels = [label for _, label in train_batches

# Create datasets
train_dataset = VideoDataset(
    video_directory=video_directory,
    transform=transform,
    test=False
)
train_dataset.batches = train_batches

test_dataset = VideoDataset(
    video_directory=video_directory,
    transform=transform,
    test=True
)
test_dataset.batches = test_batches


# Calculate class weights for Focal Loss
#class_counts_samp = dict(Counter([label for _, label in train_batches]))
#class_counts_samp_ordered = [value for key, value in sorted(class_counts_samp.items())]
#total_counts = len(train_batches)
#class_weights_samp_list = [(1/(count / total_counts)) for count in class_counts_samp_ordered] 
#class_weights = torch.tensor(class_weights_samp_list, dtype=torch.float32).to('cuda')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)
#for item_idx, (item,item1) in enumerate(train_loader):
    #print(f'Item {item_idx}: {item.shape} : {item1}')
    # Access the individual images in the batch
    #for batch_idx, batch in enumerate(item):
     #   print(f'  Batch {batch_idx}: {batch.shape}')
        # frame is a tensor with shape (num_images_per_batch, C, H, W)
        # You can process each frame here
        #for frame_idx,frame in enumerate(batch):
           #  print(f'  Frame {frame_idx}: {frame}')
    # If you want to break after the first batch for testing
    #if item_idx == 2:
        #break
print('tout')
class Conv3DModel(nn.Module):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3,3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Conv3d(in_channels=256, out_channels=4, kernel_size=1)

    def forward(self, x):
        x = x.to('cuda')
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)

model = Conv3DModel()
model.to('cuda')
#summary(model,)


criterion = FocalLoss(alpha=class_weights, gamma=2.5, logits=False, reduce=True)
optimizer = optim.Adam(model.parameters(),lr=0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust the learning rate

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

        for batch_idx, (inputs2, labels2) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs2= inputs2.permute(0, 2, 1, 3, 4)
            inputs2 = inputs2.to('cuda')
            labels2 = labels2.to('cuda')
            outputs2 = model(inputs2)
            loss = criterion(outputs2, labels2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs2, 1)

            total += labels2.size(0)
            correct += (predicted == labels2).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels2.cpu().numpy())

            if (batch_idx + 1) % 1 == 0:  # Print the statistics every 150th batch
                average_loss = running_loss / 1# divide by number of modulo!
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
    return loss_history, accuracy_history,all_labels,all_preds

# Test the model and evaluate accuracy and recall
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx,(inputs3, labels3) in enumerate(test_loader):
            inputs3 = inputs3.to('cuda')
            inputs3 = inputs3.permute(0, 2, 1, 3, 4)
            #labels = labels.to('cuda')
            outputs3 = model(inputs3)
            _, predicted3 = torch.max(outputs3, 1)
            labels_int3 =  [label_to_int[label_string.split('_')[-1]] for label_string in labels3]
            labels_tensor3 = torch.tensor(labels_int3)
            labels_tensor3 = labels_tensor3.to('cuda')
            total += len(labels_tensor3)
            correct += (predicted3 == labels_tensor3).sum().item()

            all_preds.extend(predicted3.cpu().numpy())
            all_labels.extend(labels_tensor3.cpu().numpy())

    accuracy = 100 * correct / total
    #recall = recall_score(all_labels, all_preds, average='macro')

    print(f'Test Accuracy: {accuracy:.2f}%')
    #print(f'Test Recall: {recall:.2f}')
    return all_labels,all_preds #The return is [0,1,1,1,1,0,2] and [0,0,1,1,1,1,0] for exemple

# Save predictions to a file
def save_predictions(all_labels, all_preds, output_file='predictions_event3.csv'):
    df = pd.DataFrame(list(zip(all_labels, all_preds)), columns=['video_file_label', 'predicted_label'])#Data is suppose to be 1.play .... 2.play....
    df.to_csv(output_file, index=False)

def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions, labels=list(label_to_int.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_int.keys()))
    disp.plot()
    plt.show()
    plt.savefig('confusion_matrix_conv3d1.png')
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix_conv3d1.png')})

    #conf_matrix = confusion_matrix(all_labels, all_preds)
    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[int_to_label[i] for i in range(len(label_to_int))])
    #disp.plot()
    #plt.savefig('confusion_matrix_conv3d.png')
   # wandb.log({"confusion_matrix": wandb.Image('confusion_matrix_conv3d.png')})

# Save loss, accuracy, and recall history as images
def plot_training_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss_history_conv3d1.png')
    wandb.log({"loss_history": wandb.Image('loss_history_conv3d1.png')})

def plot_accuracy(accuracy_history): 
   plt.figure()
   plt.plot(accuracy_history)
   plt.title('Accuracy History')
   plt.xlabel('Iteration')
   plt.ylabel('Accuracy')
   plt.savefig('accuracy_history_conv3d1.png')
   wandb.log({"accuracy_history": wandb.Image('accuracy_history_conv3d1.png')})

# Training
loss_history, accuracy_history,a,b = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
plot_training_loss(loss_history)
plot_accuracy(accuracy_history)

predicted_labels,real_labels = test_model(model, test_loader)
predicted_labels_strings = [int_to_label[label_int] for label_int in predicted_labels] 
real_labels_strings = [int_to_label[label_string] for label_string in real_labels]
save_predictions(real_labels_strings,predicted_labels_strings)
cm = confusion_matrix(real_labels_strings, predicted_labels_strings)
class_names = ['play', 'noevent', 'challenge', 'throwin']
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=real_labels, preds=predicted_labels, class_names=class_names)})
# Save the model
torch.save(model.state_dict(), 'conv3d2_model.pth')
wandb.save('conv3d1_model.pth')
print('Model saved successfully.')
print('joke')