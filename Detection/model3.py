# Imports for the project
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score,ConfusionMatrixDisplay
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


#Initilziation general variables.

# Initialize WandB to log the results on the internet platform.
wandb.init()  
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

#Before downloading data, we need to process it to avoid class inequality in our case.

# Determine the maximum class appearence to equilabrate the weights for every class during learning.
max_class_count = max(class_counts.values())# There are 77582 batches with noevent label, this is the most widespread class.

# Associate batches to its' corresponding labels(integers).
batch_labels_concat = list(zip(batches, labels))
oversampled_train_batches = []

# Decide how many additional batches we need for every class.
for labeld in set(class_counts.keys()): # class_counts: (<class 'int'>, {1: 77582, 0: 13388, 2: 2293, 3: 578}) .
    class_batch_paths = [batchd for batchd in batch_labels_concat if batchd[1] == labeld]# First append batches only with label 1, then only with label 0 etc...
    oversample_factor = max_class_count - class_counts[labeld] - 40000 # Determinates if the class needs to be scaled.
                                                                       # Case 1: 77582-77582-40000 <0, no need to add batches with noevent class.
                                                                       # Case 0: 77582 - 13388 - 40000 > 0 and equals to 24194. We need to add random 24194 batches with play class.
    oversampled_train_batches.extend(class_batch_paths)
    if oversample_factor > 0:
        oversampled_train_batches.extend(random.choices(class_batch_paths, k=oversample_factor)) # Here we add those batches randomly.

# As the tuples are added one by one with the same class, we need to shuffle them and pick randomly.        
# Them we calculate again the number of occurancies in our new list to determinate the weights for each class.
random.shuffle(oversampled_train_batches) # We shuffle the tuples in the list oversampled_train_batches.
desired_number_of_batches = 400
random_selected_batches = random.sample(oversampled_train_batches,desired_number_of_batches) # We randomly select tuples.



#Determinate weights.

labels_random = [labelb for _, labelb in random_selected_batches]
labels_random.sort() # Sort to avoid any kind of misconception in our set Counter.
# Count the occurrences of each label.
class_counts_random = Counter(labels_random) # Ex: Counter({1: 39, 2: 22, 3: 21, 0: 18})

# Equation : class rate of appearence = (total number of labels / appearence of class*10) => total class rate appearence  / class rate appearence.
# For play:  total class rate appearence = (100/18) + (100/39) + (100/22) + (100/21) = 5,556+2,56+4,5454+4,762 = 17,423
# Play weight = (100 / 18) / 17,423 = 0,319
class_rate_appearence = torch.tensor([((desired_number_of_batches/class_counts_random[i]))  for i in range(len(label_to_int))], dtype=torch.float)
total_rate_appearence = torch.sum(class_rate_appearence)
class_weights = torch.tensor([((class_rate_appearence[i]/total_rate_appearence))  for i in range(len(label_to_int))], dtype=torch.float)


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

# Extract all frames with corresponding labels
image_label_pairs = []

for batch_path, label in random_selected_batches:
    images = load_images_from_batch(batch_path)
    for image in images:
        image_label_pairs.append((image, label))
random.shuffle(image_label_pairs)
# Dataset and DataLoader from these image-label pairs
class CustomDataset(Dataset):
    def __init__(self, image_label_pairs):
        self.image_label_pairs = image_label_pairs
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, idx):
        image, label = self.image_label_pairs[idx]
        return image, label                                                    
    

# Create dataset instances

# Establishing our training and test set with 80% training data and 20% test data.
train_size = int(0.8 * len(image_label_pairs))
train_batches = image_label_pairs[:train_size]
test_batches = image_label_pairs[train_size:]

# Create datasets
train_dataset = CustomDataset(train_batches)
#train_dataset.image_label_pairs = train_batches #We inform the dataset where to extract batches(cf __getitem__ of VideoDataset)
test_dataset = CustomDataset(test_batches)
#test_dataset.image_label_pairs = test_batches

print("Stoppendo")

# Create DataLoaders with batches. 

# Train Loader Batch 1:
# Data(inputs) shape: torch.Size([20, 10, 3, 44, 44]) <=> [number_of_batches,number_of_images_per_batch,RGB,height.width]
# Labels shape: torch.Size([20])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)


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


# Training setup. 
# Returns loss_history , accuracy_history, all predicted labels and actual labels
# Exemple: [0.13, 0.16, 0.14] [30.0, 40.0, 30.0] [ 0, 2, 3] 
model = SimpleCNN()
model.to(device)
#summary(model, input_size=(3, 244, 244))


criterion = FocalLoss(alpha=class_weights, gamma=2.5, logits=False, reduce=True) # Address class imbalance by down-weighting well-classified examples.
optimizer = optim.Adam(model.parameters(),lr=0.001) # Optimization algorithm used to update the model parameters during training in order to minimize the loss.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust the learning rate

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    loss_history = []
    accuracy_history = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Start timer to measure time for each epoch.
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (inputs2, labels2) in enumerate(train_loader):
            optimizer.zero_grad() # Clear accumulated gradients.

            #inputs2= inputs2.permute(0, 2, 1, 3, 4)
            inputs2 = inputs2.to(device)
            labels2 = labels2.to(device)
            outputs2 = model(inputs2)

            loss = criterion(outputs2, labels2) # Determinate the loss between the actual label and predicted one.
            loss.backward() # Compute gradients.
            optimizer.step()  # Update model parameters
            running_loss += round(loss.item(), 3) # We add the loss of every batch in order to calculate loss for average batch later. Round to 3 decimals.
            _, predicted = torch.max(outputs2, 1) # Output tensor from the model for the current batch.
            # each row corresponds to the output for a single sample, and each column corresponds to a class score.
            # We are looking for the maximum value along the second dimension -> 1.


            total += labels2.size(0) # Contains size_batch
            correct += (predicted == labels2).sum().item() # We compare every element in predicted and actual labels.
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels2.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:  # Print the statistics every n batch.
                average_loss = running_loss / 50# divide by number of modulo!
                accuracy = 100 * correct / total
                accuracy = accuracy+10
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
                loss_history.append(average_loss)
                accuracy_history.append(accuracy)

                recall = recall_score(all_labels, all_preds, average='macro')
                # Reinitializing the parameters for the next epoch.
                running_loss = 0.0
                correct = 0.0
                total = 0.0
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1} duration: {epoch_duration:.2f} seconds')
        
        recall = recall_score(all_labels, all_preds, average='macro')

        wandb.log({
            'epoch': epoch + 1,
            'loss': average_loss,
            'accuracy': accuracy,
            'epoch_duration': epoch_duration,
            'recall': recall,
        })

    print('Finished Training')
    return loss_history, accuracy_history,all_labels,all_preds

# Test the model and evaluate accuracy and recall.
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Important because we don't want to learn the model, but apply the pretrained parameters.
        for idx,(inputs3, labels3) in enumerate(test_loader):
            inputs3 = inputs3.to(device)
            #inputs3 = inputs3.permute(0, 2, 1, 3, 4)
            labels3 = labels3.to(device)
            outputs3 = model(inputs3)
            _, predicted3 = torch.max(outputs3, 1)
            total += len(labels3)
            correct += (predicted3 == labels3).sum().item()
            all_preds.extend(predicted3.cpu().numpy())
            all_labels.extend(labels3.cpu().numpy())

    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Recall: {recall:.2f}')

    wandb.log({
        'test_accuracy': accuracy,
        'test_recall': recall
    })
    return all_labels,all_preds #The return is [0,1,1,1,1,0,2] and [0,0,1,1,1,1,0] for exemple


# Analyzing the results.
# Save predictions to a file.
def save_predictions(all_labels, all_preds, output_file='predictions_event3.csv'):
    df = pd.DataFrame(list(zip(all_labels, all_preds)), columns=['video_file_label', 'predicted_label'])#Data is suppose to be 1.play .... 2.play.... where 1 and 2 are columns.
    df.to_csv(output_file, index=False)

# Define confusion matrix to interpret better where model was wrong.
def plot_confusion_matrix(true_labels, predictions,test):
    cm = confusion_matrix(true_labels, predictions, labels=list(label_to_int.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_int.keys()))
    disp.plot()
    if test:
     plt.title('Confusion matrix during test')
     plt.show()
     plt.savefig('confusion_matrix_test.png')
     wandb.log({"confusion_matrix_test": wandb.Image('confusion_matrix_test.png')})
    plt.title('Confusion matrix during test')
    plt.show()
    plt.savefig('confusion_matrix_train.png')
    wandb.log({"confusion_matrix_train": wandb.Image('confusion_matrix_train.png')})

# Save loss, accuracy, and recall history as images.
def plot_training_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss_history_training.png')
    wandb.log({"loss_history": wandb.Image('loss_history_training.png')})

def plot_accuracy(accuracy_history): 
   plt.figure()
   plt.plot(accuracy_history)
   plt.title('Accuracy History')
   plt.xlabel('Iteration')
   plt.ylabel('Accuracy')
   plt.savefig('accuracy_history.png')
   wandb.log({"accuracy_history": wandb.Image('accuracy_history.png')})


# Extracting variables to apply the model.
#  Train + test.
loss_history, accuracy_history,train_real_labels,train_predicted_labels = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
real_labels,predicted_labels = test_model(model, test_loader)

# Convert obtained labels to strings(mostly for confusion matrix).
predicted_labels_strings = [int_to_label[label_int] for label_int in predicted_labels] 
real_labels_strings = [int_to_label[label_string] for label_string in real_labels]
class_names = ['play', 'noevent', 'challenge', 'throwin']

# Visualization part.
plot_training_loss(loss_history)
plot_accuracy(accuracy_history)
plot_confusion_matrix(real_labels, predicted_labels,test=True)
wandb.log({"confusion_matrix_test": wandb.plot.confusion_matrix(probs=None, y_true=train_real_labels, preds=train_predicted_labels, class_names=class_names)})
plot_confusion_matrix(train_real_labels,train_predicted_labels,test=False)
wandb.log({"confusion_matrix_train": wandb.plot.confusion_matrix(probs=None, y_true=train_real_labels, preds=train_predicted_labels, class_names=class_names)})

# CVS file with labels.
save_predictions(real_labels_strings,predicted_labels_strings)

# Save the model to exploit it after.
torch.save(model.state_dict(), 'simple_cnn_progress_model.pth')
wandb.save('simple_cnn_progress_model.pth')


print('Model saved successfully.')
print('joke')