import os
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from focal_loss_danylo import FocalLoss
from cnn_model_danylo import EventCNN
from video_set_danylo import VideoDataset
from train_model_danylo import train_model, predict_model, save_predictions, plot_training_history, plot_confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
run = wandb.init()

# Label to integer mapping
label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}

# Define transform for frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

# Initialize the dataset
video_directory = '/storage8To/student_projects/foottracker/detectionData/outputjerem'
video_ids = os.listdir(video_directory)
batches = []
labels = []

# Extract labels and batches_paths
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

# Create proper dataset with batches to distribute train set
train_dataset = VideoDataset(
    video_directory=video_directory,
    frame_count=10,
    transform=transform,
    test=False
)
train_dataset.batches = list(zip(X_train, Y_train))

# Create proper dataset with batches to distribute test set
test_dataset = VideoDataset(
    video_directory=video_directory,
    frame_count=10,
    transform=transform,
    test=True
)
test_dataset.batches = list(zip(X_test, Y_test))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Initialize the model, focal loss, and optimizer
model = EventCNN().to(device)
criterion = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model and make predictions
loss_history = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
predictions, true_labels = predict_model(model, test_loader)
save_predictions(predictions, true_labels)
plot_training_history(loss_history)
plot_confusion_matrix([label_to_int[label.split('_')[-1]] for label in true_labels], predictions)
print('ok')
