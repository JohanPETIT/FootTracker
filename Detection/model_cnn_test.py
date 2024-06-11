import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Classe VideoDataset pour charger les vidéos
class VideoDataset(Dataset):
    def __init__(self, video_directory, csv_file=None, frame_count=10, transform=None, test=False):
        if not test:
            self.dataframe = pd.read_csv(csv_file)
        self.video_directory = video_directory
        self.frame_count = frame_count
        self.transform = transform
        self.test = test

    def __len__(self):
        if self.test:
            return len(os.listdir(self.video_directory))  # Nombre de fichiers dans le répertoire de test
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.test:
            video_file = os.listdir(self.video_directory)[idx]
            video_path = os.path.join(self.video_directory, video_file)
        else:
            video_path = os.path.join(self.video_directory, self.dataframe.iloc[idx]['video_id'] + '.mp4')
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(self.frame_count):
            ret, frame = cap.read()
            if ret:
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        cap.release()
        frames = torch.stack(frames)
        
        if self.test:
            return frames, video_file
        else:
            label = self.dataframe.iloc[idx]['event_label']
            return frames, label

# Transformations pour les frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

# Création des DataLoader pour l'entraînement et le test
train_dataset = VideoDataset(
    video_directory='/storage8To/student_projects/foottracker/detectionData/train',
    csv_file='/storage8To/student_projects/foottracker/detectionData/train.csv',
    transform=transform
)

test_dataset = VideoDataset(
    video_directory='/storage8To/student_projects/foottracker/detectionData/test',
    transform=transform,
    test=True
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Définition du modèle CNN
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
        self.fc1 = nn.Linear(64 * 30 * 30, 100)  # Ajuster la taille en fonction de la taille de sortie de la dernière couche de pooling
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(100, 6)  # 6 événements possibles

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialisation du modèle, de la perte et de l'optimiseur
model = EventCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Mettre le modèle en mode entraînement
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Remettre les gradients à zéro
            outputs = model(inputs)  # Passer les inputs dans le modèle
            loss = criterion(outputs, labels)  # Calculer la perte
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()
            if batch_idx % 10 == 9:  # Afficher la perte toutes les 10 batchs
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss / 10}')
                loss_history.append(running_loss / 10)
                running_loss = 0.0
    print("Training complete")
    return loss_history

# Fonction de prédiction
def predict_model(model, test_loader):
    model.eval()  # Passer le modèle en mode évaluation
    predictions = []
    with torch.no_grad():
        for inputs, video_files in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for video_file, prediction in zip(video_files, predicted):
                predictions.append((video_file, prediction.item()))
    
    return predictions

# Sauvegarder les prédictions dans un fichier
def save_predictions(predictions, output_file='predictions.csv'):
    df = pd.DataFrame(predictions, columns=['video_file', 'predicted_label'])
    df.to_csv(output_file, index=False)

# Visualisation des résultats
def plot_training_history(loss_history):
    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

# Entraîner le modèle et effectuer les prédictions
loss_history = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
predictions = predict_model(model, test_loader)
save_predictions(predictions)
plot_training_history(loss_history)

