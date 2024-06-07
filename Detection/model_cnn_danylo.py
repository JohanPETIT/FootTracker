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

run = wandb.init() #Ajouter les graphes pour accurancy et loss 

label_to_int = {
    'start': 0,
    'play': 1,
    'end': 2,
    'challenge': 3,
    'throwin': 4,
    'no_event': 5
}

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
            label = self.dataframe.iloc[idx]['event']
            #if label not in label_to_int:
             #   raise ValueError(f"Unknown label: {label}")
            #label_int = label_to_int[label]
            #label_tensor = torch.tensor(label_int, dtype=torch.long)  # Convert label to tensor
            #return frames, label_tensor
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
        epoch_start_time = time.time()  # Début du suivi du temps
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Remettre les gradients à zéro

            # S'assurer que les inputs sont correctement dimensionnés pour le réseau
            inputs = inputs.view(-1, 3, 244, 244)

            # Calcul des prédictions du modèle
            outputs = model(inputs)

            # Assurez-vous que les outputs sont correctement dimensionnés pour CrossEntropyLoss
            # outputs devraient déjà être corrects [batch_size, num_classes]
            # Nous assumons que chaque batch a les labels correspondant aux prédictions agrégées des vidéos
            outputs = outputs.view(inputs.size(0) // 10, 10, -1).mean(1)

            # Préparation des labels pour CrossEntropyLoss
            labels = torch.tensor([label_to_int[label] for label in labels], dtype=torch.long)

            # Calcul de la perte
            loss = criterion(outputs, labels)
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids

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
    epoch_end_time = time.time()  # Fin du suivi du temps
    epoch_duration = epoch_end_time - epoch_start_time
    print(f'Epoch {epoch+1} duration: {epoch_duration:.2f} seconds')

        # Enregistrer les métriques sur wandb
    wandb.log({
            'epoch': epoch + 1,
            'loss': average_loss,
            'accuracy': accuracy,
            'epoch_duration': epoch_duration
        })
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
print('ok')