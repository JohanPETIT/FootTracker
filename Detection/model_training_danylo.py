import os
import random
from video_set_danylo import VideoDataset
from torch.utils.data import DataLoader
import time
import torch
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms

label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

class TrainingModel:
    def __init__(self, video_directory, model, criterion, optimizer, transform, frame_count=10, batch_size=32, num_epochs=10):
        self.video_directory = video_directory
        self.model = model.to('cuda')
        self.criterion = criterion
        self.optimizer = optimizer
        self.transform = transform
        self.frame_count = frame_count
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_loader, self.test_loader = self._prepare_data()

    def _prepare_data(self):
        video_ids = os.listdir(self.video_directory)
        batches = []
        labels = []

        for video_id in video_ids:
            video_path = os.path.join(self.video_directory, video_id)
            for batch_folder in os.listdir(video_path):
                batch_path = os.path.join(video_path, batch_folder)
                if os.path.isdir(batch_path):
                    label = batch_folder.split('_')[-1]
                    batches.append(batch_path)
                    labels.append(label_to_int[label])

        random.seed(42)  # Set seed for reproducibility
        selected_batches = random.sample(list(zip(batches, labels)), 1000)
        train_size = int(0.7 * len(selected_batches))
        train_batches = selected_batches[:train_size]
        test_batches = selected_batches[train_size:]

        train_dataset = VideoDataset(
            video_directory=self.video_directory,
            batches=train_batches,
            frame_count=self.frame_count,
            transform=self.transform,
            test=False
        )

        test_dataset = VideoDataset(
            video_directory=self.video_directory,
            batches=test_batches,
            frame_count=self.frame_count,
            transform=self.transform,
            test=True
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return train_loader, test_loader

    def train_model(self):
        self.model.train()
        loss_history = []
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                try:
                    self.optimizer.zero_grad()
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                    inputs = inputs.view(-1, 3, 244, 244)
                    outputs = self.model(inputs)
                    outputs = outputs.view(inputs.size(0) // 10, 10, -1).mean(1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if (batch_idx + 1) % 10 == 0:  # Print the statistics every 10th batch.
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

    def predict_model(self):
        self.model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, video_files in self.test_loader:
                inputs = inputs.view(-1, 3, 244, 244)
                inputs = inputs.to('cuda')
                outputs = self.model(inputs)
                outputs = outputs.view(inputs.size(0) // 10, 10, -1).mean(1)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
                true_labels.extend(video_files)
        return predictions, true_labels

    @staticmethod
    def save_predictions(predictions, true_labels, output_file='predictions.csv'):
        df = pd.DataFrame(list(zip(true_labels, predictions)), columns=['video_file', 'predicted_label'])
        df.to_csv(output_file, index=False)

    @staticmethod
    def plot_training_history(loss_history):
        plt.figure()
        plt.plot(loss_history, label='Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()
        

    @staticmethod
    def plot_confusion_matrix(true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions, labels=list(label_to_int.values()))
        class_names = ['play', 'noevent', 'challenge', 'throwin']
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=predictions, class_names=class_names)})