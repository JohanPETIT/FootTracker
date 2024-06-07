import wandb
import torch
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
label_to_int = {
    'start': 0,
    'play': 1,
    'event': 2,
    'challenge': 3,
    'throwin': 4,
}
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
def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions, labels=list(label_to_int.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_int.keys()))
    disp.plot()
    plt.show()