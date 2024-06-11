from model_training_danylo import TrainingModel,label_to_int
from focal_loss_danylo import FocalLoss
from cnn_model_danylo import EventCNN
from torchvision import transforms
import torch.optim as optim

def main_training():
    # Set parameters and initialize components
    video_directory = '/storage8To/student_projects/foottracker/detectionData/outputjerem'
    model = EventCNN()
    criterion = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((244, 244)),
        transforms.ToTensor()
    ])
    
    detection = TrainingModel(
        video_directory=video_directory,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        transform=transform,
        frame_count=10,
        batch_size=32,
        num_epochs=10
    )

    # Train the model
    loss_history = detection.train_model()

    # Predict on the test set
    predictions, true_labels = detection.predict_model()
    
    # Convert true labels to numerical format
    true_labels_numerical = [label_to_int[label.split('_')[-1]] for label in true_labels]

    # Save predictions to a CSV file
    detection.save_predictions(predictions, true_labels)

    # Plot training history
    detection.plot_training_history(loss_history)

    # Plot confusion matrix
    detection.plot_confusion_matrix(true_labels_numerical, predictions)

if __name__ == "__main__":
    main_training()