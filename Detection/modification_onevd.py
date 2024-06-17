import cv2
import torch
import matplotlib.pyplot as plt
from model_training_danylo import transform, label_to_int
from cnn_model_danylo import EventCNN
from model_tcnn import TCNN

# ApplyModel class
class ApplyModel:
    def __init__(self, model_weights_path):
        # Load the trained model
        self.model = TCNN().to('cuda')
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

    def extract_frame_batches_every_n_seconds(self, video_path, n_seconds, batch_size=10):
        # Initialization
        batches, predictions = [], []

        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return None, None

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the frame interval to capture a frame every n seconds
        frame_interval = int(fps * n_seconds)
        # Variables to keep track of current frame and time
        current_frame = 0
        frame_batch = []

        # Loop through the video frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            current_frame += 1
            # Capture frames every n seconds
            if current_frame % frame_interval == 0:
                transformed_frame = transform(frame)
                frame_batch.append(transformed_frame)

                # Once we've collected enough frames for a batch, make a prediction
                if len(frame_batch) == batch_size:
                    batch_tensor = torch.stack(frame_batch).to('cuda')
                    with torch.no_grad():
                        output = self.model(batch_tensor)
                        _, predicted = torch.max(output, 1)
                        predictions.extend(predicted.tolist())
                        batches.append(batch_tensor)
                    frame_batch = []  # Reset frame batch

        # Release the video capture
        cap.release()

        # Convert predictions to labels
        predicted_labels = [list(label_to_int.keys())[i] for i in predictions]
        return batches, predicted_labels

# Function to display a batch of frames
def display_batch(batch, predicted_label):
    fig, axes = plt.subplots(1, len(batch), figsize=(20, 5))
    for idx, frame in enumerate(batch):
        frame = frame.permute(1, 2, 0).cpu().numpy()  # Convert from tensor to numpy array and change dimensions
        axes[idx].imshow(frame)
        axes[idx].axis('off')
    plt.suptitle(f'Predicted Action: {predicted_label}', fontsize=16)
    plt.show()
    plt.savefig(f'illustration_12.png')
    

# Path to the model weights
model_weights_path = '/home/foottracker/myenv/FootTracker/model_weights2.pth'
# Path to the video
video_path = '/storage8To/student_projects/foottracker/detectionData/clips/573e61_5.mp4'
# Create an instance of ApplyModel
guesser = ApplyModel(model_weights_path)
# Extract batches of frames and make predictions
batches, predictions = guesser.extract_frame_batches_every_n_seconds(video_path, n_seconds=0.5, batch_size=10)

# Display the first batch as an example
if batches and predictions:
    display_batch(batches[1], predictions[1])
else:
    print("No batches to display.")

print('ok')
