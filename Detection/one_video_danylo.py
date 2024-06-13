import cv2
import torch
from model_training_danylo import transform,label_to_int
from cnn_model_danylo import EventCNN

# ApplyModel class
class ApplyModel:
    def __init__(self, model_weights_path):
        # Load the trained model
        self.model = EventCNN().to('cuda')
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

    def extract_frames_every_n_seconds(self, video_path, n_seconds):
        # Initialization
        frames, predictions = [], []

        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return None

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the frame interval to capture a frame every n seconds
        frame_interval = int(fps * n_seconds)
        # Variables to keep track of current frame and time
        current_time = 0

        # Loop through the video frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            current_time += 1
            # Capture a frame every n seconds
            if current_time % frame_interval == 0:
                frame = transform(frame)
                # Apply the model to the frame
                with torch.no_grad():
                    output = self.model(frame.unsqueeze(0).to('cuda'))  # Add unsqueeze to add batch dimension
                    _, predicted = torch.max(output, 1)
                    predictions.append(predicted.item())
                    frames.append(frame)

        # Release everything if the process is finished
        cap.release()

        # Convert predictions to labels
        predicted_labels = [list(label_to_int.keys())[i] for i in predictions]
        return predicted_labels


# Path to the model weights
model_weights_path = '/home/foottracker/myenv/FootTracker/model_weights.pth'
# Path to the video
video_path = '/storage8To/student_projects/foottracker/detectionData/clips/573e61_5.mp4'
# Create an instance of ApplyModel
guesser = ApplyModel(model_weights_path)
# Extract frames and make predictions
predictions = guesser.extract_frames_every_n_seconds(video_path, n_seconds=3)
#print('Predictions:', predictions)
#print('Check')