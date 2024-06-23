import cv2
import torch
from model_training_danylo import transform,label_to_int
from cnn_model_danylo import EventCNN
import pickle
import os, shutil


#from final import Conv3DModel,FocalLoss
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

# Open currents
with open('/home/foottracker/myenv/FootTracker/Tracking/current.pkl', 'rb') as f:
     current = pickle.load(f)
     print(current)
     f.close()
# Path to the model weights
model_weights_path = '/home/foottracker/myenv/FootTracker/model_weights2.pth'
# Path to the video
input_dir_path = '/home/foottracker/myenv/FootTracker/Tracking/input_videos/'
video_path = input_dir_path + current['video_path_mp4']
# Create an instance of ApplyModel
guesser = ApplyModel(model_weights_path)
# Extract frames and make predictions
predictions = guesser.extract_frames_every_n_seconds(video_path, n_seconds=1)
print(predictions)

with open('/home/foottracker/myenv/FootTracker/Detection/'+current['events_path'], 'wb') as f:
      pickle.dump(predictions,f)
      f.close()

# On clean les input
for filename in os.listdir(input_dir_path):
    file_path = os.path.join(input_dir_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        
