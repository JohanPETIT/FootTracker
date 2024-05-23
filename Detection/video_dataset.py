#VideoDataset handle loading videos, etracting segments around event times and process segments with transformation 
# Use Dataloader to load this data in batches -> ML model 
#re


import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd

class VideoDataset(Dataset):
    #Initialize the dataset, loading video and event times from CSV and group them by video ID 
    def __init__(self, video_directory, csv_file=None, transform=None, frame_count=30):
        # list of video in the specified directory 
        self.video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
        self.transform = transform
        self.frame_count = frame_count #stores the number of frames
        self.events = {} # to store event times per video 
        if csv_file: #read teh csv file and groups the event times 
            df = pd.read_csv(csv_file)
            self.events = df.groupby('video_id')['time'].apply(list).to_dict()

    #Get a video segment and label based on an index
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_id = os.path.basename(video_path).split('.')[0] 
        event_times = self.events.get(video_id, [])
        
        cap = cv2.VideoCapture(video_path) # reading the video 
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) # frames per second of the video 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = total_frames / fps # length of the video in seconds 
        cap.release()

        valid_event_times = [t for t in event_times if t < video_length] # list of event times that are within the length of the video 

        if not valid_event_times:
            return torch.empty(0, 3, 224, 224), -1

        segments = self.segment_video(video_path, valid_event_times) #list of frames extracted from the video
        processed_segments = [self.process_segment(segment) for segment in segments if segment] # list of transformed segments 

        if not processed_segments:
            return torch.empty(0, 3, 224, 224), -1

        return processed_segments[0], idx
    
    #extrat segments of frames from the video around specified event times 
    def segment_video(self, video_path, event_times, duration=5):
        cap = cv2.VideoCapture(video_path) #open video 
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) # fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # frames
        segments = []

        for start_time in event_times:
            start_frame = int(start_time * fps) #convert the start time to the corresponding frame 
            end_frame = start_frame + int(duration * fps) #same for the end time 
            #check if the start or end frame exceeds the total number of frames
            if start_frame >= total_frames or end_frame > total_frames:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #set current position in the video to strat_frame
            frames = []
            for _ in range(int(duration * fps)): #loops to read frames for the duration of the segment 
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if frames:
                segments.append(frames)
        cap.release()
        return segments

    #apply transformations to a segment of frames
    def process_segment(self, segment):
        processed_frames = [self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in segment] #convert each frame to RGB 
        if processed_frames:
            return torch.stack(processed_frames)
        else:
            return torch.empty(0, 3, 224, 224)
        
    #get the number of video files in the dataset
    def __len__(self):
        return len(self.video_files)

# Define transformations
transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset and DataLoader
video_dataset = VideoDataset(
    video_directory='/storage8To/student_projects/foottracker/detectionData/test',
    csv_file='/storage8To/student_projects/foottracker/detectionData/sample_submission.csv',
    transform=transform,
    frame_count=30
)

video_loader = DataLoader(video_dataset, batch_size=4, shuffle=True, num_workers=4)

# Loop to demonstrate data loading
for frames, labels in video_loader:
    if frames.nelement() == 0:
        print("Received an empty batch of frames.")
    else:
        print("Batch frames shape:", frames.shape)
        print("Batch labels:", labels)
    break  # Stop after first batch for demonstration

print("Total videos in dataset:", len(video_dataset))
