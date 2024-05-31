import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import time

class VideoDataset(Dataset):
    def __init__(self, video_directory, csv_file=None, transform=None, frame_count=30, specific_video=None):
        self.video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
        if specific_video:
            self.video_files = [os.path.join(video_directory, specific_video)]
        print("Video files:", self.video_files)
        
        self.transform = transform
        self.frame_count = frame_count
        self.events = {}
        self.event_names = {}
        
        if csv_file:
            df = pd.read_csv(csv_file)
            if specific_video:
                video_id = os.path.splitext(specific_video)[0]
                df = df[df['video_id'] == video_id]
            self.events = df.groupby('video_id')['time'].apply(list).to_dict()
            self.event_names = df.groupby('video_id')['event'].apply(list).to_dict()
        #print("1Events:", self.events)
        #print("2Event names:", self.event_names)
    
    def segment_video(self, video_path, event_times, duration):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        locate = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segments = []
        for start in event_times[0:10]:
            start_frame = int(start * fps)
            end_frame = start_frame + int(duration * fps)
            if start_frame >= total_frames or end_frame > total_frames:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(int(duration * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if frames:
                segments.append(frames)
                locate+=1
                print(locate)
        cap.release()
        return segments
    
    def __getitem__(self,idx):
        start_time = time.time()  # Start timer

        video_path = self.video_files[idx]
        video_id = os.path.basename(video_path).split('.')[0]
        event_times = self.events.get(video_id, [])
        event_names = self.event_names.get(video_id, [])
        
        #print(f"Processing video: {video_path}\nEvent times: {event_times}\nEvent names: {event_names}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = total_frames / fps
        cap.release()
        
        valid_event_times = [t for t in event_times if t < video_length]
        if not valid_event_times:
            return torch.empty(0, 3, 224, 224), -1, "", []
        
        segments = self.segment_video(video_path, valid_event_times,5)
        processed_segments = [self.process_segment(segment) for segment in segments if segment]
        if not processed_segments:
            return torch.empty(0, 3, 224, 224), -1, "", []

        print(f"Processing time for video {video_path}: {time.time() - start_time} seconds")  # End timer
        return processed_segments, video_id, event_times, event_names
 
    

    def process_segment(self, segment):
        processed_frames = [self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in segment]
        if processed_frames:
            return torch.stack(processed_frames)
        else:
            return torch.empty(0, 3, 224, 224)

    def __len__(self):
        return len(self.video_files)

# Define transformations
transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_dataset = VideoDataset(
        video_directory='/storage8To/student_projects/foottracker/detectionData/train',
        csv_file='/storage8To/student_projects/foottracker/detectionData/train.csv',
        transform=transform,
        frame_count=30,
        specific_video='1606b0e6_0.mp4'  # Specify the video file
    )

video_loader = DataLoader(video_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Loop to demonstrate data loading
for frames, video_id, event_times, event_names in video_loader:
        print("Processing batch...")
        if len(frames)== 0:
            print("Received an empty batch of frames.")
        else:
            formatted_event_times = [f"[{float(t)}]" for t in event_times]
            print(f"Event times: {formatted_event_times}\nEvent names: {event_names}")
            break  # Stop after first batch for demonstration

print("Total videos in dataset:", len(video_dataset))