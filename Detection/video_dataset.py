import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd

class VideoDataset(Dataset):
    def __init__(self, video_directory, csv_file=None, transform=None, frame_count=30):
        self.video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
        self.transform = transform
        self.frame_count = frame_count
        self.events = {}
        if csv_file:
            df = pd.read_csv(csv_file)
            self.events = df.groupby('video_id')['time'].apply(list).to_dict()

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_id = os.path.basename(video_path).split('.')[0]
        event_times = self.events.get(video_id, [])
        
        segments = self.segment_video(video_path, event_times)
        processed_segments = [self.process_segment(segment) for segment in segments if segment]

        if not processed_segments:
            return torch.empty(0, 3, 224, 224), -1

        return processed_segments[0]

    def segment_video(self, video_path, event_times, duration=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segments = []

        for start_time in event_times:
            start_frame = int(start_time * fps)
            end_frame = start_frame + int(duration * fps)
            if start_frame >= total_frames or end_frame > total_frames:
                print(f"Skipping start time {start_time} as it exceeds video length for {video_path}")
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            while len(frames) < int(duration * fps):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if frames:
                segments.append(frames)
        cap.release()
        return segments

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
