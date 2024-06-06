import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize
import cv2
import numpy as np
import os

class VideoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, frames_per_clip=10):
        self.video_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_id, timestamp, event_label, extra_info = self.video_data.iloc[idx]
        video_path = os.path.join(self.root_dir, f"{video_id}.mp4")  # Assuming video format is MP4
        frames, label = self.load_video(video_path, timestamp, event_label)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]  # Apply transformation to each frame

        return {'frames': torch.stack(frames), 'label': label}  # Stack frames to form a tensor

    def load_video(self, video_path, start_time, label):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Move the cap to start_time (in milliseconds)
        frames = []
        count = 0

        while count < self.frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
            count += 1

        cap.release()
        label = self.label_to_index(label)  # Convert label to index if needed

        return frames, label

    def label_to_index(self, label):
        # This method should convert labels to a consistent format, ideally to a tensor
        label_mapping = {'start': 0, 'end': 0, 'play': 1, 'challenge': 2, 'throwing': 3}
        return label_mapping.get(label, 0)  # Default to 'no_event'

# Define transformations (make sure to handle tensors properly)
transform = transforms.Compose([
    ToPILImage(),
    Resize((240, 320)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Usage
train_dataset = VideoDataset(csv_file='/storage8To/student_projects/foottracker/detectionData/train.csv', root_dir='storage8to', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
