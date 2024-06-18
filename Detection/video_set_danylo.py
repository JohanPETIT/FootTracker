import os
import torch
import cv2
from torch.utils.data import Dataset

# VideoDataset class
class VideoDataset(Dataset):
    def __init__(self, video_directory, batches, frame_count=10, transform=None, test=False):
        self.video_directory = video_directory
        self.frame_count = frame_count
        self.transform = transform
        self.test = test
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_path, label = self.batches[idx]
        frames = []
        for i in range(self.frame_count):
            frame_path = os.path.join(batch_path, f'frame{i:03}.png')
            frame = cv2.imread(frame_path)
            
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        if self.test:
            return frames, os.path.basename(batch_path)
        else:
            label_int = torch.tensor(label, dtype=torch.long)
            return frames, label_int