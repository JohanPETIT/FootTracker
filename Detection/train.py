from video_dataset import VideoDataset, transform
from torch.utils.data import DataLoader

# Setup paths
video_dir = '/path/to/videos'
csv_file = '/path/to/labels.csv'

# Initialize dataset
dataset = VideoDataset(video_dir, csv_file=csv_file, transform=transform)

# Create data loader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Training loop
for data in data_loader:
    frames, labels = data


