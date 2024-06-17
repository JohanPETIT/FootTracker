import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict, Counter
# Label to integer mapping
label_to_int = {
    'play': 0,
    'noevent': 1,
    'challenge': 2,
    'throwin': 3,
}
# Inverse mapping
int_to_label = {value: key for key, value in label_to_int.items()}

# Initialize class_counts as a defaultdict
class_counts = defaultdict(int)

class FootTrackerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Iterate through each video folder
        for video_id in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_id)
            if os.path.isdir(video_path):
                # Iterate through each batch folder
                for batch_id in os.listdir(video_path):
                    batch_path = os.path.join(video_path, batch_id)
                    if os.path.isdir(batch_path):
                        # Extract label from batch_id
                        label = self.extract_label(batch_id)
                        # Iterate through each image in the batch folder
                        for image_name in sorted(os.listdir(batch_path)):
                            image_path = os.path.join(batch_path, image_name)
                            if image_name.endswith(('png', 'jpg', 'jpeg')):
                                self.data.append((image_path, label))

    def extract_label(self, batch_id):
        # Extract the label from the batch_id
        # Assuming the label is the part after the last underscore
        return batch_id.split('_')[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert PIL images to tensor
])

# Create the dataset
dataset = FootTrackerDataset(root_dir='/storage8To/student_projects/foottracker/detectionData/outputjerem', transform=transform)

# Create the DataLoader
batch_size = 32  # This is the number of images per batch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Example of how to use the DataLoader
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f'Batch {batch_idx}: {images.shape}, Labels: {labels}')
    # images will have shape (batch_size, C, H, W)
    # labels will be a list of labels for each image in the batch

    # Access the individual images in the batch
    for img_idx, image in enumerate(images):
        print(f'  Image {img_idx}: {image.shape},{ labels[img_idx]}')
        # Process each image here

    # If you want to break after the first batch for testing
    if batch_idx == 0:
        break
    inputs = images