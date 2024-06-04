import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import time

class VideoDataset(Dataset):
    def __init__(self, video_directory, csv_file=None, transform=None, frame_count=10, specific_video=None):
       # Récupération de la liste des fichiers vidéo dans le répertoire spécifié
        self.video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
        # Cas d'une seule vidéo
        if specific_video:
            self.video_files = [os.path.join(video_directory, specific_video)]
        print("Video files:", self.video_files)
        
        # Initialisation
        self.transform = transform
        self.frame_count = frame_count
        self.events = {}
        self.event_names = {}
        
        if csv_file:
              # Extraction des colonnes des fichiers csv
            df = pd.read_csv(csv_file)
            if specific_video:
                video_id = os.path.splitext(specific_video)[0]
                df = df[df['video_id'] == video_id]
            # Création d'un dictionnaire d'événements(events qui contiendra les timings) et de noms d'événements par vidéo(event_names qui contiendra les labes "play",'touchin' etc)    
            self.events = df.groupby('video_id')['time'].apply(list).to_dict()
            self.event_names = df.groupby('video_id')['event'].apply(list).to_dict()
             # La clé est l'id de la vidéo, les valeurs sont soit une liste des timings pour events soit une liste des labels pour event_names
       
        #print("1Events:", self.events)
        #print("2Event names:", self.event_names)
    
    
    def __getitem__(self, idx):
        start_time = time.time()  # Start timer

        video_path = self.video_files[idx] # Ex : '/storage8To/student_projects/foottracker/detectionData/train/cfbe2e94_1.mp4'
        video_id = os.path.basename(video_path).split('.')[0] # Ex: 1606b0e6_0
        event_times = self.events.get(video_id, []) # Ex : [637.1115017409957, 638.3000000000002, 639.6115017409957]
        event_names = self.event_names.get(video_id, [])# Ex : ['start', 'play', 'end', 'start', 'play', 'play', 'end']

        #print(f"Processing video: {video_path}\nEvent times: {event_times}\nEvent names: {event_names}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = total_frames / fps
        cap.release()
        
         # Filtrage des temps d'événements valides (qui sont dans la durée de la vidéo)
        valid_event_times = [t for t in event_times if t < video_length]
        if not valid_event_times:
            return torch.empty(0, 3, 224, 224), -1, "", []
        
        # Segmentation de la vidéo autour des temps d'événements valides avec la durée 1
        segments = self.segment_video(video_path, valid_event_times,1)
     
      # Transformation des segments en utilisant les transformations spécifiées
        processed_segments = [self.process_segment(segment) for segment in segments if segment]
        if not processed_segments:
            return torch.empty(0, 3, 224, 224), -1, "", []

        print(f"4Processing time for video {video_path}: {time.time() - start_time} seconds")  # End timer
        
        return processed_segments[0], idx, video_id, event_times, event_names

    def segment_video(self, video_path, event_times, duration=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segments = []
        
        for start_time in event_times:
            start_frame = int(start_time * fps)
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

#  Les transformations pour les images
transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialisation de dataset
video_dataset = VideoDataset(
    video_directory='/storage8To/student_projects/foottracker/detectionData/train',
    csv_file='/storage8To/student_projects/foottracker/detectionData/train.csv',
    transform=transform,
    frame_count=10,
    #specific_video='1606b0e6_0.mp4'  # Specify the video file
)
# Initialisation de dataloader
video_loader = DataLoader(video_dataset, batch_size=4, shuffle=False, num_workers=4)

#To verify if the video_loader works correctly, start debugger

# Loop to demonstrate data loading
#for frames, idx, video_id, event_times, event_names in video_loader:
    #print("Processing batch...")
  #  if frames.nelement() == 0:
   #     print("Received an empty batch of frames.")
    #else:
     #   print(f"Event times: {event_times}, Event names: {event_names}")
    #break  

print("Total videos in dataset:", len(video_dataset))
print('print')