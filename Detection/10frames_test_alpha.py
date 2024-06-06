import os
import cv2
import pandas as pd
from collections import defaultdict
from bisect import bisect_right

def read_events(csv_path):
    df = pd.read_csv(csv_path)
    events = defaultdict(list)
    for _, row in df.iterrows():
        events[row['video_id']].append((row['time'], row['event']))
    for key in events:
        events[key].sort()  # S'assurer que les événements sont triés par temps
    return events

def assign_labels_to_frames(events, video_id, duration, fps):
    labels = ['no_event'] * int(duration * fps)  # Pré-initialiser tous les labels à 'no_event'
    current_event = 'no_event'
    
    for i in range(len(events[video_id])):
        start_time, event = events[video_id][i]
        end_time = duration if i == len(events[video_id]) - 1 else events[video_id][i + 1][0]
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        for j in range(start_frame, end_frame):
            labels[j] = event if event in ['challenge', 'throwin', 'play'] else current_event

    return labels

def process_video(video_path, events, output_dir):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    labels = assign_labels_to_frames(events, video_id, duration, fps)
    
    frames = []
    current_batch = 0
    success, frame = cap.read()
    frame_idx = 0
    
    while success:
        frames.append(frame)
        if len(frames) == 10:
            # Extraire les labels du batch actuel
            batch_labels = labels[frame_idx-10:frame_idx]
            # Vérifier si la liste des labels est vide
            if batch_labels:
                majority_label = max(set(batch_labels), key=batch_labels.count)
            else:
                majority_label = 'no_event'  # ou toute autre gestion de cas sans labels
            
            save_path = os.path.join(output_dir, majority_label, f"{video_id}_batch{current_batch}.jpg")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            
            # Enregistrer l'image moyenne du batch ou une image représentative
            cv2.imwrite(save_path, frames[len(frames)//2])  # Sauvegarde de la frame centrale comme représentative
            
            frames = []
            current_batch += 1
        
        frame_idx += 1
        success, frame = cap.read()

    cap.release()


def main(video_dir, csv_path, output_base_dir):
    events = read_events(csv_path)
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            process_video(video_path, events, output_base_dir)

# Paramètres
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
output_base_dir = '/storage8To/student_projects/foottracker/detectionData/output'

main(video_dir, csv_path, output_base_dir)
