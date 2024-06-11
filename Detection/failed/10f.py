import pandas as pd
import os
import cv2

# Lire le fichier Excel
df = pd.read_csv('/storage8To/student_projects/foottracker/detectionData/train.csv')
video_path='/storage8To/student_projects/foottracker/detectionData/train'
# Fonction pour assigner les labels aux frames
def assign_labels_to_frames(df, video_id, duration, fps):
    labels = ['no_event'] * int(duration * fps)
    for _, row in df.iterrows():
        if row['id-video'] == video_id:
            start_frame = int(row['start'] * fps)
            end_frame = int(row['end'] * fps)
            labels[start_frame:end_frame] = [row['event']] * (end_frame - start_frame)
    return labels

# Ouvrir la vidéo
#video_path = '/storage8To/student_projects/foottracker/detectionData/train' # Remplacez par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo a été ouverte correctement
#if not cap.isOpened():
    #print(f"Could not open video file {video_path}")
    #exit()

# Boucle principale
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

labels = assign_labels_to_frames(df, video_id, duration, fps)

frames = []
current_batch = 0
success, frame = cap.read()
frame_idx = 0

while success:
    frames.append(frame)
    if len(frames) == 10:
        batch_labels = labels[frame_idx-10:frame_idx]
        if batch_labels:
            majority_label = max(set(batch_labels), key=batch_labels.count)
        else:
            majority_label = 'no_event'
        
        save_path = os.path.join(output_dir, majority_label, f"{video_id}_batch{current_batch}.jpg")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        cv2.imwrite(save_path, frames[len(frames)//2])
        
        frames = []
        current_batch += 1
    
    frame_idx += 1
    success, frame = cap.read()

cap.release()