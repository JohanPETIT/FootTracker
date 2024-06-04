import os
import cv2
import pandas as pd

# Chemins
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
output_dir = '/storage8To/student_projects/foottracker/detectionData/output2'
labels = ['play', 'touchin', 'challenge', 'no_event']


# Création des répertoires de sortie
def create_folders(labels,output_dir):
 for label in labels:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
print('\n Folders created succesfully')

#Lecture de fichier
def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    events = {}
    for _, row in df.iterrows():
        video_id = row['video_id']
        time = row['time']
        event = row['event']
        if video_id not in events:
            events[video_id] = []
        events[video_id].append((time, event))
    print(f"Events loaded: {events}")
    return events

# Fonction pour extraire les frames
def extract_frames(video_path, tuple, output_subdir, label):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for idx, (time, event) in enumerate(tuple):
        cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        success, frame = cap.read()
        if not success:
            continue
        
        for i in range(10):
            frame_time = time + (i / fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            success, frame = cap.read()
            if not success:
                break
            frame_name = f"frame{i+1}_{label}.jpg"
            print(frame_name)
            cv2.imwrite(os.path.join(output_subdir, frame_name), frame)

    cap.release()
    print('\n Extraction completed')

# Traitement des vidéos
create_folders(labels,output_dir)
for video_id, tuples in read_csv(csv_path).items():
    video_path = os.path.join(video_dir, f"{video_id}.mp4")  
    for time, label in tuples:
        label_dir = os.path.join(output_dir, label)
        video_label_dir = os.path.join(label_dir, video_id)
        os.makedirs(video_label_dir, exist_ok=True)
        extract_frames(video_path, [(time, label)], video_label_dir, label)


print('ok')
print('ll')