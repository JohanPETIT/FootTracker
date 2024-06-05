import os
import cv2
import pandas as pd

# Paths
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
output_dir = '/storage8To/student_projects/foottracker/detectionData/output2'

# Create output directories if they don't exist
labels = ['play', 'challenge','throwin','no_event']
for label in labels:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Read CSV file
df = pd.read_csv(csv_path)

# Create dictionary of events
events_dict = {}
for _, row in df.iterrows():
    video_id = row['video_id']
    time = row['time']
    event = row['event']
    #if row['event'] == 'start'
    # event='no_event'
    #else:
    #event = row['event']
    if event not in ['start', 'end']:  # Ignore 'start' and 'end' labels
        if video_id not in events_dict:
            events_dict[video_id] = {}
        if event not in events_dict[video_id]:
            events_dict[video_id][event] = []
        events_dict[video_id][event].append(time)

# Function to extract frames
def extract_frames(video_path, times, output_subdir, label):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    for time in times:
        if frame_count == 10:  # Stop extracting frames if already extracted 10 frames
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        success, frame = cap.read()
        if not success:
            continue
        
        frame_time = time
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        success, frame = cap.read()
        if success:
            frame_name = f"frame{frame_count+1}_{label}.jpg"
            cv2.imwrite(os.path.join(output_subdir, frame_name), frame)
            frame_count += 1
            print('\n Extraction completed', frame_count)

    cap.release()
    print('\n Extraction completed') 

# Process videos
for video_id, events in events_dict.items():
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    for label, times in events.items():
        label_dir = os.path.join(output_dir, label)
        video_label_dir = os.path.join(label_dir, video_id)
        os.makedirs(video_label_dir, exist_ok=True)
        print('\n Created')
        extract_frames(video_path, times, video_label_dir, label)

print('ok')

