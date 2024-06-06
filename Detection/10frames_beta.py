import os
import cv2
import pandas as pd
from collections import defaultdict

def read_events(csv_path):
    df = pd.read_csv(csv_path)
    events = defaultdict(list)
    for _, row in df.iterrows():
        events[row['video_id']].append((row['time'], row['event']))
    for key in events:
        events[key].sort()  # Ensure events are sorted by time
    return events

def assign_labels_to_frames(events, video_id, duration, fps):
    labels = ['no_event'] * int(duration * fps)  # Pre-initialize all labels to 'no_event'
    
    if video_id not in events:
        return labels
    
    video_events = events[video_id]
    current_event = 'no_event'
    
    start_frame = 0
    for i in range(len(video_events)):
        start_time, event = video_events[i]
        end_time = duration if i == len(video_events) - 1 else video_events[i + 1][0]
        end_frame = int(end_time * fps)
        start_frame = int(start_time * fps)
        
        for j in range(start_frame, end_frame):
            if event in ['start', 'end']:
                labels[j] = current_event
            else:
                labels[j] = event
                current_event = event
    
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
            # Extract labels of the current batch
            batch_labels = labels[frame_idx-9:frame_idx+1]  # inclusive of current frame
            if batch_labels:
                # Determine the majority label in the batch
                label_counts = defaultdict(int)
                for label in batch_labels:
                    label_counts[label] += 1
                
                majority_label = max(label_counts, key=label_counts.get)
                majority_count = label_counts[majority_label]
                
                # Determine directory based on the count threshold
                if majority_count >= 8:
                    save_dir = os.path.join(output_dir, majority_label)
                else:
                    save_dir = os.path.join(output_dir, 'no_event')
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Save the middle frame of the batch
                save_path = os.path.join(save_dir, f"{video_id}_batch{current_batch}.jpg")
                cv2.imwrite(save_path, frames[len(frames)//2])  # Save the central frame as representative
                
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

# Parameters
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
output_base_dir = '/storage8To/student_projects/foottracker/detectionData/output5'

main(video_dir, csv_path, output_base_dir)
