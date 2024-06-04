import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter

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
    return events

def get_event_label(events, current_time):
    for event_time, event in events:
        if current_time >= event_time:
            return event
    return "no_event"

def segment_and_label_videos(video_dir, csv_path, output_dir, segment_length=10, threshold=8):
    events = read_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_file in os.listdir(video_dir):
        if not video_file.endswith('.mp4'):
            continue
        
        video_path = os.path.join(video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        frames = []
        segment_labels = []
        success, frame = cap.read()
        current_time = 0
        frame_number = 0
        
        while success:
            frames.append(frame)
            event_label = get_event_label(events.get(video_id, []), current_time)
            segment_labels.append(event_label)
            
            if len(frames) == segment_length:
                # Determine the majority label
                label_count = Counter(segment_labels)
                if label_count.most_common(1)[0][1] >= threshold:
                    majority_label = label_count.most_common(1)[0][0]
                else:
                    majority_label = "no_event"
                
                # Save the segment
                segment_dir = os.path.join(output_dir, majority_label)
                if not os.path.exists(segment_dir):
                    os.makedirs(segment_dir)
                
                for i, seg_frame in enumerate(frames):
                    frame_filename = os.path.join(segment_dir, f"{video_id}_frame{frame_number + i}.jpg")
                    cv2.imwrite(frame_filename, seg_frame)
                
                frames = []
                segment_labels = []
                frame_number += segment_length
            
            current_time += 1 / fps
            success, frame = cap.read()
        
        cap.release()

# Parameters
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
output_dir = '/storage8To/student_projects/foottracker/detectionData/output'

# Segment and label the videos
segment_and_label_videos(video_dir, csv_path, output_dir)
