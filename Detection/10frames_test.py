import os
import cv2
import pandas as pd
from collections import Counter
from bisect import bisect_right

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

#def get_event_label(events, current_time):
 #   for event_time, event in events:
  #      if current_time >= event_time:
  #          return event
   # return "no_event"

def get_event_label(events, current_time):
    index = bisect_right(events, (current_time,))
    if index:
        _, event = events[index - 1]
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
        batch_number = 0
        
        while success:
            frames.append(frame)
            event_label = get_event_label(events.get(video_id, []), current_time)
            segment_labels.append(event_label)
            
            if len(frames) == segment_length:
                # Count the events
                event_counts = Counter(segment_labels)
                print(f"Batch {batch_number} event counts: {event_counts}")
                
                # Determine the majority label
                if event_counts['throwin'] >= threshold:
                    majority_label = 'throwin'
                elif event_counts['challenge'] >= threshold:
                    majority_label = 'challenge'
                elif event_counts['play'] >= threshold:
                    majority_label = 'play'
                else:
                    majority_label = 'no_event'
                
                print(f"Batch {batch_number} majority label: {majority_label}")

                # Save the segment
                batch_dir = os.path.join(output_dir, f"batch{batch_number}_{majority_label}")
                if not os.path.exists(batch_dir):
                    os.makedirs(batch_dir)
                
                for i, seg_frame in enumerate(frames):
                    frame_filename = os.path.join(batch_dir, f"{video_id}_frame{i}.jpg")
                    cv2.imwrite(frame_filename, seg_frame)
                
                frames = []
                segment_labels = []
                batch_number += 1
            
            current_time += 1 / fps
            success, frame = cap.read()
        cap.release()
        # Uncomment the following line to process only the first video
        # break

# Parameters
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
output_dir = '/storage8To/student_projects/foottracker/detectionData/output'

# Segment and label the videos
segment_and_label_videos(video_dir, csv_path, output_dir)
