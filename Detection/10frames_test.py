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

def get_event_label(events, current_time):
    index = bisect_right(events, (current_time,))
    if index:
        _, event = events[index - 1]
        return event
    return "no_event"

def map_time_to_phase(current_time, duration):
    if current_time < duration * 0.25:
        return 'start'
    elif current_time < duration * 0.50:
        return 'play'
    elif current_time < duration * 0.75:
        return 'challenge'
    else:
        return 'end'
    

def reclassify_phases(segment_labels):
    reclassified_labels = []
    previous_label = 'no_event'  # Commencez avec 'no_event' pour traiter correctement le premier segment

    for current_label in segment_labels:
        if previous_label == 'no_event':
            if current_label == 'no_event':
                reclassified_labels.append('no_event')  # Conservez 'no_event' si pas d'autres événements
            elif current_label == 'start':
                reclassified_labels.append('start')
            else:
                reclassified_labels.append(current_label)
        elif previous_label == 'start':
            if current_label in ['challenge', 'start']:
                reclassified_labels.append('start')
            elif current_label == 'play':
                reclassified_labels.append('play')
            else:
                reclassified_labels.append(current_label)
        elif previous_label == 'challenge':
            if current_label in ['challenge', 'end']:
                reclassified_labels.append('challenge')
            else:
                reclassified_labels.append(current_label)
        elif previous_label == 'play':
            if current_label in ['play', 'challenge', 'throwin']:
                reclassified_labels.append('play')
            else:
                reclassified_labels.append(current_label)
        else:
            reclassified_labels.append(current_label)
        previous_label = current_label

    return reclassified_labels




def segment_and_label_videos(video_dir, csv_path, output_dir, segment_length=10):
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
            phase = map_time_to_phase(current_time, duration)
            segment_labels.append(phase)
            
            if len(frames) == segment_length:
                # Re-classify phases based on rules
                reclassified_labels = reclassify_phases(segment_labels)
                
                # Count the reclassified phases
                phase_counts = Counter(reclassified_labels)
                majority_phase = phase_counts.most_common(1)[0][0]

                # Print the batch details
                print(f"Batch {batch_number} event counts: {phase_counts}")
                print(f"Batch {batch_number} majority label: {majority_phase}")

                # Save the segment
                batch_dir = os.path.join(output_dir, f"{video_id}_batch{batch_number}_{majority_phase}")
                if not os.path.exists(batch_dir):
                    os.makedirs(batch_dir)
                
                for i, seg_frame in enumerate(frames):
                    frame_filename = os.path.join(batch_dir, f"frame{i}.jpg")
                    cv2.imwrite(frame_filename, seg_frame)
                
                frames = []
                segment_labels = []
                batch_number += 1
            
            current_time += 1 / fps
            success, frame = cap.read()
        cap.release()
# Parameters
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
csv_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
output_dir = '/storage8To/student_projects/foottracker/detectionData/output'

# Segment and label the videos
segment_and_label_videos(video_dir, csv_path, output_dir)
