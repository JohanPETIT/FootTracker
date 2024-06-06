#découpage 10 frames en png 
import os
import pandas as pd
import cv2

# Paths
excel_file_path = '/storage8To/student_projects/foottracker/detectionData/train.csv'
videos_path = '/storage8To/student_projects/foottracker/detectionData/train'
output_base_path = '/storage8To/student_projects/foottracker/detectionData/outputjerem'

# Read the Excel file
df = pd.read_csv(excel_file_path)

# Function to create directory if it doesn't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

# Function to determine the label of a frame based on the time and events
def get_frame_label(frame_time, events):
    if not events:
        return "no_event"
    
    last_event = "no_event"
    for i, (event_time, event_label) in enumerate(events):
        if frame_time < event_time:
            break
        last_event = event_label
    
    return last_event

# Iterate through each video_id in the dataframe
for video_id in df['video_id'].unique():
    video_df = df[df['video_id'] == video_id]
    video_path = os.path.join(videos_path, f"{video_id}.mp4")
    output_path = os.path.join(output_base_path, str(video_id))
    
    create_directory(output_path)
    
    # Extract event times and labels
    events = [(row['time'], row['event']) for _, row in video_df.iterrows()]
    events = sorted(events, key=lambda x: x[0])  # Ensure events are sorted by time
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue
    
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_count = 0
    frames = []
    frame_times = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = i / frame_rate
        frames.append(frame)
        frame_times.append(frame_time)
        
        if len(frames) == 10:
            # Determine the label for the batch
            labels = [get_frame_label(time, events) for time in frame_times]
            batch_label = max(set(labels), key=labels.count)  # Majority label in the batch
            
            batch_dir = os.path.join(output_path, f"batch{batch_count:05d}_{batch_label}")
            create_directory(batch_dir)
            batch_count += 1
            
            # Save the frames as .png files
            for j, frame in enumerate(frames):
                frame_filename = os.path.join(batch_dir, f"frame{j:03d}.png")
                cv2.imwrite(frame_filename, frame)
            
            frames = []
            frame_times = []
            
            print(f"Batch {batch_count:05d} of video_id {video_id} with label {batch_label} has been added")
    
    # Handle remaining frames if any
    if frames:
        labels = [get_frame_label(time, events) for time in frame_times]
        batch_label = max(set(labels), key=labels.count)
        
        batch_dir = os.path.join(output_path, f"batch{batch_count:05d}_{batch_label}")
        create_directory(batch_dir)
        batch_count += 1
        
        for j, frame in enumerate(frames):
            frame_filename = os.path.join(batch_dir, f"frame{j:03d}.png")
            cv2.imwrite(frame_filename, frame)
        
        print(f"Batch {batch_count:05d} of video_id {video_id} with label {batch_label} has been added")
    
    cap.release()

print("Processing complete.")
