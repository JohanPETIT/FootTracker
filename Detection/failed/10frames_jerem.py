#découpage en 10 frames sans labels 
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

# Iterate through each video_id in the dataframe
for index, row in df.iterrows():
    video_id = row['video_id']
    video_path = os.path.join(videos_path, f"{video_id}.mp4")
    output_path = os.path.join(output_base_path, str(video_id))
    
    create_directory(output_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_count = 0
    frames = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        if len(frames) == 10:
            batch_filename = os.path.join(output_path, f"batch{batch_count:03d}.avi")
            batch_count += 1
            
            # Write the batch to a video file
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(batch_filename, fourcc, 30.0, (width, height))
            
            for f in frames:
                out.write(f)
            
            out.release()
            frames = []
            
            print(f"Batch {batch_count:03d} of video_id {video_id} has been added")
    
    # Handle remaining frames if any
    if frames:
        batch_filename = os.path.join(output_path, f"batch{batch_count:03d}.avi")
        batch_count += 1
        
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(batch_filename, fourcc, 30.0, (width, height))
        
        for f in frames:
            out.write(f)
        
        out.release()
        print(f"Batch {batch_count:03d} of video_id {video_id} has been added")
    
    cap.release()

print("Processing complete.")
