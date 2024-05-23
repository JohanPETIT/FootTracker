import os
import cv2

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def analyze_videos(directory):
    total_frames = 0
    video_files = [file for file in os.listdir(directory) if file.endswith('.mp4')]
    for video_file in video_files:
        frame_count = count_frames(os.path.join(directory, video_file))
        total_frames += frame_count
    return len(video_files), total_frames

def get_total_size(directory):
    total_size = 0
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    return total_size



# Example usage
train_dir = '/storage8To/student_projects/foottracker/detectionData/train'
test_dir = '/storage8To/student_projects/foottracker/detectionData/test'
val_dir = '/storage8To/student_projects/foottracker/detectionData/clips'

train_videos, train_frames = analyze_videos(train_dir)
test_videos, test_frames = analyze_videos(test_dir)
val_videos, val_frames = analyze_videos(val_dir)
train_size = get_total_size(train_dir)
test_size = get_total_size(test_dir)
val_size = get_total_size(val_dir)

total_videos = train_videos + test_videos + val_videos
total_frames = train_frames + test_frames + val_frames
total_size = train_size + test_size + val_size

# Print the results
print(f"Training videos: {train_videos}, Frames: {train_frames}, Size: {train_size} bytes")
print(f"Testing videos: {test_videos}, Frames: {test_frames}, Size: {test_size} bytes")
print(f"Validation videos: {val_videos}, Frames: {val_frames}, Size: {val_size} bytes")
print(f"Total: Videos={total_videos}, Frames={total_frames}, Size: {total_size} bytes" )