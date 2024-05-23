import cv2

def get_video_resolution(video_path):
    print(f"Attempting to open video file at: {video_path}")  # Debug: Show which file we're trying to open
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

video1_path = '/storage8To/student_projects/foottracker/detectionData/train/1606b0e6_0.mp4'
video1_resolution = get_video_resolution(video1_path)

print(f"Resolution of video 1: {video1_resolution}")