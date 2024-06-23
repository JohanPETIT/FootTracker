import cv2

# Function to play video
def play_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: Width={frame_width}, Height={frame_height}, FPS={fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video or failed to read the frame.")
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Press 'q' to quit the video display window
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'path_to_your_video.mp4' with the path to your video file
    video_path = '/storage8To/student_projects/foottracker/detectionData/clips/08fd33_1.mp4'
    play_video(video_path)



