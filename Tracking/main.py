from outils import read_video, save_video

def main():
 # Read video
 video_frames = read_video('Tracking/input_videos/video1.mp4')

 # Save video
 save_video(video_frames, 'Tracking/output_videos/video1.avi')

if __name__ == '__main__':
 main()

