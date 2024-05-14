from outils import read_video, save_video
from trackers import Tracker

def main():
 # On lit la vidéo en entrée
 video_frames = read_video('Tracking/input_videos/video1.mp4')

 # On instancie le Tracker
 tracker = Tracker('Tracking/modeles/best.pt')

 tracks = tracker.get_objects_tracks(video_frames)


 # On enregistre la vidéo une fois les modifs apportées
 save_video(video_frames, 'Tracking/output_videos/video1.avi')

if __name__ == '__main__': # Fait fonctionner le main
 main()
