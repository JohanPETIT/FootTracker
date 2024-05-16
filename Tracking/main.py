from outils import read_video, save_video
from trackers import Tracker

def main():
 # On lit la vidéo en entrée
 video_frames = read_video('Tracking/input_videos/video1.mp4')

 # On instancie le Tracker
 tracker = Tracker('Tracking/modeles/best.pt')

 # On applique le tracking
 tracks = tracker.get_objects_tracks(video_frames, read_from_file=True, file_path='Tracking/tracks_files/tracks.pkl')

 # On dessine les annotations
 output_video_frames = tracker.draw_annotations(video_frames,tracks)

 # On enregistre la vidéo une fois les modifs apportées
 save_video(output_video_frames, 'Tracking/output_videos/video1.avi')

if __name__ == '__main__': # Fait fonctionner le main
 main()
