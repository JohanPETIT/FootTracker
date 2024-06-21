from outils import read_video, save_video, clean_directory
from trackers import Tracker
from team_assigner import TeamAssigner
import pickle
from camera_movement_estimator import CameraMovementEstimator
from perspective_transformer import PerspectiveTransformer
import moviepy.editor as moviepy
import os, shutil 
def main():
 
 input_videos_folder = '/home/foottracker/myenv/FootTracker/Tracking/input_videos'
 output_videos_folder = '/home/foottracker/myenv/FootTracker/Tracking/output_videos'
 tracks_folder = '/home/foottracker/myenv/FootTracker/Tracking/tracks_files'
 events_folder = '/home/foottracker/myenv/FootTracker/Detection/events_files'

 # On clean les calculs des vidéos précédentes
 clean_directory(output_videos_folder)
 clean_directory(tracks_folder)
 clean_directory(events_folder)
 
 # On ouvre les infos de la vidéo qu'on vient d'envoyer
 with open('/home/foottracker/myenv/FootTracker/Tracking/current.pkl', 'rb') as f:
     current = pickle.load(f)
     print(current)
     f.close()

 # On lit la vidéo en entrée
 video_frames = read_video('/home/foottracker/myenv/FootTracker/Tracking/input_videos/'+current['video_path_mp4'])

 # On instancie le Tracker
 tracker = Tracker('/home/foottracker/myenv/FootTracker/Tracking/modeles/best.pt')

 # On applique le tracking
 tracks = tracker.get_objects_tracks(video_frames, read_from_file=True, file_path='/home/foottracker/myenv/FootTracker/Tracking/'+current['tracks_path'])

 # On interpole les positions de la balle
 tracks["ball"] = tracker.interpolate_ball(tracks["ball"])

 # On récupère les positions des entités
 tracker.add_position_to_tracks(tracks)

 # On estime les mouvements de la caméra
 camera_movement_estimator = CameraMovementEstimator(video_frames[0])
 camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_file=True, file_path='/home/foottracker/myenv/FootTracker/Tracking/tracks_files/camera_movement_'+current['unique_code']+'.pkl')

 camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

 # On applique la transformation de perspective sur la partie du terrain qu'on voit toujours
 perspective_transformer = PerspectiveTransformer()
 perspective_transformer.add_transformed_positions_to_tracks(tracks)

 # On instancie un TeamAssigner
 team_assigner = TeamAssigner()

 # On récupère les couleurs des 2 équipes
 team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

 # Pour chaque joueur dans chaaque frame, on lui associe son équipe (et sa couleur respective) et on l'enregistre dans les tracks
 for frame_num, player_track in enumerate(tracks['players']):
  for player_id, track in player_track.items():
        team = team_assigner.assign_player_team(video_frames[frame_num], track['bbox'], player_id)
        tracks['players'][frame_num][player_id]['team'] = team
        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

# On enregistre les tracks avant de les envoyer au local
 with open('/home/foottracker/myenv/FootTracker/Tracking/'+current['tracks_path'], 'wb') as f:
      pickle.dump(tracks,f)
      f.close()

 # On dessine les annotations
 output_video_frames = tracker.draw_annotations(video_frames,tracks)

 # On enregistre la vidéo une fois les modifs apportées
 output_avi_path = '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video_'+current['unique_code']+'.avi'
 save_video(output_video_frames, output_avi_path)

 # On convertit la vidéo en MP4
 output_mp4_path = '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video_'+current['unique_code']+'.mp4'
 clip = moviepy.VideoFileClip(output_avi_path)
 clip.write_videofile(output_mp4_path)

 # On clean les input
 clean_directory(input_videos_folder)

# Fait fonctionner le main
if __name__ == '__main__':
 main()
