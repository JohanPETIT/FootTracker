from outils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from foot_statistics import Possession

def main():
 # On lit la vidéo en entrée
 video_frames = read_video('/home/foottracker/myenv/FootTracker/Tracking/input_videos/video1.mp4')

 # On instancie le Tracker
 tracker = Tracker('/home/foottracker/myenv/FootTracker/Tracking/modeles/best.pt')

 # On applique le tracking
 tracks = tracker.get_objects_tracks(video_frames, read_from_file=True, file_path='/home/foottracker/myenv/FootTracker/Tracking/tracks_files/tracks.pkl')

 # On interpole les positions de la balle
 tracks["ball"] = tracker.interpolate_ball(tracks["ball"])

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

 # On associe le joueur le plus proche de la balle et on calcule la possession de l'équipe
 possession_assigner = Possession()
 for frame_num, _ in enumerate(tracks['players']):
   team_1_possession, team_2_possession = possession_assigner.calculate_possession(tracks, frame_num)
 print(tracks)

 # On dessine les annotations
 output_video_frames = tracker.draw_annotations(video_frames,tracks)

 # On enregistre la vidéo une fois les modifs apportées
 save_video(output_video_frames, '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video1.avi')

if __name__ == '__main__': # Fait fonctionner le main
 main()
