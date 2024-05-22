from outils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from closest_player import ClosestPlayer

def main():
 # On lit la vidéo en entrée
 video_frames = read_video('Tracking/input_videos/video1.mp4')

 # On instancie le Tracker
 tracker = Tracker('Tracking/modeles/best.pt')

 # On applique le tracking
 tracks = tracker.get_objects_tracks(video_frames, read_from_file=True, file_path='Tracking/tracks_files/tracks.pkl')

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
 player_assigner = ClosestPlayer()
 team_ball_possession = [] # On veut associer à chaque frame de cette liste l'équipe qui a la balle

 for frame_num, player_track in enumerate(tracks['players']):
   ball_bbox = tracks['ball'][frame_num][1]['bbox'] # On récupère la bbox de la balle
   assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox) # On récupère le joueur le plus proche du ballon

 # On note quel joueur a le ballon s'il y en a un, et à quelle équipe il appartient
   if assigned_player != -1:
     tracks['players'][frame_num][assigned_player]['has_ball'] = True
     team_ball_possession.append(tracks['players'][frame_num][assigned_player]['team'])
   else:
     team_ball_possession.append(team_ball_possession[-1]) # S'il n'y a pas de joueur qui a le ballon, on prend la dernière équipe qui l'avait

 # On dessine les annotations
 output_video_frames = tracker.draw_annotations(video_frames,tracks)

 # On enregistre la vidéo une fois les modifs apportées
 save_video(output_video_frames, 'Tracking/output_videos/video1.avi')

if __name__ == '__main__': # Fait fonctionner le main
 main()
