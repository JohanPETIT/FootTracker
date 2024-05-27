import sys
import os
import paramiko
import pickle
from foot_statistics import Possession, SpeedCalculator
from graphic_interface import plot_page
import config


def main():

    # On récupère les tracks
    tracks = get_tracks()

    # On associe le joueur le plus proche de la balle et on calcule la possession de l'équipe
    possession_assigner = Possession()
    for frame_num, _ in enumerate(tracks['players']):
        team_1_possession, team_2_possession = possession_assigner.calculate_possession(tracks, frame_num)

    # On calcule la vitesse et la distance parcourue des joueurs
    speed_calculator = SpeedCalculator()
    speed_calculator.add_speed_and_distance_to_tracks(tracks)
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            speed = tracks['players'][frame_num][player_id]['speed']
            distance = tracks['players'][frame_num][player_id]['distance']

    plot_page()
    

def get_tracks():
    
    # On se connecte en SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('localhost', username=config.username, password=config.password, port=config.port)

    # On exécute la partie tracking du code
    stdin, stdout, stderr= client.exec_command('/home/foottracker/myenv/bin/python3 /home/foottracker/myenv/FootTracker/Tracking/main.py')

    # On récupère le fichier des tracks
    sftp = client.open_sftp()

    tracks_distant = '/home/foottracker/myenv/FootTracker/Tracking/tracks_files/tracks.pkl'
    tracks_local = 'tracks_files/tracks.pkl'

    sftp.get(tracks_distant, tracks_local)

    # On récupère la vidéo
    video_distant = '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video1.avi'
    video_local = 'output_videos/video1.avi'

    sftp.get(video_distant, video_local)

    # On ferme la connexion
    sftp.close()

    client.close()

    # On écrit le fichier des tracks en local et on le retourne
    with open(tracks_local, 'rb') as f:
        tracks = pickle.load(f)
    return tracks


if __name__ == '__main__':
    main()

