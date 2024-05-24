import sys
import os
import paramiko
import pickle
from foot_statistics import Possession
from graphic_interface import plot_page


def main():

    # On récupère les tracks
    tracks = get_tracks()

    # On associe le joueur le plus proche de la balle et on calcule la possession de l'équipe
    possession_assigner = Possession()
    for frame_num, _ in enumerate(tracks['players']):
        team_1_possession, team_2_possession = possession_assigner.calculate_possession(tracks, frame_num)

    plot_page()
    

def get_tracks():
    
    # On se connecte en SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('localhost', username='foottracker', password='ft_2024@', port=2224)

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

