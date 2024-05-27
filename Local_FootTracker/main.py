import sys
import os
import paramiko
import pickle
import streamlit as st
import moviepy.editor as moviepy

from foot_statistics import Possession, SpeedCalculator
from graphic_interface import plot_page
import config



def main():
    
    # On définit les paths pour les tracks et les vidéos
    tracks_path = 'tracks_files/tracks.pkl'
    video_path_avi = 'output_videos/video1.avi'
    video_path_mp4 = 'output_videos/video1.mp4'

    # On teste s'il existe déjà un fichier des tracks enregistré pour ne pas tout réexécuter
    if os.path.exists(tracks_path) and os.path.exists(video_path_mp4):
       # S'il existe, on l'ouvre et on charge les tracks
        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)
        return tracks
    else:
        # On récupère les tracks
        tracks = get_tracks(tracks_path, video_path_avi)

        # On convertit la vidéo en MP4
        clip = moviepy.VideoFileClip("output_videos/video1.avi")
        clip.write_videofile("output_videos/video1.mp4")

    # On associe le joueur le plus proche de la balle et on calcule la possession de l'équipe
    possession_assigner = Possession()
    for frame_num, _ in enumerate(tracks['players']):
        team_1_possession, team_2_possession = possession_assigner.calculate_possession(tracks, frame_num)

    # On calcule la vitesse et la distance parcourue des joueurs
    speed_calculator = SpeedCalculator()
    speed_calculator.add_speed_and_distance_to_tracks(tracks)
    #for frame_num, player_track in enumerate(tracks['players']):
     #   for player_id, track in player_track.items():
      #      speed = tracks['players'][frame_num][player_id]['speed']
       #     distance = tracks['players'][frame_num][player_id]['distance']
    plot_page(video_path_mp4)
    

def get_tracks(tracks_path, video_path):
    
    # Sinon, on se connecte en SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('localhost', username=config.username, password=config.password, port=config.port)

    # On exécute la partie tracking du code
    stdin, stdout, stderr= client.exec_command('/home/foottracker/myenv/bin/python3 /home/foottracker/myenv/FootTracker/Tracking/main.py')
    exit_status = stdout.channel.recv_exit_status()
    if(exit_status==0):
        # On récupère le fichier des tracks
        sftp = client.open_sftp()

        tracks_distant = '/home/foottracker/myenv/FootTracker/Tracking/tracks_files/tracks.pkl'

        sftp.get(tracks_distant, tracks_path)

        # On récupère la vidéo
        video_distant = '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video1.avi'

        sftp.get(video_distant, video_path)

        # On ferme la connexion
        sftp.close()

        client.close()

if __name__ == '__main__':
    main()

