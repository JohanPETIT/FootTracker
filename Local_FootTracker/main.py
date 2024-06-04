import sys
import os
import paramiko
import pickle
import streamlit as st
import moviepy.editor as moviepy

from foot_statistics import Possession, SpeedCalculator
from graphic_interface import Interface
import config



def main():
    # On met la page en mode large par défault
    st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:")
    
    # On définit les paths pour les tracks et les vidéos
    tracks_path = 'tracks_files/tracks.pkl'
    video_path_avi = 'output_videos/video1.avi'
    video_path_mp4 = 'output_videos/video1.mp4'

    # On teste s'il existe déjà un fichier des tracks enregistré pour ne pas tout réexécuter
    if os.path.exists(tracks_path) and os.path.exists(video_path_mp4):
       # S'il existe, on l'ouvre et on charge les tracks
        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)
            f.close()

    else:
        # On récupère les tracks
        test = get_tracks(tracks_path, video_path_avi)

        # On convertit la vidéo en MP4
        clip = moviepy.VideoFileClip("output_videos/video1.avi")
        clip.write_videofile("output_videos/video1.mp4")

        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)
            f.close()
            print(True)


    # On instancie l'interface
    graphical_interface = Interface(tracks)
    graphical_interface.plot_page(video_path_mp4)

    

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

    return True

###########
def get_tracks(tracks_path, video_path):
    # Votre code existant ici
    return tracks  # Assurez-vous que cela renvoie les tracks calculés

if __name__ == '__main__':
    main()

