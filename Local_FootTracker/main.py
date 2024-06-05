import sys
import os
import pickle
import streamlit as st
import moviepy.editor as moviepy
from foot_statistics import Possession, SpeedCalculator
from graphic_interface import Interface, home
from outils import get_tracks



def main():
    # On met la page en mode large par défault
    st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:")
    
    # On définit les paths pour les tracks et les vidéos
    remote_tracks_path =  '/home/foottracker/myenv/FootTracker/Tracking/tracks_files/tracks.pkl'
    remote_avi_path = '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video1.avi'
    local_tracks_path = 'tracks_files/tracks.pkl'
    output_local_path_avi = 'output_videos/video1.avi'
    output_local_path_mp4 = 'output_videos/video1.mp4'

    # On teste s'il existe déjà un fichier des tracks enregistré pour ne pas tout réexécuter
    if os.path.exists(local_tracks_path) and os.path.exists(output_local_path_mp4):
       # S'il existe, on l'ouvre et on charge les tracks
        with open(local_tracks_path, 'rb') as f:
            tracks = pickle.load(f)
            f.close()

    else:
        # On récupère les tracks
        get_tracks(remote_tracks_path, local_tracks_path, remote_avi_path, output_local_path_avi)

        # On convertit la vidéo en MP4
        clip = moviepy.VideoFileClip(output_local_path_avi)
        clip.write_videofile(output_local_path_mp4)

        with open(local_tracks_path, 'rb') as f:
            tracks = pickle.load(f)
            f.close()


    home()


if __name__ == '__main__':
    main()

