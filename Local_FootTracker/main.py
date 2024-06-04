import sys
import os
import pickle
import streamlit as st
import moviepy.editor as moviepy
from foot_statistics import Possession, SpeedCalculator
from graphic_interface import Interface
from outils import get_tracks



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
        get_tracks(tracks_path, video_path_avi)

        # On convertit la vidéo en MP4
        clip = moviepy.VideoFileClip("output_videos/video1.avi")
        clip.write_videofile("output_videos/video1.mp4")

        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)
            f.close()


    # On instancie l'interface
    graphical_interface = Interface(tracks)
    graphical_interface.plot_page(video_path_mp4)


if __name__ == '__main__':
    main()

