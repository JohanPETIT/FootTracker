import streamlit as st
import numpy as np
from plotly_football_pitch import make_pitch_figure, PitchDimensions, SingleColourBackground, add_heatmap
from foot_statistics import Possession, SpeedCalculator, BallHeatmap

#Empêche de réexécuter tout le code dès qu'on clique sur qqc
@st.experimental_fragment
class Interface():
    def __init__(self, tracks):
        self.tracks = tracks

    
    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    # Dessine la page
    @st.experimental_fragment
    def plot_page(self,video_path):

        print('hello')
        # On charge la vidéo
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()

        # On initialise 2 colonnes pour avoir les stats à coté de la vidéo
        col1, col2 = st.columns(2)

        # Vidéo colonne gauche
        with col1:
            st.video(video_bytes, autoplay=True, muted=True)

        # Stats colonne droite
        with col2:
            # On dessine le truc pour sélectionner
            option = st.selectbox("Quelle statistique vous intéresse ?", ("Possession", "Position du ballon", "Vitesse des joueurs", "Autre"), index=None, placeholder="Choisissez une option !")
            
            # Possession 
            if(option == "Possession"):
                self.plot_possession()

            # Position du ballon 
            if(option == "Position du ballon"):
                self.plot_ball_heatmap()

            # Vitesse des joueurs
            if(option == "Vitesse des joueurs"):
                pass
                #self.plot_speeds()
                 

    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    @st.experimental_fragment
    def plot_possession(self):

        # On associe le joueur le plus proche de la balle et on calcule la possession de l'équipe
        possession_assigner = Possession()
        last_frame = len(self.tracks['players'])-1

        time = st.slider(label="Temps de la vidéo (s)",max_value=round(last_frame/24), step=10)
        team_1_possession, team_2_possession = possession_assigner.calculate_possession(self.tracks, time*24)

        # Affichage avec les colonnes
        col1, col2 = st.columns(2)
        with col1:  
            st.metric(label="Possession équipe 1", value= str(round(team_1_possession*100))+'%')

        with col2:  
            st.metric(label="Possession équipe 2", value= str(round(team_2_possession*100))+'%')

    
    # Heatmap de la balle
    def plot_ball_heatmap(self):
        # define number of grid squares for heatmap data
        rows = 5
        columns = 6
        
        heatmap = BallHeatmap(rows, columns)
        heatmap.calculateHeatmap(self.tracks)

        # On dessine le terrain
        dimensions = PitchDimensions()
        fig = make_pitch_figure(dimensions)

        data = np.array([
            [1 for _ in range(columns)]
            for _ in range(rows)
        ])

        fig = add_heatmap(fig, data)
        st.plotly_chart(fig)

    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    @st.experimental_fragment
    def plot_pitch(self):
        # On dessine le terrain
            dimensions = PitchDimensions()
            fig = make_pitch_figure(dimensions, pitch_background=SingleColourBackground("#74B72E"))
            st.plotly_chart(fig)


#    def plot_speeds(self):
        # On calcule la vitesse et la distance parcourue des joueurs
       # speed_calculator = SpeedCalculator()
      #  speed_calculator.add_speed_and_distance_to_tracks(self.tracks)
 #   for frame_num, player_track in enumerate(self.tracks['players']):
  #      for player_id, track in player_track.items():
        #    speed = self.tracks['players'][frame_num][player_id]['speed']
         #   distance = self.tracks['players'][frame_num][player_id]['distance']