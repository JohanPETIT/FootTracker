
import streamlit as st
import numpy as np
import pandas as pd
from plotly_football_pitch import make_pitch_figure, PitchDimensions, SingleColourBackground, add_heatmap
from foot_statistics import Possession, SpeedCalculator, BallHeatmap
from outils import get_team_colors, save_video, send_new_video, get_tracks
import uuid
import moviepy.editor as moviepy
import pickle


class Interface():
                    
    def __init__(self, tracks, remote_tracks_path, local_tracks_path, remote_avi_path, output_local_avi_path, output_local_mp4_path):
        self.tracks = tracks # Les tracks

        self.num_frames = len(self.tracks['players']) # Le nombre de frames de la vidéo
        self.period_seconds = 10 # En secondes, la période à laquelle on veut calculer les stats

        self.team1_color = get_team_colors(tracks)[0].astype(int).tolist() # Couleur de l'équipe 1
        self.team2_color = get_team_colors(tracks)[1].astype(int).tolist() # Couleur de l'équipe 2

        self.remote_tracks_path = remote_tracks_path
        self.remote_avi_path = remote_avi_path

        self.local_tracks_path = local_tracks_path
        self.output_local_avi_path = output_local_avi_path
        self.output_local_mp4_path = output_local_mp4_path

    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    # Dessine la page
    @st.experimental_fragment
    def plot_page(self):

        uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4"]) # On upload la vidéo
        if uploaded_file is not None:
            video_bytes = uploaded_file.getvalue()

            current = {}                    
            current['unique_code'] = str(uuid.uuid4()) # Créée un code unique
            current['video_path_avi'] = 'video_'+current['unique_code']+'.avi' # Créée un nom de nouvelle vidéo unique
            current['video_path_mp4'] = 'video_'+current['unique_code']+'.mp4' # Créée un nom de nouvelle vidéo unique
            current['tracks_path'] = 'tracks_files/tracks_'+current['unique_code']+'.pkl' # Créée un nom de nouveaux tracks unique

            with open('current.pkl', 'wb') as f:
                pickle.dump(current,f)
                f.close()

            save_video(video_bytes, 'input_videos/'+current['video_path_mp4'])
            send_new_video('input_videos/'+current['video_path_mp4'])

            self.remote_tracks_path='/home/foottracker/myenv/FootTracker/Tracking/'+current['tracks_path']
            self.remote_avi_path='/home/foottracker/myenv/FootTracker/Tracking/'+'output_videos/'+current['video_path_avi']

            self.local_tracks_path = current['tracks_path']
            self.output_local_avi_path = 'output_videos/'+current['video_path_avi']
            self.output_local_mp4_path = 'output_videos/'+current['video_path_mp4']

            get_tracks(self.remote_tracks_path, self.local_tracks_path, self.remote_avi_path, self.output_local_avi_path)
            
            # On convertit la vidéo en MP4
            clip = moviepy.VideoFileClip(self.output_local_avi_path)
            clip.write_videofile(self.output_local_mp4_path)

            with open(self.local_tracks_path, 'rb') as f:
                self.tracks = pickle.load(f)
                self.num_frames = len(self.tracks['players']) # Le nombre de frames de la vidéo
                self.team1_color = get_team_colors(self.tracks)[0].astype(int).tolist() # Couleur de l'équipe 1
                self.team2_color = get_team_colors(self.tracks)[1].astype(int).tolist() # Couleur de l'équipe 2
                f.close()


        # Si aucune vidéo n'est téléchargée, utilisez la vidéo initiale
        with open(self.output_local_mp4_path, 'rb') as video_file:
            video_bytes = video_file.read()

        # On initialise 2 colonnes pour avoir les stats à coté de la vidéo
        col1, col2 = st.columns(2)

        # Vidéo colonne gauche
        with col1:
            st.video(video_bytes, autoplay=True, muted=True)


        # Stats colonne droite
        with col2:
            # On dessine le truc pour sélectionner
            option = st.selectbox("Quelle statistique vous intéresse ?", ("Possession", "Position du ballon", "Top speed du match", "Distance parcourue par l'équipe", "Autre"), index=None, placeholder="Choisissez une option !")
            
            # Possession 
            if(option == "Possession"):
                self.plot_possession()

            # Position du ballon 
            if(option == "Position du ballon"):
                self.plot_ball_heatmap()

            # Vitesse des joueurs
            if(option == "Top speed du match"):
                self.plot_speeds()

            # Vitesse des joueurs
            if(option == "Distance parcourue par l'équipe"):
                self.plot_distances_covered()
                 

    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    @st.experimental_fragment
    def plot_possession(self):

        possession_assigner = Possession() # Initialisation
        last_frame = self.num_frames-1 # Dernière frame
        self.period_seconds # Intervalle en secondes pour lequel on veut calculer

        # Initialiser total_distance avec des zéros
        possession = [[0] * 2 for _ in range(self.num_frames)]

        # On note la possession de chaque équipe pour l'afficher selon la période de temps
        for frame_num in range(0, self.num_frames, 24*self.period_seconds):
            possession[frame_num][0], possession[frame_num][1] = possession_assigner.calculate_possession(self.tracks, frame_num)

        # Dataframe préparé pour être affiché par le graphe
        possession = pd.DataFrame({
            'Temps en secondes': list(range(0, int(self.num_frames/24), self.period_seconds)),
            'Possession de l\'équipe 1 (%)': [possession[24*frame_num][0]*100 for frame_num in range(0, int(self.num_frames/24), self.period_seconds)],
            'Possession de l\'équipe 2 (%)': [possession[24*frame_num][1]*100 for frame_num in range(0, int(self.num_frames/24), self.period_seconds)]
            })

        # Bar chart de la possession en fonction du temps
        st.bar_chart(possession, x='Temps en secondes', y=['Possession de l\'équipe 1 (%)', 'Possession de l\'équipe 2 (%)'], color=[self.team1_color, self.team2_color])

        # Slider pour afficher plus précisément la stat à un temps précis
        time = st.slider(label="Temps de la vidéo (s)",max_value=round(last_frame/24), step=10)
        team_1_possession, team_2_possession = possession_assigner.calculate_possession(self.tracks, time*24) # On recalcule la possession pour ce temps précis

        # Affichage avec les colonnes
        col1, col2 = st.columns(2)
        with col1:  
            st.metric(label="Possession équipe 1", value= str(round(team_1_possession*100))+'%') # Possession team 1

        with col2:  
            st.metric(label="Possession équipe 2", value= str(round(team_2_possession*100))+'%') # Possession team 2


    
    # Heatmap de la balle
    @st.experimental_fragment
    def plot_ball_heatmap(self):
        # define number of grid squares for heatmap data
        rows = 5
        columns = 6
        
        heatmap = BallHeatmap(rows, columns)
        zones =heatmap.calculateHeatmap(self.tracks)

        # On dessine le terrain
        dimensions = PitchDimensions()
        fig = make_pitch_figure(dimensions)

        data = np.array([
            [zones[x][y] for x in range(rows)]
            for y in range(columns)
        ])
        
        aire=BallHeatmap.detect_area(34,11.5)
        print(aire)
        
        fig = add_heatmap(fig, data)
        st.plotly_chart(fig)


    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    @st.experimental_fragment
    def plot_pitch(self):
        # On dessine le terrain
            dimensions = PitchDimensions()
            fig = make_pitch_figure(dimensions, pitch_background=SingleColourBackground("#74B72E"))
            st.plotly_chart(fig)


    @st.experimental_fragment
    def plot_speeds(self):
        # On calcule la vitesse et la distance parcourue des joueurs
        speed_calculator = SpeedCalculator()
        top_speed, track_id, frame_num, _= speed_calculator.add_speed_and_distance_to_tracks(self.tracks) # Retourne la top vitesse, le joueur et le moment

        # Affichage avec les colonnes
        col1, col2, col3 = st.columns(3)
        with col1:  
            st.metric(label="Top speed :", value= str(round(top_speed, 1))+'km/h') # Top vitesse

        with col2:  
            st.metric(label="Performed by number :", value= track_id) # Joueur qui a fait la top vitesse

        with col3:  
            st.metric(label="At time : (s)", value= round(frame_num/24)) # Moment de la top vitesse

            #   for frame_num, player_track in enumerate(self.tracks['players']):
            #      for player_id, track in player_track.items():
                    #    speed = self.tracks['players'][frame_num][player_id]['speed']
                    #   distance = self.tracks['players'][frame_num][player_id]['distance']

    @st.experimental_fragment
    def plot_distances_covered(self):

        # Initialiser total_distance avec des zéros
        total_distance = [[0] * 2 for _ in range(self.num_frames)]

        # On calcule la vitesse et la distance parcourue des joueurs
        speed_calculator = SpeedCalculator()
        frame_window = speed_calculator.add_speed_and_distance_to_tracks(self.tracks)[3]

        # On récupère l'équipe du joueur et sa couleur pour chaque joueur de chaque frame
        for frame_num in range(0, self.num_frames, frame_window):
            if (frame_num>0):
                for frame_batch in range(frame_num-frame_window+1, frame_num+1):
                    total_distance[frame_batch][0] += total_distance[frame_num-frame_window][0]
                    total_distance[frame_batch][1] += total_distance[frame_num-frame_window][1]
            for player_id, track in self.tracks['players'][frame_num].items():
                team = self.tracks['players'][frame_num][player_id]['team'] # On récupère l'équipe du joueur

                # On récupère la distance parcourue totale du joueur si elle n'est pas 0
                if self.tracks['players'][frame_num][player_id].get('distance') != None:
                    total_distance[frame_num][team] +=  self.tracks['players'][frame_num][player_id]['distance'] # Distance totale équipe jusqu'à la frame n = distance joueur sur le nouvel intervalle + distance totale équipe jusqu'à frame n-1
        # Dataframe préparé pour être affiché par le graphe
        total_distance = pd.DataFrame({
            'Temps en secondes': list(range(0, int(self.num_frames/24), self.period_seconds)),
            'Distance de l\'équipe 1 (m)': [total_distance[24*frame_num][0] for frame_num in range(0, int(self.num_frames/24), self.period_seconds)],
            'Distance de l\'équipe 2 (m)': [total_distance[24*frame_num][1] for frame_num in range(0, int(self.num_frames/24), self.period_seconds)]
            })
        
        # Graphe des distances (les couleur RGB sont mises en int sinon ça marche pas)
        st.area_chart(total_distance, x='Temps en secondes', y=['Distance de l\'équipe 1 (m)', 'Distance de l\'équipe 2 (m)'], color=[self.team1_color, self.team2_color])