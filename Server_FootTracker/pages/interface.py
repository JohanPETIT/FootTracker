
import streamlit as st
import numpy as np
import pandas as pd
from plotly_football_pitch import make_pitch_figure, PitchDimensions, SingleColourBackground, add_heatmap
from foot_statistics import Possession, SpeedCalculator, BallHeatmap
from outils import get_team_colors


st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:", initial_sidebar_state="collapsed")
class Interface():
                    
    def __init__(self):
        self.tracks = st.session_state['tracks'] # Les tracks

        self.events = st.session_state['events'] # Les events

        self.num_frames = len(self.tracks['players']) # Le nombre de frames de la vidéo

        self.time_video_seconds = int(self.num_frames/(24))
        self.time_video_minutes = int(self.time_video_seconds/60)
        
        if self.time_video_minutes <= 3:
            self.period_seconds = 15
        if self.time_video_minutes >= 3 and self.time_video_minutes <= 15:
            self.period_seconds = int(self.time_video_seconds/3) # En secondes, la période à laquelle on veut calculer les stats

        elif self.time_video_minutes >= 16 and self.time_video_minutes <= 30:
            self.period_seconds = int(self.time_video_seconds/6)
        else :
            self.period_seconds = int(self.time_video_seconds/9)
            
        

        self.team1_color = get_team_colors(self.tracks)[0] # Couleur de l'équipe 1
        self.team2_color = get_team_colors(self.tracks)[1] # Couleur de l'équipe 2

        self.output_local_mp4_path = st.session_state['video_path'] # Chemin d'accès de la video mp4
    
    # Fonction principale d'affichage de la page
    @st.experimental_fragment
    def plot_page(self):
        # Si aucune vidéo n'est téléchargée, utilisez la vidéo initiale
        with open(self.output_local_mp4_path, 'rb') as video_file:
            video_bytes = video_file.read()

        # On initialise 2 colonnes pour avoir les stats à coté de la vidéo
        col1, col2 = st.columns(2)

        # Vidéo colonne gauche
        with col1:
            st.video(video_bytes, autoplay=True, muted=True, loop=True)


        # Stats colonne droite
        with col2:
            # On dessine le truc pour sélectionner
            option = st.selectbox("Quelle statistique vous intéresse ?", ("Possession", "Position du ballon", "Top speed du match", "Distance parcourue par l'équipe", "Événements du match", "Autre"), index=None, placeholder="Choisissez une option !")
            
            # Possession 
            if(option == "Possession"):
                self.plot_possession()

            # Position du ballon 
            if(option == "Position du ballon"):
                self.plot_ball_heatmap()

            # Vitesse des joueurs
            if(option == "Top speed du match"):
                self.plot_speeds()

            # Distance totale parcourue de chaque équipe
            if(option == "Distance parcourue par l'équipe"):
                self.plot_distances_covered()

            # Événements du match (passe, duel, touche)
            if(option == "Événements du match"):
                self.plot_events()
                    
                 

    #Empêche de réexécuter tout le code dès qu'on clique sur qqc
    @st.experimental_fragment
    def plot_possession(self):

        possession_assigner = Possession() # Initialisation
        last_frame = self.num_frames-1 # Dernière frame

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
        rows = 10 # 5
        columns = 5
        
        heatmap = BallHeatmap(rows, columns)
        zones = heatmap.calculateHeatmap(self.tracks)

        # On dessine le terrain
        dimensions = PitchDimensions()
        fig = make_pitch_figure(dimensions)

        print(zones.shape[0])

        data = np.array([
            [zones[x][y] for x in range(zones.shape[0])]
            for y in range(zones.shape[1])
        ])
        
        #fig = add_heatmap(fig, data)
        #smoothed heatmap
        fig = add_heatmap(fig, data, zsmooth='best', colorscale='YlOrRd')
        st.plotly_chart(fig)


    # Affiche un terrain (pour la heatmap)
    @st.experimental_fragment
    def plot_pitch(self):
        # On dessine le terrain
            dimensions = PitchDimensions()
            fig = make_pitch_figure(dimensions, pitch_background=SingleColourBackground("#74B72E"))
            st.plotly_chart(fig)

    # Affiche la top vitesse du match, à quel instant et par quel joueur
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

    # Affiche les distances totales parcourues par les 2 équipes
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


    # Affiche les événements du match, leur nombre et leur moment
    @st.experimental_fragment
    def plot_events(self):
        # Dictionnaire qui compte le nombre d'occurences des événements du match
        event_counter={
            "play" : 0,
            "challenge" : 0,
            "throwin" : 0
        }

        # Répertorie l'événement à chaque seconde
        event_at_second = {}

        # Pour chaque seconde on répertorie l'événement et on le compte
        for second_num, event in enumerate(self.events):
            if event=="play":
                event_counter["play"] += 1
                event_at_second[second_num] = "Passe"
                continue
            if event=="challenge":
                event_counter["challenge"] += 1
                event_at_second[second_num] = "Duel"
                continue
            if event == "throwin":
                event_counter["throwin"] += 1
                event_at_second[second_num] = "Touche"
                continue
            event_at_second[second_num] = "Rien"

        # Affichage metrics avec les colonnes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Nombre de passes", value=event_counter["play"])
        with col2:
            st.metric(label="Nombre de duels", value=event_counter["challenge"])
        with col3:
            st.metric(label="Nombre de touches", value=event_counter["throwin"])
        
        # Slider pour afficher plus précisément la stat à un temps précis
        time = st.slider(label="Temps de la vidéo (s)",max_value=round((self.num_frames-1)/24), step=1)
        event_at_time = event_at_second[time]
        # Metric de l'événement repertorié au temps indiqué
        st.metric(label="Événement à t = " + str(time) + " s :", value=event_at_time)

# On instancie l'interface et on lance l'affichage de la page
interface = Interface()
interface.plot_page()