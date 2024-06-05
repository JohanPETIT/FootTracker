import sys
import numpy as np
from outils import get_center_bbox, distance

# Classe qui associe la balle au joueur le plus proche
class Possession():
    def __init__(self):
        self.max_player_ball_distance = 70  # Définit la distance en pixels au dessus de laquelle on associe la balle à aucun joueur

    # Fonction principale pour associer la balle à un joueur
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_bbox(ball_bbox)


        min_distance = 99999
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = distance((player_bbox[0], player_bbox[-1]), ball_position) # Distance entre la gauche du joueur (x1) et la balle
            distance_right = distance((player_bbox[2], player_bbox[-1]), ball_position) # Distance entre la droite du joueur (x2) et la balle
            final_distance = min(distance_left, distance_right) # On prend la distance minimale entre ces 2 pour avoir le pied le plus proche de la balle

            if final_distance < self.max_player_ball_distance: # On teste si on a besoin d'associer un joueur à la balle ou non
                if final_distance < min_distance: # On teste si la distance de ce joueur avec la balle est inférieure à celle du joueur actuellement plus proche de la balle
                    min_distance = final_distance # Si oui, c'est donc lui le nouveau joueur le plus proche de la balle donc on update les paramètres
                    assigned_player = player_id

        return assigned_player
    
    # Calcule la possession des 2 équipes jusqu'à une certaine frame
    def calculate_possession(self, tracks, last_frame):
        team_ball_possession = [] # On veut associer à chaque frame de cette liste l'équipe qui a la balle
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox'] # On récupère la bbox de la balle
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox) # On récupère le joueur le plus proche du ballon


            # On note quel joueur a le ballon s'il y en a un, et à quelle équipe il appartient
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_possession.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_possession.append(team_ball_possession[-1]) # S'il n'y a pas de joueur qui a le ballon, on prend la dernière équipe qui l'avait
        
        # On a maintenant les informations de quelle équipe à la balle pour toutes les frames de la vidéo
        team_ball_possession = np.array(team_ball_possession)

        # On calcule le nombre de fois que chaque équipe a la balle jusqu'à la frame actuelle
        team_ball_possession_until_frame = team_ball_possession[:last_frame+1]
        team_1_num_frames = team_ball_possession_until_frame[team_ball_possession_until_frame==0].shape[0]
        team_2_num_frames = team_ball_possession_until_frame[team_ball_possession_until_frame==1].shape[0]

        team_1_possession = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2_possession = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        
        return team_1_possession, team_2_possession

        


        

