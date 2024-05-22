import sys

from outils import get_center_bbox, distance

# Classe qui associe la balle au joueur le plus proche
class ClosestPlayer():
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

