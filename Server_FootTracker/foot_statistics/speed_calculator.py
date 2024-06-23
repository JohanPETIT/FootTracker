from outils import distance

class SpeedCalculator():
    def __init__(self):
        self.frame_window = 72 # Intervalle de frames pendant laquelle on va calculer la vitesse du joueur
        self.frame_rate = 24 # Frame rate de la vidéo

    # Fonction principale qui calcule la distance parcourue et la vitesse des joueurs
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        top_speed = 0
        for object, object_tracks in tracks.items():
            if object=="referees" or object=="ball": # On veut calculer uniquement la vitesse des joueurs
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window): # On boucle sur le nombre de frames avec une incrémentation de l'intervalle
                last_frame = min(frame_num+self.frame_window, number_of_frames-1) # On prend la dernière frame de l'intervalle, qui est donc soit notre frame + l'intervalle ou juste la dernière frame

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]: # Si le track est dans la première frame mais pas dans la dernière de l'intervalle on ne calcule pas sa vitesse
                        continue
                    
                    # On récupère la position du joueur sur le terrain aux frames de début et de fin de l'intervalle 
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # On teste si le joueur est resté dans le rectangle de transformation durant toute l'intervalle
                    if start_position is None or end_position is None:
                        continue

                    distance_covered = distance(start_position, end_position)
                    time_passed = (last_frame-frame_num)/self.frame_rate
                    speed_metres_per_second = distance_covered/time_passed
                    speed_km_per_hour = speed_metres_per_second*3.6

                    # On teste si la nouvelle vitesse calculée est un nouveau record
                    if speed_km_per_hour > top_speed:
                        top_speed = speed_km_per_hour
                        top_id = track_id
                        top_frame = frame_num

                    # On initialise la distance totale pour chaque entité pas encore repertoriée
                    if object not in total_distance:
                        total_distance[object]={}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    # On ajoute la distance calculée dans le dernier intervalle au total de distance parcourue
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame): # On écrit la vitesse et la distance parcourue dans l'intervalle sur TOUTES les frames de l'intervalle
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = distance_covered

        return top_speed, top_id, top_frame, self.frame_window


