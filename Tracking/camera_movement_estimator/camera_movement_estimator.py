import pickle
import cv2
import os
import numpy as np
from outils import distance, measure_xy_distance


class CameraMovementEstimator():
    def __init__(self, frame):

        # Distance minimale pour considérer un mouvement de caméra
        self.minimum_distance = 5
        
        # Paramètres pour le LK
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

        )

        # On met en gris la première image
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # On extrait la bannière du haut et du bas
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050]=1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features

        )


    # Fonction principale qui calcule les mouvements de la caméra
    def get_camera_movement(self, frames, read_from_file = False, file_path=None):

        # On lit si ce fichier n'a pas déjà été calculé 
        if read_from_file and file_path is not None and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                camera_movement = pickle.load(f)
                return camera_movement


        # On initialise
        camera_movement = [[0,0]]*len(frames)

        # On convertit l'image en gris
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # On répère les coins de l'image
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Activation optical flow
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                final_distance = distance(new_features_point, old_features_point)
                if final_distance>max_distance:
                    max_distance = final_distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()
        
        if file_path is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement


    # A chaque frame, on ajuste la position des joueurs en fonctions des mouvements de la camera calculés
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (measure_xy_distance(position, camera_movement))
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
