import numpy as np
import cv2

class PerspectiveTransformer():
    def __init__(self):
        pitch_width = 68 # Largeur du terrain et du triangle qu'on étudie
        pitch_length = 23.32 # Longueur du rectangle qu'on étudie, calculé proportionnellement à la longueur du terrain

        # On donne les positions en pixel des coins du rectangle 
        # DONNEES A RENTRER A LA MAIN !!!
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ], np.float32)

        # On associe aux positions en pixel des coins du rectangle, les positions définies dans notre nouveau référentiel
        self.target_vertices = np.array([
            [0, pitch_width],
            [0,0],
            [pitch_length, 0],
            [pitch_length, pitch_width]
        ], np.float32)

        # On convertit en float
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices_vertices = self.pixel_vertices.astype(np.float32)

        # On initialise le modèle qui va effectuer la transformation
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)


    # Transforme la perspective d'un seul point, du référentiel de la caméra au référentiel réel
    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 # On teste si la position de l'entité est dans notre rectangle qu'on souhaite transformer
        if not is_inside:
            return None
        
        reshaped_point = point.reshape(-1,1,2).astype(np.float32) # On le reshape pour qu'il passe en entrée du perspective_transformer
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer) # On applique la transformation de perspective
        final_point = transform_point.reshape(-1,2) # On le re reshape sous le format d'output qu'on souhaite ici

        return final_point
    # Applique le changement de perspective sur la position de toutes les entités et les ajoute aux tracks
    def add_transformed_positions_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted'] # On prend la position ajustée de chaque entité
                    position = np.array(position)
                    position_transformed = self.transform_point(position) # On en transforme la perspective
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist() # On la remet au bon format d'output
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed