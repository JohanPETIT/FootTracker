from ultralytics import YOLO
import supervision as sv
import pickle
import os

# Classe d'objet Tracker 
class Tracker:

    # Constructeur d'un objet Tracker, qui possède deux attributs : son modèle YOLO et un objet Tracker supervision
    def __init__(self, model_path): 
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    # Détection des objets avec YOLO
    def detect_frames(self, frames):
        batch_size = 20 # Donne la taille d'un groupe de frames qu'on veut détecter
        detections = []
        for i in range(0, len(frames), batch_size): 
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1) # On applique YOLO sur le groupe de frames. La confidence donne une valeur en dessous de laquelle on ne détecte pas l'objet
            detections += detections_batch # On ajoute les frames qu'on vient de prédire au total des frames prédites
        return detections

    # Fonction principale de tracking
    def get_objects_tracks(self, frames, read_from_file=False, file_path=None):

        # On teste s'il existe déjà un fichier des tracks enregistré pour ne pas tout réexécuter
        if read_from_file and file_path is not None and os.path.exists(file_path):
            # S'il existe, on l'ouvre et on charge les tracks
            with open(file_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # On active la détection d'objets
        detections = self.detect_frames(frames)

        # Initialisation des tracks
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections): # On boucle sur l'ensemble des détections et on note le numéro de la détection dans frame_num
            cls_names = detection.names # Dictionnaire : numéro -> classe
            cls_names_inv = {v:k for k,v in cls_names.items()} # Dictionnaire : classe -> numéro

            detection_supervision = sv.Detections.from_ultralytics(detection) # On convertit les détections au format de la librairie SuperVision

            # On convertit les gardiens en joueurs
            for object_ind, class_id in enumerate(detection_supervision.class_id): # On boucle sur l'ensemble des id des classes, pour chaque frame
                if cls_names[class_id] == "goalkeeper": # On teste si le joueur est un gardien
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"] # On remplace sa classe par une classe de joueur normal
            
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision) # On ajoute un track_id à chaque entité pour la suivre à travers les frames

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # On recense tout ce qui se passe sur l'image (bbox, cls_id, track_id) pour chaque frame
            for frame_detection in detection_with_tracks:
                # On récup les listes des bbox, cls_id et des track_id pour chaque frame soumise au tracker
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox} # On recense la position du joueur (represénté par son track id) à cette frame

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox} # On recense la position de l'arbitre (represénté par son track id) à cette frame

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox} # On recense la position de la balle (sans track_id car il n'y en a qu'une) à cette frame

        # On enregistre le fichier des tracks s'il n'existe pas déjà
        if file_path is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(tracks,f)

        return tracks
