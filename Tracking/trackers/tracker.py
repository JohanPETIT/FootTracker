from ultralytics import YOLO
import supervision as sv

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
            detections.append(detections_batch) # On ajoute les frames qu'on vient de prédire au total des frames prédites
            break
        return detections


    def get_objects_tracks(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections): # On boucle sur l'ensemble des détections et on note le numéro de la détection dans frame_num
            cls_names = detection.names # Dictionnaire : numéro -> classe
            cls_names_inv = {v:k for k,v in cls_names.items()} # Dictionnaire : classe -> numéro

            detection_supervision = sv.Detections.from_ultralytics(detection) # On convertit les détections au format de la librairie SuperVision
