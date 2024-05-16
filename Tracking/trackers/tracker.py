from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np
import os
from outils import get_center_bbox, get_width_bbox

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

    # Dessine les annotations de la vidéo
    def draw_annotations(self, video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # On copie la frame pour ne pas écrire sur la liste des frames originales
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]


            # On dessine les annotations de la balle
            for _, ball in ball_dict.items(): 
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

                
            # On dessine les annotations des joueurs
            for track_id, player in player_dict.items(): # .items() renvoie la clé et la valeur du dictionnaire, donc ici track_id et player
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id) # On dessine une ellipse d'une certaine couleur autour de la bbox d'un joueur

            # On dessine les annotations des arbitres
            for _, referee in referee_dict.items(): # Pas besoin du track_id des arbitres ici, et on détectera ensuite si l'entité est un joueur s'il a un track_id
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))
 

            output_video_frames.append(frame)
        return output_video_frames
            

    # Dessine une ellipse
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # Le y2 d'en bas à droite de la bbox

        x_center, y_center = get_center_bbox(bbox) # On récup les centres x et y de la bbox
        width = get_width_bbox(bbox) # On récup sa largeur

        # On dessine l'ellipse avec CV2
        cv2.ellipse(
            frame,
            center = (x_center, y2), # Centre de l'ellipse, ici au milieu et tout en bas de la bbox
            axes=(int(width), int(0.35*width)), # Axes de l'ellipse
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # On donne les paramètres pour dessiner le rectangle
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2 
        y1_rect = (y2-rectangle_height//2) +15 # +15 pour décaler le rectangle en bas de l'ellipse
        y2_rect = (y2+rectangle_height//2) +15 # +15 pour décaler le rectangle en bas de l'ellipse

        if track_id is not None: # On teste si c'est un joueur (avec un track_id) ou un arbitre (sans) pour savoir si on doit dessiner le rectangle

            # Dessine le rectangle
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)), # Représente le coin en haut à gauche du rectangle
                (int(x2_rect), int(y2_rect)), # Représente le coin en bas à droite du rectangle
                color,
                cv2.FILLED
            )

            x1_text = x1_rect+12 # 12 c'est le padding pour centrer le texte dans le rectangle

            # Permet de gérer les grands nombres sinon ça dépasse à droite
            if (track_id > 9 & track_id <100):
                x1_text -=5 
            if track_id > 99:
                x1_text -= 10 
            
            # Ecrit le texte
            cv2.putText(
                frame,
                f"{track_id}", # Permet de donner une variable dans une chaine de caractères
                (int(x1_text), int(y1_rect+15)), # L'endroit où on met le texte
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, # Font size
                (255,255,255), # Couleur
                2 # Thickness

            )


        return frame
    
    # Dessine un triangle
    def draw_triangle(self, frame, bbox, color):
        y= int(bbox[1])
        x, _ = get_center_bbox(bbox)
        
        # Donne les coordonnées des 3 points du triangle à relier
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # Dessine le triangle
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Dessine les contours du triangle

        return frame

