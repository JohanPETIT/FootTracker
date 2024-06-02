import cv2

# Lit la vidéo frame par frame et retourne la liste des frames à la fin
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # print(cap.isOpened()) # Teste si le path de la vidéo est bon
    frames = []
    while True:
        ret, frame = cap.read() # Lit la prochaine frame, ret = False si on atteint la fin de la vidéo
        if not ret:
            break
        frames.append(frame) # Ajoute la dernière frame à la liste des frames
    return frames

# Enregistre la vidéo une fois les modifications apportées (détections, annotations etc...)
def save_video(output_video_frames, output_video_path):
    output_format = cv2.VideoWriter_fourcc(*'XVID') # Donne le format d'output
    out = cv2.VideoWriter(output_video_path, output_format, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # On définit le path de sortie, son format vidéo, le nombre de FPS et la taille des images de sortie
    for frame in output_video_frames: # Ecrit la vidéo de sortie frame par frame
        out.write(frame)
    out.release()
