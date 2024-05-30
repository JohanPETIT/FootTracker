import numpy as np
class BallHeatmap():
    def __init__(self, rows, cols):
        self.heatmap = []
        self.pitch_width = 68 # Largeur du terrain et du triangle qu'on étudie
        self.pitch_height = 23.32 # Longueur du rectangle qu'on étudie, calculé proportionnellement à la longueur du terrain
        self.rows = rows
        self.columns = cols
        self.cell_width = self.pitch_width/cols # on divise la largeur du pitch par le nombre de colonnes 
        self.cell_height = self.pitch_height/rows # on divise la hauteur du pitch par le nombre de lignes
        self.heatmap = np.zeros((rows, cols)) # initialisation de la heatmap avec des zéros
        print(self.cell_width,self.cell_height)
    
    # Fonction qui permet de détecter la zone dans laquelle se trouve la balle en donnant ses coordonnées
    def detect_area(x,y,pitch_width=68,pitch_height=23.32):
      zone_width = pitch_width / 5
      zone_height = pitch_height / 6
      col = int(x // zone_width) # division entière pour avoir le numéro de la colonne col
      row = int(y // zone_height) # division entière pour avoir le numéro de la ligne row
      zone_number = (row,col) 
      print('zone identifiée:')
      return zone_number # on retourne zone_number qui est un tuple (row,col)
    
    # Fonction qui permet de calculer la heatmap de la balle   
    def calculateHeatmap(self,tracks):
        #print(self.heatmap)
        # on parcourt pour chaque frame les tracks de la balle
        for frame_num, track in enumerate(tracks['ball']):
          # track_id est l'identifiant de la balle, 
          for track_id, track_info in track.items():
            position = tracks['ball'][frame_num][track_id]['position_transformed']
            if position is not None:
              x, y = position
              #print(x,y)
              row = int(y//self.cell_height)//3 # division par 3 car on dépassait la limite autorisée, en faisant cela on a de bons résultats
              col= int(x//self.cell_width) # on divise le x par la largeur de la zone pour avoir le numéro de la colonne
              #print(row,col)
              #on teste si on sort pas des limites 
              if 0 <= col < self.columns and 0 <= row < self.rows:
                self.heatmap[row][col] += 1
              #print(self.heatmap)
        return self.heatmap

    
