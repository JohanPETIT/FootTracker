
import numpy as np
class BallHeatmap():
    def __init__(self, rows, cols):
        self.heatmap = []
        self.pitch_width = 68 # Largeur du terrain et du triangle qu'on étudie
        self.pitch_height = 40.81 # Longueur du rectangle qu'on étudie, calculé proportionnellement à la longueur du terrain
        self.rows = rows
        self.columns = cols
        self.cell_width = self.pitch_width/cols # on divise la largeur du pitch par le nombre de colonnes 
        self.cell_height = self.pitch_height/rows # on divise la hauteur du pitch par le nombre de lignes
        self.heatmap = np.zeros((rows, cols)) # initialisation de la heatmap avec des zéros
        print(self.cell_width,self.cell_height)
    
    # Fonction qui permet de détecter la zone dans laquelle se trouve la balle en donnant ses coordonnées
    def detect_area(x,y,pitch_width=68,pitch_height=40.81):
      zone_width = pitch_width / 5
      zone_height = pitch_height / 6
      col = int(x // zone_width) # division entière pour avoir le numéro de la colonne col
      row = int(y // zone_height) # division entière pour avoir le numéro de la ligne row
      zone_number = (row,col) 
      #print('zone identifiée:')
      return zone_number # on retourne zone_number qui est un tuple (row,col)
    
    # Fonction qui permet de calculer la heatmap de la balle   
    def calculateHeatmap(self, tracks):
      for frame_num, track in enumerate(tracks['ball']):
          # track_id est l'identifiant de la balle, 
          for track_id, track_info in track.items():
            position = tracks['ball'][frame_num][track_id]['position_transformed']
            # Normalisation des positions en pourcentage du terrain
            if position is not None:
              x,y = position
              x_x = (x / self.pitch_height) * 100
              y_y = (y / self.pitch_width)  * 100
              print(x_x,y_y)
            # Détermination de la zone basée sur le pourcentage
              col = int(x_x  * self.columns / 100)
              row = int(y_y  * self.rows /100)
              print(col,row)
            # Assurez-vous que les indices sont dans les limites
              if 0 <= col < self.columns +1 and 0 <= row < self.rows +1:
                if col == 0 and row != 0 :
                  self.heatmap[row-1][col] += 1
                elif row ==0 and col != 0:
                  self.heatmap[row][col-1] += 1
                elif col != 0 and row != 0:
                  self.heatmap[row-1][col-1] += 1
                  
      return self.heatmap

    

