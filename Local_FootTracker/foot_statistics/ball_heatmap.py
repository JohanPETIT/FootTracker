import numpy as np
class BallHeatmap():
    def __init__(self, rows, cols):
        self.heatmap = []
        self.pitch_width = 68 # Largeur du terrain et du triangle qu'on étudie
        self.pitch_height = 23.32 # Longueur du rectangle qu'on étudie, calculé proportionnellement à la longueur du terrain
        self.rows = rows
        self.columns = cols
        self.cell_width = self.pitch_width/cols
        self.cell_height = self.pitch_height/rows
        self.heatmap = np.zeros((rows, cols))
        print(self.cell_width,self.cell_height)
      
    def detect_area(x,y,pitch_width=68,pitch_height=23.32):
      zone_width = pitch_width / 5
      zone_height = pitch_height / 6
      col = int(x // zone_width)
      row = int(y // zone_height)
      zone_number = (row,col)
      print('zone identifiée:')
      return zone_number
       
    def calculateHeatmap(self,tracks):
        print(self.heatmap)
        for frame_num, track in enumerate(tracks['ball']):
          for track_id, track_info in track.items():
            position = tracks['ball'][frame_num][track_id]['position_transformed']
            print(position) 
            #inutile ici
            if position is not None:
              x, y = position
              #print(x,y)
              row = int(y//self.cell_height)//3
              col= int(x//self.cell_width)
              print(row,col)
              #on teste si on sort pas des limites 
              if 0 <= col < self.columns and 0 <= row < self.rows:
                self.heatmap[row][col] += 1
              #print(self.heatmap)
        return self.heatmap

    
