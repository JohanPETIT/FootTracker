class BallHeatmap():
    def __init__(self, rows, cols):
        self.heatmap = []
        self.pitch_width = 68 # Largeur du terrain et du triangle qu'on étudie
        self.pitch_height = 23.32 # Longueur du rectangle qu'on étudie, calculé proportionnellement à la longueur du terrain
        self.rows = rows
        self.columns = cols
        self.cell_width = self.pitch_width/cols
        self.cell_height = self.pitch_height/rows


    def calculateHeatmap(self, tracks):
            # On récupe les positions de la balle dans le référentiel du terrain
            for frame_num, track in enumerate(tracks['ball']):
                for track_id, track_info in track.items():
                        position = tracks['ball'][frame_num][track_id]['position_transformed']

                        # On récupère le numéro de la zone associé
                        if position is not None:
                              x, y = position
                              col = col = int(x // self.cell_width)
                              row = int(y // self.cell_height)

                              # On teste si ce n'est pas en dehors des limites
                              if col >= self.cols or row >= self.rows or col < 0 or row < 0:
                                return None
                              # Calculate zone number based on row and column
                              self.heatmap[row][col] +=1
                    
