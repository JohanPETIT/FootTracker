import numpy as np
class BallHeatmap():
  def __init__(self, rows, cols):
    self.pitch_width = 64 # Largeur du terrain et du triangle qu'on étudie
    self.pitch_height = 40 # Longueur du rectangle qu'on étudie, calculé proportionnellement à la longueur du terrain (une ligne c'est 5.83)

    self.real_pitch_width = 64 # Largeur d'un terrain entier
    self.real_pitch_height = 100 # Longueur d'un terrain entier

    self.num_rows = rows
    self.num_columns = cols

    self.column_height = self.pitch_height/self.num_columns
    self.column_percentage = (self.column_height/self.real_pitch_height)*100

    self.num_free_columns = int((self.real_pitch_height - self.pitch_height)/self.column_height)
    if(self.num_free_columns % 2 != 0):
        self.num_free_columns +=1


    self.heatmap = np.zeros((cols+self.num_free_columns, rows)) # initialisation de la heatmap avec des zéros


    

  def calculateHeatmap(self, tracks):
    for frame_num in range(len(tracks['players'])):
      position = tracks['ball'][frame_num][1]['position_transformed']
      if position is not None :
        x_small ,y_small = position

        if(int(x_small)>=5):
          x_small_percentage = (x_small / self.pitch_height) * 100
          
          full_col_number = int(x_small_percentage  * self.num_columns / 100)
          total_col_number = int(self.num_free_columns/2 + full_col_number) ## Attention au /2

          y_percentage = (x_small/self.pitch_width)*100
          total_row_number = int(y_percentage * self.num_rows /100)
          print("second " + str(int(frame_num/24)) + " coor : " + str(x_small) + "," + str(y_small))
          print("second " + str(int(frame_num/24)) + " coor : " + str(total_col_number) + "," + str(total_row_number))
          self.heatmap[total_col_number][total_row_number] += 1

    return self.heatmap