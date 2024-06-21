# Retourne le centre d'une bbox
def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

# Retourne la largeur d'une bbox
def get_width_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return x2-x1

# Calcule la distance scalaire entre 2 points de coordonnées (x,y)
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

# Retourne la distance vectorielle (x,y) entre 2 points de coordonnées (x,y)
def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]

# Retourne le milieu tout en bas d'une bbox
def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int(y2)