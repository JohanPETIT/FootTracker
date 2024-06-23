from sklearn.cluster import KMeans 
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None


    # Renvoie le K-Means clustering d'une image
    def get_clustering_model(self, image):

        # On reshape l'image en un tableau 2D
        image_2d = image.reshape(-1, 3)

        # On applique le K-Means clustering pour différencier la couleur du maillot et de l'arrière plan
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    # Recense la couleur du maillot d'un joueur
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # On coupe l'image pour obtenir seulement l'image d'un joueur (sa bbox)

        top_half_image = image[0:int(image.shape[0]/2),:] # On isole la partie supérieure de l'image car c'est là que se trouve le maillot

        kmeans = self.get_clustering_model(top_half_image) # On récupère le k-means de l'image
        labels = kmeans.labels_ # On récupère les labels (0 ou 1) pour chaque pixel, qui permetttent d'identifier si ce pixel appartient au background ou au joueur
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1]) # On les reshape pour avoir que ceux qui concernent la partie supérieure de l'image

        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1, -1]] # On récupère les labels des pixels des coins, on suppose que ceux-ci appartiennent au background
        background_cluster = max(set(corner_clusters), key=corner_clusters.count) # On identifie le label qui revient le plus souvent parmi les 4 coins, celui-ci sera le label du background
        player_cluster = 1-background_cluster # Le label du player est l'opposé binaire de celui du background

        player_color = kmeans.cluster_centers_[player_cluster] # Retourne la couleur RGB du label associé au joueur

        return player_color



    def assign_team_color(self, frame, player_detections):

        # On récupère la couleur de maillot de chaque joueur
        player_colors=[]
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # On refait un kmeans pour découper l'ensemble des couleurs des joueurs en couleurs de 2 équipes distinctes
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        # On enregistre le modèle kmeans pour le réutiliser pour prédire l'équipe d'un joueur
        self.kmeans = kmeans

        # On écrit quelles sont les couleurs officielles des 2 équipes
        self.team_colors[0] = kmeans.cluster_centers_[0]
        self.team_colors[1] = kmeans.cluster_centers_[1]

    # Donne l'équipe à laquelle un joueur appartient en matchant sa couleur avec la couleur d'une des 2 équipes
    def assign_player_team(self, frame, player_bbox, player_id):
        # Si on connait déjà l'équipe d'un joueur grâce à une frame précédente, pas besoin de la recalculer
        #if player_id in self.player_team_dict:
         #   return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox) # On récupère la couleur du maillot du joueur

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] # On prédit à quelle équipe appartient ce joueur

        self.player_team_dict[player_id] = team_id # On enregistre l'équipe du joueur

        return team_id
