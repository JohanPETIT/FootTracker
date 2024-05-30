def get_team_colors(tracks):
            team_colors = {}
            # On récupère l'équipe du joueur et sa couleur pour chaque joueur de chaque frame
            for frame_num, player_track in enumerate(tracks['players']):
                for player_id, track in player_track.items():
                    team = tracks['players'][frame_num][player_id]['team'] # On récupère l'équipe du joueur
                    player_color = tracks['players'][frame_num][player_id]['team_color'] # On récupère la couleur de l'équipe du joueur
                    
                    # Teste si on a déjà recensé la couleur d'une équipe ou non
                    if team_colors.get(team) is None:
                        team_colors[team] = player_color

                    # Une fois les 2 couleurs d'équipes renseignées, les retourner
                    if team_colors.get(0) is not None and team_colors.get(0) is not None:
                        return team_colors.get(0), team_colors.get(1)