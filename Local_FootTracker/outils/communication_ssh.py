import paramiko
import config

def get_tracks(tracks_path, video_path):
    
    # Sinon, on se connecte en SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('localhost', username=config.username, password=config.password, port=config.port)

    # On exécute la partie tracking du code
    stdin, stdout, stderr= client.exec_command('/home/foottracker/myenv/bin/python3 /home/foottracker/myenv/FootTracker/Tracking/main.py')
    exit_status = stdout.channel.recv_exit_status()
    if(exit_status==0):
        # On récupère le fichier des tracks
        sftp = client.open_sftp()

        tracks_distant = '/home/foottracker/myenv/FootTracker/Tracking/tracks_files/tracks.pkl'

        sftp.get(tracks_distant, tracks_path)

        # On récupère la vidéo
        video_distant = '/home/foottracker/myenv/FootTracker/Tracking/output_videos/video1.avi'

        sftp.get(video_distant, video_path)

        # On ferme la connexion
        sftp.close()

        client.close()

def send_new_video(video_path):
    # Sinon, on se connecte en SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('localhost', username=config.username, password=config.password, port=config.port)

    # On récupère le fichier des tracks
    sftp = client.open_sftp()

    remote_path = '/home/foottracker/myenv/FootTracker/Tracking/input_videos/new_video.avi'

    sftp.put(video_path, remote_path)

    # On ferme la connexion
    sftp.close()

    

    client.close()