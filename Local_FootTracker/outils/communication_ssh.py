import paramiko
import config

def get_tracks(remote_tracks_path, local_tracks_path, remote_video_path, local_video_path):
    
    # Sinon, on se connecte en SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('localhost', username=config.username, password=config.password, port=config.port)

    # On exécute la partie tracking du code
    stdin, stdout, stderr= client.exec_command('/home/foottracker/myenv/bin/python3 /home/foottracker/myenv/FootTracker/Tracking/main.py')
    exit_status = stdout.channel.recv_exit_status()
    if(exit_status==0):
        sftp = client.open_sftp()

        # On récupère le fichier des tracks
        sftp.get(remote_tracks_path, local_tracks_path)

        # On récupère la vidéo
        sftp.get(remote_video_path, local_video_path)

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

    remote_path = '/home/foottracker/myenv/FootTracker/Tracking/'

    
    sftp.put('current.pkl', remote_path+'current.pkl')
    sftp.put(video_path, remote_path+str(video_path))

    # On ferme la connexion
    sftp.close()
    

    client.close()