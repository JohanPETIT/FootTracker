import streamlit as st
from st_pages import Page, hide_pages, show_pages
import pickle
import moviepy.editor as moviepy
from outils import save_video, send_new_video, get_tracks_and_events
import uuid
import os



class MyApp():
    # Initialisation
    def __init__(self):
         self.local_tracks_path = None # Le path du fichier des tracks local
         self.local_video_path = None # Le path du fichier de la vidéo local
         self.local_events_path = None # Le path du fichier des events local
         self.file = None # Donne le nom de fichier à modifier pour renommage
         self.test = False # teste si n entame un renommage ou non

    def main(self):
        # On met la page en mode large par défault
        st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:", initial_sidebar_state="collapsed")
        print('hello')
        
        uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4"]) # On upload la vidéo
        if uploaded_file is not None: # Si on vient d'upload un fichier
            with st.spinner('Vidéo en cours d\'analyse... C\'est assez long alors allez chercher des popcorn en attendant !'):
                video_bytes = uploaded_file.getvalue() # On récupère la vidéo sous forme de byte

                # Toutes les info à envoyer au SSH
                current = {}                    
                current['unique_code'] = str(uuid.uuid4()) # Créée un code unique
                current['video_path_mp4'] = 'video_'+current['unique_code']+'.mp4' # Créée un nom de nouvelle vidéo mp4 unique
                current['tracks_path'] = 'tracks_files/tracks_'+current['unique_code']+'.pkl' # Créée un nom de nouveaux tracks unique
                current['events_path'] = 'events_files/events_'+current['unique_code']+'.pkl' # Créée un nom de nouveaux tracks unique

                # On écrit ce qu'il y a à savoir sur la vidéo dans un pkl pour l'envoyer en ssh
                with open('current.pkl', 'wb') as f: 
                    pickle.dump(current,f)
                    f.close()

                # On enregistre la vidéo dans input_vidéos (sous format MP4)
                save_video(video_bytes, 'input_videos/'+current['video_path_mp4'])
                # On l'envoie au traitement via SSH
                send_new_video('input_videos/'+current['video_path_mp4'])

                remote_tracks_path='/home/foottracker/myenv/FootTracker/Tracking/'+current['tracks_path'] # Le chemin d'accès des tracks SSH
                remote_video_path='/home/foottracker/myenv/FootTracker/Tracking/'+'output_videos/'+current['video_path_mp4'] # Chemin d'accès video AVI SSH
                remote_events_path='/home/foottracker/myenv/FootTracker/Detection/'+current['events_path'] # Chemin d'accès des events SSH

                self.local_tracks_path = current['tracks_path'] # Chemin des tracks de la vidéo en local
                self.local_video_path = 'output_videos/'+current['video_path_mp4'] # Chemin vidéo mp4 local
                self.local_events_path = current['events_path'] # Chemin des events de la vidéo en local
                

                # On récupère les tracks et la vidéo annotée via SSH
                get_tracks_and_events(remote_tracks_path, self.local_tracks_path, remote_video_path, self.local_video_path, remote_events_path, self.local_events_path)
            
        app.button()

    # Print la liste des vidéos et les boutons pour les renommer/supprimer
    @st.experimental_fragment
    def button(self):
        # On teste si on est en mode renommmage ou non. Si oui on donne le fichier à renommer à la nouvelle page avec le form
        if self.test :
            st.session_state['file'] = self.file
            st.switch_page("pages/form.py")
        
        for file in os.listdir('output_videos'):
            if file[-4:] == '.mp4':
                col1, col2 = st.columns(2) # On instancie 2 colonnes
                with col1:
                    if st.button(file, use_container_width=True):
                        with open('tracks_files/tracks_'+file[6:-4]+'.pkl', 'rb') as f:
                            tracks = pickle.load(f)
                            st.session_state['tracks'] = tracks
                            f.close()
                        with open('events_files/events_'+file[6:-4]+'.pkl', 'rb') as f:
                            events = pickle.load(f)
                            st.session_state['events'] = events
                            f.close()
                        st.session_state['video_path'] = 'output_videos/'+file
                        st.switch_page("pages/interface.py")
                with col2:
                    if st.button(':red-background[:wastebasket:]', key=str(uuid.uuid4()), on_click=self.delete_file, kwargs=dict(file=file)):
                        pass
                    if st.button(':blue-background[:lower_left_ballpoint_pen:]', key=str(uuid.uuid4()),on_click=self.form, kwargs=dict(file=file)):
                        pass

    # Dit si on passe en mode renommage ou non et donne le fichier à renommer
    def form(self, file=None):
        self.test = True
        self.file = file

    # Supprime tous les fichiers associés à une vidéo
    def delete_file(self, file=None):
        os.remove('input_videos/'+file)
        os.remove('output_videos/'+file)
        os.remove('tracks_files/tracks_'+file[6:-4]+'.pkl')
        os.remove('events_files/events_'+file[6:-4]+'.pkl')

if __name__ == '__main__':
        app = MyApp()
        app.main()