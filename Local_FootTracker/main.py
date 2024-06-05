import streamlit as st
import pickle
import moviepy.editor as moviepy
from outils import save_video, send_new_video, get_tracks
import uuid
import os


class MyApp():
    # Initialisation
    def __init__(self):
         self.local_tracks_path = None
         self.output_local_mp4_path = None
         self.file = None
         self.rename= False

    def main(self):
        # On met la page en mode large par défault
        st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:")
        print('hello')
        
        uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4"]) # On upload la vidéo
        if uploaded_file is not None: # Si on vient d'upload un fichier
            with st.spinner('Vidéo en cours d\'analyse... C\'est assez long alors allez chercher des popcorn en attendant !'):
                video_bytes = uploaded_file.getvalue() # On récupère la vidéo sous forme de byte

                # Toutes les info à envoyer au SSH
                current = {}                    
                current['unique_code'] = str(uuid.uuid4()) # Créée un code unique
                current['video_path_avi'] = 'video_'+current['unique_code']+'.avi' # Créée un nom de nouvelle vidéo avi unique
                current['video_path_mp4'] = 'video_'+current['unique_code']+'.mp4' # Créée un nom de nouvelle vidéo mp4 unique
                current['tracks_path'] = 'tracks_files/tracks_'+current['unique_code']+'.pkl' # Créée un nom de nouveaux tracks unique

                # On écrit ce qu'il y a à savoir sur la vidéo dans un pkl pour l'envoyer en ssh
                with open('current.pkl', 'wb') as f: 
                    pickle.dump(current,f)
                    f.close()

                # On enregistre la vidéo dans input_vidéos (sous format MP4)
                save_video(video_bytes, 'input_videos/'+current['video_path_mp4'])
                # On l'envoie au traitement via SSH
                send_new_video('input_videos/'+current['video_path_mp4'])

                remote_tracks_path='/home/foottracker/myenv/FootTracker/Tracking/'+current['tracks_path'] # Le chemin d'accès des tracks SSH
                remote_avi_path='/home/foottracker/myenv/FootTracker/Tracking/'+'output_videos/'+current['video_path_avi'] # Chemin d'accès video AVI SSH

                self.local_tracks_path = current['tracks_path'] # Chemin des tracks de la vidéo en local
                output_local_avi_path = 'output_videos/'+current['video_path_avi'] # Chemin vidéo avi local
                self.output_local_mp4_path = 'output_videos/'+current['video_path_mp4'] # Chemin vidéo mp4 local

                # On récupère les tracks et la vidéo annotée via SSH
                get_tracks(remote_tracks_path, self.local_tracks_path, remote_avi_path, output_local_avi_path)
                
                # On convertit la vidéo en MP4
                clip = moviepy.VideoFileClip(output_local_avi_path)
                clip.write_videofile(self.output_local_mp4_path)
            

        app.button()

    @st.experimental_fragment
    def button(self):
        print('test')
        for file in os.listdir('output_videos'):
            if file[-4:] == '.mp4':
                col1, col2, col3 = st.columns(3) # On instancie 2 colonnes
                with col1:
                    if st.button(file, use_container_width=True):
                        with open('tracks_files/tracks_'+file[6:-4]+'.pkl', 'rb') as f:
                            tracks = pickle.load(f)
                            st.session_state['tracks'] = tracks
                            st.session_state['video_path'] = 'output_videos/'+file
                            f.close()
                        st.switch_page("pages/interface.py")
                with col2:
                    st.write(file)
                    if st.button(':red-background[:wastebasket:]', key=str(uuid.uuid4()), on_click=self.delete_file, kwargs=dict(file=file)):
                        pass
                with col3:
                    if st.button(':blue-background[:lower_left_ballpoint_pen:]', key=str(uuid.uuid4()),on_click=self.form, kwargs=dict(file=file)):
                        pass

    def form(self, file=None):
        with st.form("Nouveau nom"):
            self.name = st.text_input("Nouveau nom") 
            print(self.name)
            submitted = st.form_submit_button("Submit", on_click=self.rename_file, kwargs=dict(file=file, new_name=self.name))



    def rename_file(self, file=None, new_name=None):
        print("nom : "+new_name)
        old_file = os.path.join("input_videos", file)
        new_file = os.path.join("input_videos", 'video_'+new_name+'.mp4')
        os.rename(old_file, new_file)

        old_file = os.path.join("output_videos", file)
        new_file = os.path.join("output_videos", 'video_'+new_name+'.mp4')
        os.rename(old_file, new_file)

        old_file = os.path.join("tracks_files", 'tracks_'+file[6:-4]+'.pkl')
        new_file = os.path.join("tracks_files", 'tracks_'+new_name+'.pkl')
        os.rename(old_file, new_file)

        old_file = os.path.join("output_videos", file)
        new_file = os.path.join("output_videos", 'video_'+new_name+'.avi')
        os.rename(old_file, new_file)

    def delete_file(self, file=None):
        os.remove('input_videos/'+file)
        os.remove('output_videos/'+file)
        os.remove('tracks_files/tracks_'+file[6:-4]+'.pkl')
        os.remove('output_videos/'+file[:-4]+'.avi')

if __name__ == '__main__':
        app = MyApp()
        app.main()