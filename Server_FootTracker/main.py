moi
import streamlit as st
import io
import pickle
import uuid
import os
import shutil
from outils import save_video, send_new_video, get_tracks_and_events

class MyApp():
    def __init__(self):
        self.local_tracks_path = None
        self.local_input_dir_path = '/media/louis/0942120d-db71-4a3d-ab0d-413b70a189f9/input_videos/'
        self.local_output_dir_path = '/media/louis/0942120d-db71-4a3d-ab0d-413b70a189f9/output_videos/'
        self.local_output_video_path = None
        self.local_events_path = None
        self.file = None
        self.test = False

    def main(self):
        st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:", initial_sidebar_state="collapsed")
        
        uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4"])
        if uploaded_file is not None:
            with st.spinner('Vidéo en cours d\'analyse... C\'est assez long alors allez chercher des popcorn en attendant !'):
                # Initialize variables for remote paths
                current = {}
                current['unique_code'] = str(uuid.uuid4())
                current['video_path_mp4'] = 'video_'+current['unique_code']+'.mp4'
                current['tracks_path'] = 'tracks_files/tracks_'+current['unique_code']+'.pkl'
                current['events_path'] = 'events_files/events_'+current['unique_code']+'.pkl'

                # Save current info to a pickle file
                with open('current.pkl', 'wb') as f:
                    pickle.dump(current, f)

                # Save video to local input directory
                video_bytes = uploaded_file.read()  # Read the entire file into memory initially

                # Save video using chunking
                chunk_size = 500 * 1024  # 500 KB chunks (adjust as needed)
                chunk_number = 1
                with io.BytesIO(video_bytes) as video_buffer:
                    while True:
                        chunk = video_buffer.read(chunk_size)
                        if not chunk:
                            break
                        # Save chunk to local input directory
                        save_video(chunk, f"{self.local_input_dir_path}part_{chunk_number}_{current['video_path_mp4']}")
                        chunk_number += 1

                # Send the first chunk for processing via SSH
                send_new_video(self.local_input_dir_path + f"part_1_{current['video_path_mp4']}", current['video_path_mp4'])

                remote_tracks_path = '/home/foottracker/myenv/FootTracker/Tracking/' + current['tracks_path']
                remote_video_path = '/home/foottracker/myenv/FootTracker/Tracking/' + 'output_videos/' + current['video_path_mp4']
                remote_events_path = '/home/foottracker/myenv/FootTracker/Detection/' + current['events_path']

                self.local_tracks_path = current['tracks_path']
                self.local_events_path = current['events_path']

                # Get tracks and annotated video via SSH (assuming function handles chunking internally)
                get_tracks_and_events(remote_tracks_path, self.local_tracks_path, remote_video_path,
                                      self.local_output_dir_path + current['video_path_mp4'], remote_events_path,
                                      self.local_events_path)

                # Clean up input directory
                for filename in os.listdir(self.local_input_dir_path):
                    file_path = os.path.join(self.local_input_dir_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

        self.button()

    @st.experimental_fragment
    def button(self):
        if self.test:
            st.session_state['file'] = self.file
            st.switch_page("pages/form.py")

        for file in os.listdir(self.local_output_dir_path):
            if file.endswith('.mp4'):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(file, use_container_width=True):
                        with open(f'tracks_files/tracks_{file[6:-4]}.pkl', 'rb') as f:
                            tracks = pickle.load(f)
                            st.session_state['tracks'] = tracks
                        with open(f'events_files/events_{file[6:-4]}.pkl', 'rb') as f:
                            events = pickle.load(f)
                            st.session_state['events'] = events
                        st.session_state['video_path'] = os.path.join(self.local_output_dir_path, file)
                        st.switch_page("pages/interface.py")
                with col2:
                    if st.button(':wastebasket:', key=str(uuid.uuid4()), on_click=self.delete_file, kwargs=dict(file=file)):
                        pass
                    if st.button(':lower_left_ballpoint_pen:', key=str(uuid.uuid4()), on_click=self.form, kwargs=dict(file=file)):
                        pass


    def form(self, file=None):
        self.test = True
        self.file = file

    def delete_file(self, file=None):
        os.remove(os.path.join(self.local_output_dir_path, file))
        os.remove(f'tracks_files/tracks_{file[6:-4]}.pkl')
        os.remove(f'events_files/events_{file[6:-4]}.pkl')

if __name__ == '__main__':
    app = MyApp()
    app.main()
