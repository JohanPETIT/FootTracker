import streamlit as st
import pickle
import moviepy.editor as moviepy
from outils import save_video, send_new_video, get_tracks
import uuid


def main():
    # On met la page en mode large par défault
    st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:")
    
    uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4"]) # On upload la vidéo
    if uploaded_file is not None:
        video_bytes = uploaded_file.getvalue()

        current = {}                    
        current['unique_code'] = str(uuid.uuid4()) # Créée un code unique
        current['video_path_avi'] = 'video_'+current['unique_code']+'.avi' # Créée un nom de nouvelle vidéo unique
        current['video_path_mp4'] = 'video_'+current['unique_code']+'.mp4' # Créée un nom de nouvelle vidéo unique
        current['tracks_path'] = 'tracks_files/tracks_'+current['unique_code']+'.pkl' # Créée un nom de nouveaux tracks unique

        with open('current.pkl', 'wb') as f:
            pickle.dump(current,f)
            f.close()

        save_video(video_bytes, 'input_videos/'+current['video_path_mp4'])
        send_new_video('input_videos/'+current['video_path_mp4'])

        remote_tracks_path='/home/foottracker/myenv/FootTracker/Tracking/'+current['tracks_path']
        remote_avi_path='/home/foottracker/myenv/FootTracker/Tracking/'+'output_videos/'+current['video_path_avi']

        local_tracks_path = current['tracks_path']
        output_local_avi_path = 'output_videos/'+current['video_path_avi']
        output_local_mp4_path = 'output_videos/'+current['video_path_mp4']

        get_tracks(remote_tracks_path, local_tracks_path, remote_avi_path, output_local_avi_path)
        
        # On convertit la vidéo en MP4
        clip = moviepy.VideoFileClip(output_local_avi_path)
        clip.write_videofile(output_local_mp4_path)


        if st.button("Page 1"):
            with open(local_tracks_path, 'rb') as f:
                tracks = pickle.load(f)
                st.session_state['tracks'] = tracks
                st.session_state['video_path'] = output_local_mp4_path
                f.close()
            st.switch_page("pages/interface.py")


if __name__ == '__main__':
    main()

