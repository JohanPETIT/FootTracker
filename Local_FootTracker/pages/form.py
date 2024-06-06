import os
import streamlit as st
def form():
    name = st.text_input("Nouveau nom") 
    if name:
        old_file = os.path.join("input_videos", st.session_state['file'])
        new_file = os.path.join("input_videos", 'video_'+name+'.mp4')
        os.rename(old_file, new_file)

        old_file = os.path.join("output_videos", st.session_state['file'])
        new_file = os.path.join("output_videos", 'video_'+name+'.mp4')
        os.rename(old_file, new_file)

        old_file = os.path.join("tracks_files", 'tracks_'+st.session_state['file'][6:-4]+'.pkl')
        new_file = os.path.join("tracks_files", 'tracks_'+name+'.pkl')
        os.rename(old_file, new_file)

        old_file = os.path.join("output_videos", 'video_'+st.session_state['file'][6:-4]+'.avi')
        new_file = os.path.join("output_videos", 'video_'+name+'.avi')
        os.rename(old_file, new_file)

        st.switch_page('main.py')

form()