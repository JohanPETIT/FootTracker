import os
import streamlit as st

st.set_page_config(layout='wide', page_title="FootTracker", page_icon=":soccer:", initial_sidebar_state="collapsed")

def form():
    name = st.text_input("Nouveau nom") 
    if name:
        old_file = os.path.join("output_videos", st.session_state['file'])
        new_file = os.path.join("output_videos", 'video_'+name+'.mp4')
        os.rename(old_file, new_file)

        old_file = os.path.join("tracks_files", 'tracks_'+st.session_state['file'][6:-4]+'.pkl')
        new_file = os.path.join("tracks_files", 'tracks_'+name+'.pkl')
        os.rename(old_file, new_file)

        old_file = os.path.join("events_files", 'events_'+st.session_state['file'][6:-4]+'.pkl')
        new_file = os.path.join("events_files", 'events_'+name+'.pkl')
        os.rename(old_file, new_file)

        st.switch_page('main.py')

form()