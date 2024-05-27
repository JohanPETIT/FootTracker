import streamlit as st
from plotly_football_pitch import make_pitch_figure, PitchDimensions, SingleColourBackground


@st.experimental_fragment
def plot_page(video_path):

    # On charge la vidéo
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, autoplay=True)

    # On dessine le truc pour sélectionner
    option = st.selectbox("Quelle statistique vous intéresse ?", ("Possession", "Autre", "Autre"))

    # On dessine le terrain
    dimensions = PitchDimensions()
    fig = make_pitch_figure(dimensions, pitch_background=SingleColourBackground("#74B72E"))
    st.plotly_chart(fig)