import streamlit as st
from plotly_football_pitch import make_pitch_figure, PitchDimensions, SingleColourBackground

#Empêche de réexécuter tout le code dès qu'on clique sur qqc
@st.experimental_fragment
# Dessine la page
def plot_page(video_path):

    # On charge la vidéo
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()

    # On initialise 2 colonnes pour avoir les stats à coté de la vidéo
    col1, col2 = st.columns(2)

    # Vidéo colonne gauche
    with col1:
        st.video(video_bytes, autoplay=True, muted=True)

    # Stats colonne droite
    with col2:
        # On dessine le truc pour sélectionner
        option = st.selectbox("Quelle statistique vous intéresse ?", ("Possession", "Position du ballon", "Autre"), index=None, placeholder="Choisissez une option !")
        
        # Possession 
        if(option == "Possession"):
            st.write('test')

        # Position du ballon 
        if(option == "Position du ballon"):
            # On dessine le terrain
            dimensions = PitchDimensions()
            fig = make_pitch_figure(dimensions, pitch_background=SingleColourBackground("#74B72E"))
            st.plotly_chart(fig)

    