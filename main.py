import streamlit as st
import pandas as pd
from functionality import welcome, data_obs_intro, uploading, formatting_data_intro, ml_regressions_intro
from some_features import FileManagement


def main():

    st.sidebar.title("Navigation")

    # Initialiser l'état des boutons dans st.session_state
    if 'page' not in st.session_state:
        st.session_state.page = "Welcome"

    # Initialiser FileManagement dans st.session_state s'il n'est pas présent
    if 'file_manager' not in st.session_state:
        st.session_state.file_manager = None

    # Stocker dt pour qu'il soit accessible à tout moment
    if 'dt' not in st.session_state:
        st.session_state.dt = None

    # Boutons pour chaque rubrique
    if st.sidebar.button("Welcome"):
        st.session_state.page = "Wellcome"
    if st.sidebar.button("Data observation"):
        st.session_state.page = "Data observation"
    if st.sidebar.button("Data Formatting"):
        st.session_state.page = "Data Formatting"
    if st.sidebar.button("ML/ Regression Models"):
        st.session_state.page = "ML/ Regression Models"

    # Afficher le contenu en fonction de la sélection

    if st.session_state.page == "Welcome":
        welcome()
        if st.session_state.dt is None:  # Charger dt seulement si non défini
            dt = uploading()
            st.session_state.file_manager = FileManagement(dt)
        else:
            dt = st.session_state.dt
            st.session_state.file_manager = FileManagement(dt)

    # Page Data Observation

    if st.session_state.page == "Data observation" and st.session_state.file_manager is not None:
        data_obs_intro()
        st.session_state.file_manager.data_obs()

    # Page Data Formatting

    if st.session_state.page == "Data Formatting" and st.session_state.file_manager is not None:
        formatting_data_intro()
        st.session_state.file_manager.data_formatting(st.session_state.file_manager.get_data())

    # Page ML/ Regression Models

    if st.session_state.page == "ML/ Regression Models" and st.session_state.file_manager is not None:
        ml_regressions_intro()
        df = st.session_state.file_manager.data_clearing(st.session_state.file_manager.get_data())
        if isinstance(df, pd.DataFrame) and df is not None and not df.empty:
            def_fin = df
            st.session_state.file_manager.linear_model(def_fin)
        else:
            st.write("No data-set submitted yet...")


main()
        
