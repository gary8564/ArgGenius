import streamlit as st
from .session_state import session
from .utils import store_data
        
def clicked(stage, reset=False, results=None):
    session.update("status", stage)
    if reset:
        store_data(results)
        session.clear()
        #st.experimental_rerun()
    