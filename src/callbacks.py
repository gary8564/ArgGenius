import streamlit as st
from .session_state import session
from .utils import store_data
        
def clicked(stage, reset=False, results=None):
    session.update("status", stage)
    if results:
        store_data(results)
    if reset:
        session.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        #st.experimental_rerun()
    