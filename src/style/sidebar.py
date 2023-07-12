import streamlit as st
from PIL import Image
import base64
from pathlib import Path
import os

@st.cache_data(show_spinner=False)
def render_sidebar():
    current_path = os.getcwd()
    logo_path = current_path + "/img/cmv_logo.png"
    img_path = current_path + "/img/artificial-intelligence-vs-human-intelligence.png"
    with open(logo_path, "rb") as f:
        logo = base64.b64encode(f.read()).decode("utf-8")
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read()).decode("utf-8")
    
    sidebar_markdown = f"""
    
    <center>
    <img src="data:image/png;base64,{logo}" width="75" height="75" />
    <h1>
    About
    </h1>  
    &nbsp;    
    </center>
    
    <hr>
    
    <p>
    A human-AI competition project from Webis Group aiming to explore the argumentative capacity of hot-debated LLM.
    </p> 
    
    <center>
    <img src="data:image/png;base64,{img}"  alt="Argument" width="250">
    </center>
    
    <center>
    <a href="https://github.com/gary8564">
    <img src = "https://cdn-icons-png.flaticon.com/512/733/733609.png" width="23" /></a>
    
    <a href="mailto:chia.hao.chang@uni-weimar.de">
    <img src="https://cdn-icons-png.flaticon.com/512/646/646094.png" alt="email" width = "27" ></a>
    </center>
    
    &nbsp;


    """

    st.sidebar.markdown(sidebar_markdown, unsafe_allow_html=True)
    
    