import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st 
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI 
import openai
import textwrap
import pandas as pd
import numpy as np
import random
import re
from PIL import Image
import base64
from src.config import *
from src.session_state import session
from src.style.sidebar import render_sidebar 
from src.callbacks import clicked

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title=PAGE_TITLE, page_icon=Image.open(PAGE_ICON))
css_file = "./src/style/style.css"
# Design style
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# Initialize state
session.init("winner", "")
session.init("status", 0)
session.init("arguAI", "")
session.init("arguHuman", "")
session.init("index", -1)

def falcon_agi(topic, stance):
    # Load the HuggingFaceHub API token from the .env file
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Load the LLM model from the HuggingFaceHub
    repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.9, "max_new_tokens": 500}
    )
    template="You are a debate master that can produce a persuasive argument \
              to convince people to change their viewpoints. \
              Now, you are going to take part in a debate competition on a online discussion forum. \
              Given the debate issue {topic}, please write a succinct and compelling argument {stance}. \
              The argument should be less than 30 words.\
              Please avoid exposing your AI identity. \
              That is, do not begin your argument with `as an AI language model` \
              nor write anything irrelevant to the topic."
              
    prompt = PromptTemplate(
                input_variables=["topic", "stance"],
                template=template,
             ) 
    print(prompt.format(topic=topic, stance=stance))
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)
    response = llm_chain.run({
        'topic': topic,
        'stance': stance
    })
    wrapped_text = textwrap.fill(
        response, width=100, break_long_words=False, replace_whitespace=False
    )
    return wrapped_text
 
def gorilla_agi(topic, stance):
    openai.api_key = "EMPTY" # Key is ignored and does not matter
    openai.api_base = "http://34.132.127.197:8000/v1"
    def get_gorilla_response(prompt="I would like to translate from English to French.", model="gorilla-7b-hf-v0"):
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    template="You are a debate master that can produce a persuasive argument \
              to convince people to change their viewpoints. \
              Now, you are going to take part in a debate competition on a online discussion forum. \
              Given the debate issue {topic}, please write a succinct and compelling argument {stance}. \
              The argument should be less than 50 words.\
              To be more persuasive, try to give a attractive hook or some statistical facts in your argument. \
              Please avoid exposing your AI identity. \
              Do not include `as an AI language model` in your answer. \
              Do not write anything irrelevant to the topic."
              
    prompt = PromptTemplate(
                input_variables=["topic", "stance"],
                template=template,
             ) 
    prompt = prompt.format(topic=topic, stance=stance)
    print(prompt)
    response = get_gorilla_response(prompt, model="gorilla-7b-hf-v0" )
    wrapped_text = textwrap.fill(
        response, width=100, break_long_words=False, replace_whitespace=False
    )
    return wrapped_text

def gen_counter_arguAI(input_topic, input_stance):
    assert input_stance in ALL_STANCES_OPTIONS
    if input_stance in ["Agree", "Disagree"]:
        stance = "in favor of the topic statement" if input_stance == "Disagree" else "in opposition to the topic statement"
        return gorilla_agi(input_topic, stance)
    elif input_stance in ["Christianity", "Atheism"]:
        stance = "in favor of Christianity" if input_stance == "Atheism" else "in favor of Atheism"
        return gorilla_agi(input_topic, stance)
    elif input_stance in ["Evolution", "Creation"]:
        stance = "in favor of Creation" if input_stance == "Evolution" else "in favor of Evolution"
        return gorilla_agi(input_topic, stance)
    else:
        stance = "in favor of Personal persuit" if input_stance == "Advancing the common good" else "in favor of Personal persuit"
        return gorilla_agi(input_topic, stance)

def gen_counter_arguHuman(input_topic, input_stance):
    corpus_path = "./data/dagstuhl-15512-argquality-corpus-annotated.csv"
    match_index = ARGU_TOPIC_OPTIONS_DISPLAY.index(input_topic)
    df_human = pd.read_csv(corpus_path, encoding='latin-1', sep='\t')
    df_human = df_human[["issue", "argument", "stance"]]
    df_topic = df_human[df_human["issue"] == ARGU_TOPIC_OPTIONS_DB[match_index]]
    df_topic = df_topic.dropna(how='all')
    df_topic_counter = df_topic
    if input_stance in ["Christianity", "Atheism"]:
        if input_stance == "Atheism": 
            df_topic_counter = df_topic[df_topic["stance"] == "christianity"]
        else:
            df_topic_counter = df_topic[df_topic["stance"] == "atheism"]
            
    elif input_stance in ["Evolution", "Creation"]:
        if input_stance == "Evolution": 
            df_topic_counter = df_topic[df_topic["stance"] == "creation"]
        else:
            df_topic_counter = df_topic[df_topic["stance"] == "evolution"]
            
    elif "gay" in input_topic.lower():
        if input_stance == "Agree": 
            df_topic_counter = df_topic[df_topic["stance"] == "allowing-gay-marriage-is-wrong"]
        else:
            df_topic_counter = df_topic[df_topic["stance"] == "allowing-gay-marriage-is-right"]
            
    elif "father" in input_topic.lower():
        if input_stance == "Agree": 
            df_topic_counter = df_topic[df_topic["stance"] == "fatherless"]
        else:
            df_topic_counter = df_topic[df_topic["stance"] == "lousy-father"]
            
    elif input_stance in ["Personal persuit", "Advancing the common good"]:
        if input_stance == "Personal persuit": 
            df_topic_counter = df_topic[df_topic["stance"] == "advancing-the-common-good"]
        else:
            df_topic_counter = df_topic[df_topic["stance"] == "personal-pursuit"]
            
    else: 
        if input_stance == "Agree":
            df_topic_counter = df_topic[df_topic["stance"].str.contains("no")]
        else:
            df_topic_counter = df_topic[df_topic["stance"].str.contains("yes")]
    df_topic_counter_argu = df_topic_counter["argument"]
    df_topic_counter_argu = df_topic_counter_argu.iloc[random.randint(0, df_topic_counter_argu.size-1)]
    df_topic_counter_argu = df_topic_counter_argu.replace("</br>","")
    df_topic_counter_argu = df_topic_counter_argu.replace("<br/>","")
    re.sub(r'^https?:\/\/.*[\r\n]*', '', df_topic_counter_argu)

    wrapped_text = textwrap.fill(
        str(df_topic_counter_argu), width=100, break_long_words=False, replace_whitespace=False
    )
    return wrapped_text

def main_page(results):
    with st.sidebar:
        render_sidebar()
    
    st.image('./img/ArgGenius.png')  # TITLE and Creator information
    st.write('\n')  # add spacing

    input_topic = st.selectbox('Select topics',
                                ARGU_TOPIC_OPTIONS_DISPLAY,
                                index=0,
                                key="topic",
                                on_change=clicked,
                                args=(0,))
    input_stance = ""
    if "Christianity or Atheism" in input_topic:
        input_stance = st.radio(
            label="Select your stance",
            options=("Christianity", "Atheism"),
            horizontal=True,
            key="stance",
            on_change=clicked,
            args=(0,)
        )
    elif "Evolution versus Creation" in input_topic:
        input_stance = st.radio(
            label="Select your stance",
            options=("Evolution", "Creation"),
            horizontal=True,
            key="stance",
            on_change=clicked,
            args=(0,)
        )
    elif "Personal persuit or Advancing the common good" in input_topic:
        input_stance = st.radio(
            label="Select your stance",
            options=("Personal persuit", "Advancing the common good"),
            horizontal=True,
            key="stance",
            on_change=clicked,
            args=(0,)
        )
    else:
        input_stance = st.radio(
            label="Select your stance",
            options=("Agree", "Disagree"),
            horizontal=True,
            key="stance",
            on_change=clicked,
            args=(0,)
        )
    st.write("\n")  # add spacing
    topic = session.get('topic')
    stance = session.get('stance')
    arguAI = session.get("arguAI")
    arguHuman = session.get("arguHuman")
    if session.get("status") < 2 or session.get("index") == -1:
        index = random.randint(0, 1)
        session.update("index", index)
    index = session.get("index")
    st.button('Generate', on_click=clicked, args=(1,))
    if session.get("status") > 0:
        if session.get("status") == 1:
            with st.spinner('Processing...'):
                input_contents = []  
                if (input_topic != "") and (input_stance != ""):
                    input_contents.append(str(input_topic))
                    input_contents.append(str(input_stance)) 
                if (len(input_contents) < 2):  # remind user to provide data 
                    st.error('Please select all required fields!') 
                else:  # initiate gen process  
                    arguAI = gen_counter_arguAI(input_topic,
                                            input_stance)
                    arguHuman = gen_counter_arguHuman(input_topic, input_stance)
        
        if arguAI != "" and arguHuman != "":
            session.update("arguAI", arguAI)
            session.update("arguHuman", arguHuman)
            gen_argus = [arguAI, arguHuman]
            st.write('\n')  # add spacing
            subheader = st.empty()
            subheader.markdown('\n### Here are two counterarguments!\n')
            arg1, arg2 = st.columns(2) 
            with arg1:
                arg1_title = st.empty()
                arg1_placeholder = st.empty()
                arg1_button = st.empty()
            
            with arg2:
                 arg2_title = st.empty()
                 arg2_placeholder = st.empty()
                 arg2_button = st.empty()
            hint = st.empty()
            
            arg1_title.markdown("#### Argument 1")
            arg2_title.markdown("#### Argument 2")
            arg1_placeholder.write(gen_argus[index])
            arg2_placeholder.write(gen_argus[1-index])
            button_label = ":+1:"
            arg1_button.button(label=button_label, key="arg1", on_click=clicked, args=(2,))
            arg2_button.button(label=button_label, key="arg2", on_click=clicked, args=(3,))
            hint.write("\nPlease thumb up the one you feel more convinced!\n")            
            if session.get("status") > 1:
                subheader.empty()
                arg1_title.empty()
                arg2_title.empty()
                hint.empty()
                arg1_placeholder.empty()
                arg2_placeholder.empty()
                arg1_button.empty()
                arg2_button.empty()
                if session.get("status") == 2:
                    imagepath = './img/AI_win.png' if index == 0 else './img/Human_win.png'
                    with open(imagepath, "rb") as f:
                        image = base64.b64encode(f.read()).decode("utf-8")
                    winner = "AI " if index == 0 else "Human"
                    session.update("winner", winner)
                    text = f'{winner} win!!!'
                    st.balloons()
                    st.markdown(f"""
                                <center>
                                    <img src="data:image/png;base64,{image}" />
                                    <p>
                                        {text}
                                    </p>  
                                    &nbsp;    
                                </center>
                                """, unsafe_allow_html=True)
                        
                if session.get("status") == 3:
                    imagepath = './img/Human_win.png' if index == 0 else './img/AI_win.png'
                    with open(imagepath, "rb") as f:
                        image = base64.b64encode(f.read()).decode("utf-8")
                    winner = "Human " if index == 0 else "AI"
                    session.update("winner", winner)
                    text = f'{winner} win!!!'
                    st.balloons()
                    st.markdown(f"""
                                <center>
                                    <img src="data:image/png;base64,{image}" />
                                    <p>
                                        {text}
                                    </p>  
                                    &nbsp;    
                                </center>
                                """, unsafe_allow_html=True)
                    
            new_sample = {
                'issue': session.get('topic'),
                'stance': session.get('stance'),
                'arguAI': session.get('arguAI'),
                'arguHuman': session.get('arguHuman'),
                'Winner': session.get("winner")
            }
            results.append(new_sample)
            
            st.write("randm index for block1:", index)
            st.write("button state:", session.get("status"))
            st.write("results:", results)
    st.button("Reset", key="reset", on_click=clicked, args=(0, True, results))  
                    
if __name__ == '__main__':
    # call main function
    result_db = []
    main_page(result_db)
                
                

            