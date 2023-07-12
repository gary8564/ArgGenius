import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st 
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI 
from langchain.prompts.few_shot import FewShotPromptTemplate
import openai
from getpass import getpass
import textwrap
import pandas as pd
import numpy as np
import random
import re
import time
from PIL import Image
import base64
from src.config import *
from src.session_state import session
from src.style.sidebar import render_sidebar 
from src.callbacks import clicked
from src.utils import load_data

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

@st.cache_data(show_spinner=False)
def fewShotPromptTemplate():
    # create examples from ChatGPT
    examples = [
        {
            "query": "Given the debate issue `gay marriage`, please write a succinct and compelling argument in opposition to the issue. \
                      The argument should write in a conversational manner and be less than 50 words. \
                      Please avoid exposing your AI identity, such as mentioning you are an AI model in your answer.",
            "answer": 
                """
                It is crucial to recognize that arguments against gay marriage have been widely discredited and are no longer supported by the majority of society. \
                Nonetheless, I can present a historical argument that have been used in the past, \
                but please remember that this argument do not reflect my personal beliefs or values. \
                Marriage has historically been defined as the union between one man and one woman. \
                Changing this definition undermines the institution of marriage and its long-standing cultural, religious, and social significance. \
                Also, marriage is primarily intended for procreation and the raising of children. \
                Same-sex couples cannot naturally conceive children and therefore should not be allowed to marry.
                """
        },
        {
            "query": "Given the debate issue `ban plastic water bottles`, please write a succinct and compelling argument in opposition to the issue. \
                      The argument should write in a conversational manner and be less than 50 words. \
                      Please avoid exposing your AI identity, such as mentioning you are an AI model in your answer.",
            "answer": 
                """
                Banning plastic water bottles might seem like a good idea, but banning plastic water bottles isn't the solution. \
                Did you know that plastic bottles only make up 3% of total waste? \
                According to the EPA, paper and cardboard contribute way more! \
                Plus, it could negatively impact vulnerable communities that rely on bottled water. \
                Let's focus on recycling and tackling the bigger sources of waste instead. (Source: United States Environmental Protection Agency)
                """
        },
        {
            "query": "Given the debate issue `If your spouse committed murder and he or she confided in you, you would turn them in.`, \
                      please write a succinct and compelling argument in opposition to the issue. \
                      The argument should write in a conversational manner and be less than 50 words. \
                      Please avoid exposing your AI identity, such as mentioning you are an AI model in your answer.",
            "answer":
                """
                No way, man! If my spouse committed murder and told me, I couldn't just turn them in. \
                Love and loyalty run deep, but there are logical reasons too. \
                Snitching might put me in danger and tear our family apart. Let the justice system do its thing, \
                but I won't be the one to break the bond.
                """
        },
        {
            "query": "Given the debate issue `Christianity or Atheism`, please write a succinct and compelling argument in opposition to the issue. \
                      The argument should write in a conversational manner and be less than 50 words. \
                      Please avoid exposing your AI identity, such as mentioning you are an AI model in your answer.",
            "answer":
                """
                Christianity offers hope, purpose, and a moral compass. Its teachings promote love, forgiveness, and the value of every individual.\
                The transformative power of faith and the promise of eternal life provide comfort and guidance in navigating life's challenges.\
                It's all about living a meaningful life and making a positive impact. Who wouldn't want that?
                """
        }
    ]
        
    # create a example template
    example_template = """
    Layman: {query}
    Expert: {answer}
    """
    
    # create a prompt example from above template
    example_prompt = PromptTemplate(input_variables=["query", "answer"], template=example_template)
    
    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """
    You are the argument writing expert that can help laymen to generate a persuasive argument \
    to convince people to change their viewpoints. \
    Here are some examples exerpted from your conversations with a layman: 
    """
    # and the suffix our user input and output indicator
    suffix = """
    Layman: {query} 
    Expert: 
    """
    
    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )
    
    return few_shot_prompt_template

@st.cache_resource(show_spinner=False)
def load_llm_models():
    # Load the HuggingFaceHub API token from the .env file
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    # Load the LLM model from the HuggingFaceHub
    repo_id_1 = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    repo_id_2 = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    falcon = HuggingFaceHub(repo_id=repo_id_1, 
                         model_kwargs={"temperature": 1.0, "max_new_tokens": 500})
    flan = HuggingFaceHub(repo_id=repo_id_2, 
                          model_kwargs={"temperature": 1.0, "max_new_tokens": 500})
    gpt = OpenAI(model_name="text-davinci-003", temperature=0.9)
    return falcon, flan, gpt

def agi(topic, stance, model="gpt", fewshot=False):
    assert model in ["gpt", "falcon", "flan"]
    falcon, flan, gpt = load_llm_models()
    llm = gpt 
    if model == "falcon":
        llm = falcon
    elif model == "flan":
        llm = flan    
    if fewshot:
        query = f"Given the debate issue {topic}, please write a concise and compelling argument {stance}. \
                  Please write in less than 30 words and in a more conversational way. \
                  "
        few_shot_prompt_template = fewShotPromptTemplate()
        print(few_shot_prompt_template.format(query=query))
        llm_chain = LLMChain(prompt=few_shot_prompt_template, llm=llm)
        response = llm_chain.run(query)
        wrapped_text = textwrap.fill(response, width=100, break_long_words=False, replace_whitespace=False)
        return wrapped_text
    else: 
        template="You are a debate expert that can produce a persuasive argument \
              to convince people to change their viewpoints. \
              Now, given the debate issue {topic}, \
              please write a succinct and compelling argument {stance}. \
              Your answer should be in less than 50 words and in a more informal, conversational way.\
              Also, in your argument please mention convincing and logical reasons or citations from studies or websites to support your claim. \
              If citations are used, please mentioned where you cite them. \
              Avoid exposing your AI identity in your answer, such as mentioning you are an AI model in your answer. \
              "
        prompt = PromptTemplate(
                    input_variables=["topic", "stance"],
                    template=template,
                ) 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({
                    'topic': topic,
                    'stance': stance
                    })
        wrapped_text = textwrap.fill(response, width=100, break_long_words=False, replace_whitespace=False)
        return wrapped_text

def gen_counter_arguAI(input_topic, input_stance):
    assert input_stance in ALL_STANCES_OPTIONS
    if input_stance in ["Agree", "Disagree"]:
        stance = "in favor of the issue" if input_stance == "Disagree" else "in opposition to the issue"
        return agi(input_topic, stance)
    elif input_stance in ["Christianity", "Atheism"]:
        stance = "in favor of Christianity" if input_stance == "Atheism" else "in favor of Atheism"
        return agi(input_topic, stance)
    elif input_stance in ["Evolution", "Creation"]:
        stance = "in favor of Creation" if input_stance == "Evolution" else "in favor of Evolution"
        return agi(input_topic, stance, model="falcon")
    else:
        stance = "in favor of Personal persuit" if input_stance == "Advancing the common good" else "in favor of Advancing the common good"
        return agi(input_topic, stance, model="falcon")

@st.cache_data(show_spinner=False)
def load_arguHuman_data():
    corpus_path = "./data/dagstuhl-15512-argquality-corpus-annotated.csv"
    df_human = pd.read_csv(corpus_path, encoding='latin-1', sep='\t')
    df_human = df_human[["issue", "argument", "stance"]]
    return df_human

def gen_counter_arguHuman(input_topic, input_stance):
    df_human = load_arguHuman_data()
    match_index = ARGU_TOPIC_OPTIONS_DISPLAY.index(input_topic)
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
    topic_counter_argu = df_topic_counter_argu.iloc[random.randint(0, df_topic_counter_argu.size-1)]
    topic_counter_argu = topic_counter_argu.replace("</br>","")
    topic_counter_argu = topic_counter_argu.replace("<br/>","")
    if re.search('(^http://[\w\s\.\/]*)', topic_counter_argu): 
        topic_counter_argu = topic_counter_argu.replace(" ", "")
        topic_counter_argu = topic_counter_argu.replace(" ", "")
    wrapped_text = textwrap.fill(
        str(topic_counter_argu), width=100, break_long_words=False, replace_whitespace=False
    )
    return wrapped_text

@st.cache_data(show_spinner=False)
def design_page():
    with st.sidebar:
        render_sidebar()
    
    st.image('./img/ArgGenius.png')  # TITLE and Creator information
    st.write('\n')  # add spacing

def main_page():
    design_page()
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
    if os.path.isfile(SAVE_FILEPATH):
        results = load_data()
        session.update("results", results)
    if session.get("status") == -1:
        with st.spinner('Showing current leaderboard...'):
            if not session.get("results"):
                st.warning("No competition results yet. Please provide thumb-up feedbacks for us first.", icon="⚠️")
            else:
                df_template = pd.DataFrame({
                        "Participant name": ["Human", "AI"],
                        "Score": [0, 0]
                    })
                df_results = pd.DataFrame(results)
                df = df_results["Winner"].value_counts().reset_index()
                df.columns = ['Participant name', 'Score']
                leaderboard = pd.concat([df,df_template]).groupby('Participant name').agg('max').reset_index()
                leaderboard = leaderboard.sort_values(by='Score', ascending=False)
                st.dataframe(leaderboard, use_container_width=True, hide_index=True)
                time.sleep(0.25)
    if (session.get("status") > 0 and session.get("status") < 2) or session.get("index") == -1:
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
                    with st.spinner('The winner goes to...'):
                        time.sleep(0.5)
                    imagepath = './img/AI_win.png' if index == 0 else './img/Human_win.png'
                    with open(imagepath, "rb") as f:
                        image = base64.b64encode(f.read()).decode("utf-8")
                    winner = "AI" if index == 0 else "Human"
                    session.update("winner", winner)
                    text = f'{winner}!!!'
                    st.markdown(f"""
                                <center>
                                    <img src="data:image/png;base64,{image}" />
                                    <p>
                                        {text}
                                    </p>  
                                    &nbsp;    
                                </center>
                                """, unsafe_allow_html=True)
                    st.balloons()
                        
                if session.get("status") == 3:
                    with st.spinner('The winner goes to...'):
                        time.sleep(1.5)
                    imagepath = './img/Human_win.png' if index == 0 else './img/AI_win.png'
                    with open(imagepath, "rb") as f:
                        image = base64.b64encode(f.read()).decode("utf-8")
                    winner = "Human" if index == 0 else "AI"
                    session.update("winner", winner)
                    text = f'{winner}!!!'
                    st.markdown(f"""
                                <center>
                                    <img src="data:image/png;base64,{image}" />
                                    <p>
                                        {text}
                                    </p>  
                                    &nbsp;    
                                </center>
                                """, unsafe_allow_html=True)
                    st.balloons()
                    
                new_sample = {
                    'Issue': session.get('topic'),
                    'Stance': session.get('stance'),
                    'ArguAI': session.get('arguAI'),
                    'ArguHuman': session.get('arguHuman'),
                    'Winner': session.get("winner")
                }
                if not session.has("results"):
                    session.init("results", [])
                session.get("results").append(new_sample)
            
    st.button("Show Leaderboard", key="leaderboard", on_click=clicked, args=(-1, False, session.get("results")))
    st.button("Reset", key="reset", on_click=clicked, args=(0, True, session.get("results")))
    #st.write("randm index for block1:", index)
    #st.write("button state:", session.get("status"))
    #st.write("results:", session.get("results"))
                    
if __name__ == '__main__':
    # call main function
    main_page()
                
                

            