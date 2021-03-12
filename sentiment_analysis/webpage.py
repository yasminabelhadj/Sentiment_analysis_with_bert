import streamlit as st
from annotated_text import annotated_text
import os
import requests
from transformers import BertModel
from utils import run   , run_local_model
from classifier import load_model

st.write("# Sentiment analysis using bert")
st.sidebar.header("Options")
n_class = st.sidebar.slider("Max. number of classes", min_value=1, max_value=10, value=3, step=1)
local_model = st.sidebar.checkbox('Upload model from local folder')
hugging_face_model = st.sidebar.checkbox('Bring model from huggingface hub')

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
url = API_ENDPOINT +'/predict'

review = st.text_input("Please provide your review:", value="This movie is super bad and the worst thing I've ever watched")
run_query = st.button("Run")

if local_model and not hugging_face_model : 
    n_class = 0
    path = st.sidebar.text_input('Please enter the full path to the folder of the classification model.')
    get_local_model = st.sidebar.button("Get_model from disk")
    
    if get_local_model :
        with st.spinner('Getting model from disk') : 
            local_model_ = load_model(path)
        if local_model_ != None :
            st.write('Got model')
        else : 
            st.write('Model doesn not exists in the directory provided')
        print(run_query)
    if run_query : 
            print('hello')
            run_local_model(review = review , path = path  )
        


elif hugging_face_model and not local_model : 
    name_model = st.sidebar.text_input('Please enter the name of the model from huggingface hub' , value = 'bert-base-cased')
    get_model = st.sidebar.button("Get_model from hub")
    #This is to check if the model_name we typed exists in huggingface or not
    model = BertModel.from_pretrained (name_model)
    if get_model and model == None : 
        st.write('Can not load model from huggingface, please make sure')
    else : 
        if run_query:
            run(review = review,name_model= name_model, n_class=n_class )
