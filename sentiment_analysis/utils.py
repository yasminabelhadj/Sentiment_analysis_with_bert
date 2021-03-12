

import requests
import streamlit as st
import os

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
url = API_ENDPOINT +'/predict'


def format_request(review, name_model, n_class = None, filters=None):
    if filters == None:
        return {
       "review": review,
       "name_model": name_model,
       "n_class": n_class  
       }
    return {
        "review": review,
        "filters": {
            "option1":[filters]
        },
        "name_model": name_model,
        "n_class": n_class 
    } 


def run(review , name_model, n_class , url = url) : 
        with st.spinner("Analysing sentiments from given text"):
            req = format_request(review, name_model= name_model, n_class=n_class)
            data = {'review' : req['review'] , 'name_model' : req['name_model']  , 'n_class' : req['n_class'] }
            predictions = requests.post(url, json =data ).json()
            
        st.write(predictions)


def run_local_model(review , path, n_class = 0,  url = url) : 
    with st.spinner("Analysing sentiments from given text"):
        req = format_request(review, name_model= path , n_class= n_class )
        data = {'review' : req['review'] , 'name_model' : req['name_model'] , 'n_class' : req['n_class']  }
        predictions = requests.post(url, json =data ).json()
            
    st.write(predictions)



def get_sentiment(preds ): 
    if preds == '0' : 
        return 'Negative'
    elif preds == '1' : 
        return 'Neutral' 
    elif preds == '2' : 
        return 'Positive'