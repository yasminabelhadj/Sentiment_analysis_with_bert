FROM python:3.8.5

RUN pip3 install transformers 
RUN pip3 install fastapi 
RUN pip3 install uvicorn
RUN pip3 install pandas
RUN pip3 install torch
RUN pip3 install streamlit

COPY ./sentiment_analysis  /app/sentiment_analysis

WORKDIR /app/sentiment_analysis
 
CMD  ["uvicorn" , "rest_api:app" ,  "--reload" , "--port" , "8000"]
#CMD ["streamlit" , "run" , "sentiment_analysis/webpage.py"]
