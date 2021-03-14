FROM python:3.8.5

RUN pip3 install pandas
RUN pip3 install streamlit
RUN pip3 install requests


COPY .  /app/UI

WORKDIR /app/UI

EXPOSE 8501
 
CMD  ["streamlit" ,"run" , "webpage.py"]
