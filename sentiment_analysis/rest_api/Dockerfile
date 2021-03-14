FROM python:3.8.5

RUN pip3 install transformers 
RUN pip3 install fastapi 
RUN pip3 install uvicorn
RUN pip3 install pandas
RUN pip3 install torch

COPY .  /app/rest_api

WORKDIR /app/rest_api

EXPOSE 8000
 
CMD  ["uvicorn" , "rest_api:app" ,  "--reload" , "--port" , "8000"]
