from fastapi import FastAPI 
import uvicorn
from pydantic import BaseModel, conlist
from fastapi.responses import JSONResponse
from typing import Any, Collection, Dict, List, Optional, Union
from classifier import sentiment_analysis , load_model
from datasilo_for_prediction import create_data_loader
import pandas as pd
#from utils import get_sentiment


app = FastAPI(title = 'Sentiment analysis')

class review_sentiment(BaseModel):
    review: str 
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None
    name_model : str
    n_class: int = 0
   

@app.post('/predict')
async def get_answer(data : review_sentiment) : 
    review = dict(data)['review']
    name_model = dict(data)['name_model']
    n_class = dict(data)['n_class']
    if n_class != 0 :
        
        Model = sentiment_analysis(n_class , name_model)

    else : 

        Model = load_model(name_model)
    
    d = {'content' : [str(review)] }
    df = pd.DataFrame(d)

    
    tokenizer = Model.tokenizer
    dataloader = create_data_loader (df, tokenizer, max_len = 32, batch_size = 8)

    def get_sentiment(preds ): 
        if preds == '0' : 
            return 'Negative'
        elif preds == '1' : 
            return 'Neutral' 
        elif preds == '2' : 
            return 'Positive'

    for data in dataloader : 

        input_ids = data['input_ids'] 

        attention_mask = data['attention_mask']

        prediction = Model.predict ( input_ids , attention_mask) 
        sentiment = get_sentiment(str(prediction.numpy()[0]))
    

    
    return JSONResponse (sentiment)
    

#if __name__ == '__main__':
    #uvicorn.run(app, host='127.0.0.1', port=8000)