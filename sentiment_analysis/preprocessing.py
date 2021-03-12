

def sentiment(rating) : 
    if (rating == 1 )|(rating == 2) : 
        return 0 
    elif (rating == 3) : 
        return 1 
    elif (rating >= 4) : 
        return 2 

def create_labels(df) : 
    df['sentiment'] = df.score.apply(sentiment)
    return df