import argparse
import pandas as pd
from trainer import Trainer
from datasilo_for_prediction import create_data_loader
from classifier import sentiment_analysis , load_model


# Define the parser
parser = argparse.ArgumentParser(description='' )
parser.add_argument('--model_name', action="store", dest='model_name' , default = 'bert-base-cased', help = 'enter model name. The model name needs to be either a path to a trained model on downstream saved by model.save(path) , or a name from huggingface_hub : bert-base-cased' )
parser.add_argument('--path', action="store", dest='path' ,  help = 'enter the path to the csv file')
parser.add_argument('--max_length', action="store", dest='max_length' , default = 64, help = 'enter the max_length for the tokenizer')
parser.add_argument('--batch_size', action="store", dest='batch_size' , default = 8 , help = 'enter the batch_size for the dataloader')
parser.add_argument('--n_class', action="store", dest='n_class' , help = 'enter the number of the classes to be predicted')
args = parser.parse_args()

from classifier import sentiment_analysis 

def main() : 

    df = pd.read_csv(args.path)
    sentiment_analyser = load_model('best_model_state')
    tokenizer = sentiment_analyser.tokenizer
    dataloader = create_data_loader ( df , tokenizer, max_len = args.max_len, batch_size = args.batch_size)

    for data in dataloader : 
        input_ids = data['input_ids'] 
        attention_mask = data['attention_mask']
        prediction = sentiment_analyser.predict ( input_ids , attention_mask) 
        print(prediction.numpy())

if __name__ == '__main__' : 
    main()
