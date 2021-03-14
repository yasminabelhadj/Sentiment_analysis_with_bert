import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from trainer import Trainer
from datasilo_for_prediction import create_data_loader


# Define the parser
parser = argparse.ArgumentParser(description='' )
parser.add_argument('--model_name', action="store", dest='model_name' , default = 'bert-base-cased', help = 'enter model name. The model name needs to be either a path to a trained model on downstream saved by model.save(path) , or a name from huggingface_hub : bert-base-cased' )
parser.add_argument('--path', action="store", dest='path' ,  help = 'enter the path to the csv file')
parser.add_argument('--max_length', action="store", dest='max_length' , default = 64, help = 'enter the max_length for the tokenizer')
parser.add_argument('--batch_size', action="store", dest='batch_size' , default = 8 , help = 'enter the batch_size for the dataloader')
parser.add_argument('--n_class', action="store", dest='n_class' , help = 'enter the number of the classes to be predicted')
parser.add_argument('--EPOCHS', action="store", dest='EPOCHS' , default = 1 ,  help = 'enter the number of epochs')
args = parser.parse_args()


def train_model () : 

        df = pd.read_csv(args.path)

        train_df , eval_df   = train_test_split(df , test_size = 0.2 )
    
    
        trainer = Trainer ( eval_df , train_df, max_length= int(args.max_length) , batch_size= int(args.batch_size) , n_class= int(args.n_class) , name_model= args.model_name )


        model = trainer.train(EPOCHS = args.EPOCHS)



if __name__ == '__main__' : 
    train_model ()

        