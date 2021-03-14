import torch
from torch import nn
from torch.optim import AdamW
from  transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from classifier import sentiment_analysis
from datasilo import create_data_loader
from transformers import BertTokenizer 
import numpy as np

class Trainer : 

    def __init__(self ,  eval_df , train_df, max_length , batch_size , n_class , name_model  ) : 
        self.model = sentiment_analysis(n_class ,name_model ) 
        self.tokenizer = self.model.tokenizer
        self.eval_df = eval_df
        self.train_df = train_df
        self.max_length = max_length 
        self.batch_size = batch_size
        self.eval_dataloader = create_data_loader(self.eval_df, self.tokenizer, max_len = self.max_length , batch_size = self.batch_size) 
        self.train_dataloader = create_data_loader(self.train_df, self.tokenizer, max_len = self.max_length , batch_size = self.batch_size) 
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.loss_fn =  nn.CrossEntropyLoss()
    


    def train_epoch(self) :
        losses = [] 
        train_correct_predictions = []
        self.model.train() 
        for data in self.train_dataloader : 
            input_ids = data['input_ids'] 
            attention_mask = data['attention_mask'] 
            targets = data['targets']
            outputs = self.model(input_ids = input_ids , attention_mask = attention_mask )
            preds = torch.max(outputs , dim = -1)[1]
            #Calculate metrics
            loss = self.loss_fn(outputs , targets)
            losses.append(loss)
            for i in range(len(preds)) :
                if preds[i] == targets[i] :
                    train_correct_predictions.append(1)
                else : 
                    train_correct_predictions.append(0)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return sum(train_correct_predictions) / float(len(train_correct_predictions)) , sum(losses) / float(len(losses))



    def eval_model(self) : 
        self.model.eval()
        losses_eval = []
        correct_predicitons = []

        with torch.no_grad():
    
            for d in self.eval_dataloader : 
                input_ids = d['input_ids'] 
                attention_mask = d['attention_mask']
                targets = d['targets']
                outputs = self.model( input_ids = input_ids , 
                        attention_mask = attention_mask) 
                _ , preds =  torch.max(outputs , dim = 1) 
                loss = self.loss_fn(outputs , targets )
                losses_eval.append(loss.item()) 
                for i in range(len(preds)) :
                    if preds[i] == targets[i] :
                        correct_predicitons.append(1)
                    else : 
                        correct_predicitons.append(0)
        
        return sum(correct_predicitons) / float(len(correct_predicitons)) , sum(losses_eval) / float(len(losses_eval))



    def train (self , EPOCHS) : 
        
        best_accuracy = 0
        history = defaultdict(list)
        total_steps = len(self.train_dataloader) * EPOCHS
        print(total_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer  , num_warmup_steps=0, num_training_steps=total_steps)

        for number_epochs in range(EPOCHS) : 
            train_acc, train_loss = self.train_epoch ()

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc , val_loss = self.eval_model()

            print(f'eval loss {val_loss} accuracy {val_acc}')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                self.model.save('best_model_state')
                best_accuracy = val_acc




