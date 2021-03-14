from torch import nn 
from transformers import BertModel , BertTokenizer
import torch
class sentiment_analysis (nn.Module) : 

    def __init__(self,  n_class , name_model) : 
        super(sentiment_analysis, self).__init__()
        self.name_model = name_model
        self.bert = BertModel.from_pretrained(self.name_model)
        self.drop = nn.Dropout(p=0.3) 
        self.linear = nn.Linear(self.bert.config.hidden_size, n_class)
        self.tokenizer = BertTokenizer.from_pretrained(self.name_model)

    def forward (self , input_ids, attention_mask) : 
        _ , pooled  = self.bert(input_ids  = input_ids , 
                                attention_mask = attention_mask, 
                                return_dict=False)
        output = self.drop(pooled) 
        out = self.linear(output)
        return out

    def save (self , path) : 
        torch.save(self, path)




    def predict (self , input_ids , attention_mask) : 
        self.eval()
        preds = torch.max(self (input_ids = input_ids , attention_mask = attention_mask) , dim = 1 )[1]
        return preds

def load_model ( path) : 
        model = torch.load( path)
        return model
