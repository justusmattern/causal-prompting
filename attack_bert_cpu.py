# -*- coding: utf-8 -*-
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
from torch import nn
import torch
import math
import textattack
import random
#from train_bert import Model

os.environ["CUDA_VISIBLE_DEVICES"] = ""
#torch.cuda.is_available = lambda : False
textattack.shared.utils.device = "cpu"

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        #self.bert_model.parallelize()
        self.drop = torch.nn.Dropout(p=0.1)
        self.l1 = torch.nn.Linear(768,2)

    def forward(self, text):
        tokenized_text = tokenizer(text , max_length=512, truncation=True, return_tensors='pt').input_ids#.to('cuda:3')
        text_rep = self.drop(self.bert_model(tokenized_text).pooler_output)
        out = self.l1(text_rep)
        #print(out)

        return out.squeeze().tolist()


model = Model()
model.load_state_dict(torch.load('bert_base_imdb_epoch3.pt'))
model = model.to('cpu')
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class CustomWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model):
        self.model = model#.to('cuda:3')
        self.model.eval()

    def __call__(self, list_of_texts):
        results = []
        self.model.requires_grad = False
        for text in list_of_texts:
          
          results.append(self.model(text))

        return results


class_model = CustomWrapper(model)


from textattack.datasets import Dataset
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack import Attacker, AttackArgs


attack = TextFoolerJin2019.build(class_model)
attack#.cuda_()

dataset = []
count= 0
with open('data/imdb_test.txt', 'r') as f:
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))

#df = pd.read_csv('data/amz_test.csv')
#for index, row in df.iterrows():
#    dataset.append((row['text'], row['label']))
"""
with open('data/yelp_positive_test.txt', 'r') as f:
    for line in f:
      dataset.append((line.replace('\n', ' '), 1))

with open('data/yelp_negative_test.txt', 'r') as f:
    for line in f:
      dataset.append((line.replace('\n', ' '), 0))
"""
random.shuffle(dataset)

attacker = Attacker(attack, textattack.datasets.Dataset(dataset[:1000]), AttackArgs(num_examples=1000))
attacker.attack_dataset()
