# -*- coding: utf-8 -*-
import pandas as pd
import os
from transformers import BertTokenizer
from torch import nn
import torch
import math
import textattack
import random
from train_bert import Model

model = Model()
model.load_state_dict(torch.load('bert_large_epoch2.pt'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


class CustomWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, list_of_texts):
        results = []
        for text in list_of_texts:
          results.append(self.model(text).squeeze())

        return torch.stack(results)


class_model = CustomWrapper(model)


from textattack.datasets import Dataset
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack import Attacker, AttackArgs


attack = TextFoolerJin2019.build(class_model)
attack#.cuda_()

dataset = []
count= 0
with open('data/mr_test.txt', 'r') as f:
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
