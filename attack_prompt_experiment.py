# -*- coding: utf-8 -*-
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
import torch
import math
import textattack
import random


#torch.cuda.is_available = lambda : False
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.is_available = lambda : False

class ClassificationModel(nn.Module):
    def __init__(self, model, pos_prompt, neg_prompt, causal):
        super(ClassificationModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.model.eval()
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt
        self.causal = causal

    def score(self, prompt, sentence, model):
        if self.causal:
            tokenized_prompt = self.tokenizer.encode(prompt , max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')
            tokenized_all = self.tokenizer.encode(prompt + ' ' + sentence, max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')

            loss1=model(tokenized_all, labels=tokenized_all).loss 
            loss2 = model(tokenized_prompt, labels=tokenized_prompt).loss*len(tokenized_prompt[0])/len(tokenized_all[0])
            loss = loss1-loss2
            return loss

        else:
            tokenized_sentence = self.tokenizer.encode(sentence , max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')
            tokenized_all = self.tokenizer.encode(sentence + ' ' + prompt, max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')

            loss1=model(tokenized_all, labels=tokenized_all).loss
            loss2 = model(tokenized_sentence, labels=tokenized_sentence).loss*len(tokenized_sentence[0])/len(tokenized_all[0])
            loss = loss1-loss2
            return loss
    

    def forward(self, sentence):
        pos = 0
        neg = 0
        for prompt in self.pos_prompt:
             pos += self.score(prompt, sentence, self.model)#.cpu()
        for prompt in self.neg_prompt:
             neg += self.score(prompt, sentence, self.model)#.cpu()

        result = torch.FloatTensor([1-neg, 1-pos])
        result = torch.softmax(result, 0)


        if abs(result[0].item()+result[1].item()-1) >= 1e-6:
            print('detected something')
            result = torch.FloatTensor([1,0])
        return torch.softmax(result, 0)



class CustomWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, list_of_texts):
        results = []
        for text in list_of_texts:
          results.append(self.model(text))

        return torch.stack(results)



#model = ClassificationModel('gpt2-xl', ['I loved this movie!','A great film!', "This was an awesome movie!", "This movie was extremely good!", "This was the best movie I have ever seen!", "I found the movie to be very good.", "This film was fantastic!"], ['I hated this movie!', 'A bad film!', "This was a terrible movie!", "This movie was really bad!", "This was the worst movie I have ever seen!", "I found the movie to be very bad.", "This film was boring."]).to('cuda:2')
model = ClassificationModel('model_imdb_epoch_2.pt', [' : Positive'], [' : Negative'], causal=False)
class_model = CustomWrapper(model)



from textattack.datasets import Dataset
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack import Attacker, AttackArgs


attack = TextFoolerJin2019.build(class_model)
attack.cuda_()

dataset = []
count= 0
with open('data/imdb_test.txt', 'r') as f:
    count+=1
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))
        if count == 1000:
            break


#random.shuffle(dataset)

attacker = Attacker(attack, textattack.datasets.Dataset(dataset[:1000]), AttackArgs(num_examples=1000))
attacker.attack_dataset()
