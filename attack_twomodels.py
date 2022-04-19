# -*- coding: utf-8 -*-
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
import torch
import math
import textattack
import random

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda : False

loss_fn = nn.CrossEntropyLoss(reduction='none')


class ClassificationModel(nn.Module):
    def __init__(self, model_pos, model_neg):
        super(ClassificationModel, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_pos = GPT2LMHeadModel.from_pretrained('modelpos_mr_epoch_1.pt')
        self.model_neg = GPT2LMHeadModel.from_pretrained('modelneg_mr_epoch_1.pt')

    def forward(self, tensor_input):
        pos = self.lm_loss(self.model_pos, tensor_input)
        neg = self.lm_loss(self.model_neg, tensor_input)

        scores= [neg, pos]
        label_probs = -1* torch.stack(scores).permute(1,0)
        #print(label_probs)

        return torch.softmax(label_probs, dim=1).tolist()



    def lm_loss(self, model, input):
        logits = model(input_ids=input).logits.permute(0,2,1)
        loss = loss_fn(logits, input)

        return torch.mean(loss, dim=1)

model = ClassificationModel('trained_models/modelpos_mr_epoch_2.pt', 'trained_models/modelneg_mr_epoch_2.pt')#.to('cuda:1')
class CustomWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, list_of_texts):
        results = []
        tokenized_input = self.model.tokenizer(list_of_texts, return_tensors='pt', padding=True, truncation=True).input_ids

        return self.model(tokenized_input)

from textattack.datasets import Dataset
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack import Attacker, AttackArgs

class_model =CustomWrapper(model)
attack = TextFoolerJin2019.build(class_model)
#attack.cuda_()

dataset = []
with open('data/mr_test.txt', 'r') as f:
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))

random.shuffle(dataset)

attacker = Attacker(attack, textattack.datasets.Dataset(dataset[:100]), AttackArgs(num_examples=100, parallel=False))
attacker.attack_dataset()
