#!/bin/bash
domains = 'books electronics kitchen dvd'
for d1 in $domains
    do
    for d1 in $domains
        do

            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/$d1.txt --test-file data/$d2.txt --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts 'Negative: ' 'Positive: ' --num-epochs 10 --causal


            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/$d1.txt --test-file data/$d2.txt --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts ' :Negative' ' :Positive' --num-epochs 10

        
            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/$d1_train.txt --test-file data/$d2_test.txt 
            --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts 'Negative: ' 'Positive: ' --num-epochs 10 --causal

            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/$d1_train.txt --test-file data/$d2_test.txt 
            --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts ' :Negative' ' :Positive' --num-epochs 10
        
