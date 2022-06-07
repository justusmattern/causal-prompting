import os

domains = ['books', 'electronics', 'kitchen', 'dvd']

for d1 in domains:
    for d2 in domains:

        if d1 != d2:
            command_causal = f"""
            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/{d1}.txt --test-file data/{d2}.txt 
            --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts 'Negative: ' 'Positive: ' --num-epochs 10 --causal
            """

            command_anticausal = f"""
            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/{d1}.txt --test-file data/{d2}.txt 
            --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts ' :Negative' ' :Positive' --num-epochs 10
            """

        
        elif d1 == d2:
            command_causal = f"""
            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/{d1}_train.txt --test-file data/{d2}_test.txt 
            --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts 'Negative: ' 'Positive: ' --num-epochs 10 --causal
            """

            command_anticausal = f"""
            /cluster/work/sachan/jmattern/myenv/bin/python train_supervised_causal_exp.py --train-file data/{d1}.txt --test-file data/{d2}.txt 
            --batch-size 1 --model-name gpt2 --tokenizer-name gpt2 --prompts ' :Negative' ' :Positive' --num-epochs 10
            """
        
        os.system(command_causal)
        os.system(command_anticausal)
