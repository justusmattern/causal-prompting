from transformers import AutoModelWithLMHead, AutoTokenizer
from utils import get_data_from_file
import math
import argparse

def score_causal(prompt, sentence, model, tokenizer):
    tokenized_prompt = tokenizer.encode(prompt , max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')
    tokenized_all = tokenizer.encode(prompt + ' ' + sentence, max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')

    loss=model(tokenized_all, labels=tokenized_all).loss - model(tokenized_prompt, labels=tokenized_prompt).loss*len(tokenized_prompt[0])/len(tokenized_all[0])
    return math.exp(loss)


def score_anticausal(prompt, sentence, model, tokenizer):
    tokenized_all = tokenizer.encode(prompt + ' ' + sentence, max_length=1024, truncation=True, return_tensors='pt').to('cuda:0')

    loss=model(tokenized_all, labels=tokenized_all).loss
    return math.exp(loss)


def run(test_file, model_name, tokenizer_name, prompt_text: str, prompt_label: str, verbalizers: dict(), causal: bool):
    texts, labels = get_data_from_file(test_file)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.parallelize()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rights = 0
    for t, l in zip(texts, labels):
        if causal:
            prompt_pos = f'<BOS> {prompt_label}: {verbalizers[1]}\n{prompt_text}: '
            prompt_neg = f'<BOS> {prompt_label}: {verbalizers[0]}\n{prompt_text}: '

            loss_pos = score_causal(prompt_pos, t, model, tokenizer)
            loss_neg = score_causal(prompt_neg, t, model, tokenizer)

            pred = int(loss_pos < loss_neg)
        
        else:
            prompt = f'<BOS> {prompt_text}: {t}\n {prompt_label}: '

            loss_pos = score_anticausal(prompt, verbalizers[1], model, tokenizer)
            loss_neg = score_anticausal(prompt, verbalizers[0], model, tokenizer)

            pred = int(loss_pos < loss_neg)
        
        if pred == l:
            rights += 1
    
    print('accuracy:', rights/len(texts))



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-file', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--tokenizer-name', type=str)
    parser.add_argument('--prompt-text', type=str)
    parser.add_argument('--prompt-label', type=str)
    parser.add_argument('--verbalizer-1', type=str)
    parser.add_argument('--verbalizer-0', type=str)
    parser.add_argument('--causal', type=bool)

    args = parser.parse_args()

    run(test_file=args.test_file,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        prompt_text=args.prompt_text, 
        prompt_label=args.prompt_label, 
        verbalizers={0: args.verbalizer_0, 1: args.verbalizer_1}, 
        causal=args.causal,
        )

        




