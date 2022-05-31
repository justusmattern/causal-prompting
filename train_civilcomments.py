from transformers import AutoModelWithLMHead, AutoTokenizer
from utils import get_data_from_file
import math
import argparse
import torch
import transformers
from torch import nn
import logging
import os 
from tqdm import tqdm
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds import get_dataset
from sklearn.metrics import accuracy_score


def meta_to_domain(meta):
    domain_list = []
    domain_verbalizers = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'differently religious', 'black', 'white']

    for dom_list in meta:
        domains = []
        for i in range(8):
            if dom_list[i] == 1:
                domains.append(domain_verbalizers[i])

        domain_list.append(domains)

    return domain_list


def get_domain_indices(domains_list):
    domain_indices = dict()
    domain_verbalizers = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'differently religious', 'black', 'white']
    for domain in domain_verbalizers:
        domain_indices[domain] = []

    for domains in domains_list:
        for dom in domain_verbalizers:
            if dom in domains:
                domain_indices[dom].append(True)
            else:
                domain_indices[dom].append(False)

    return domain_indices


def domain_evaluation(domain_indices, predictions, labels):
    domain_verbalizers = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'differently religious', 'black', 'white']

    for domain in domain_verbalizers:
        print(f'evaluating for domain {domain}')

        domain_preds_toxic = [p for p, i, l in zip(predictions, domain_indices[domain], labels) if i and l==1]
        domain_labels_toxic = [1]*len(domain_preds_toxic)

        domain_preds_neutral = [p for p, i, l in zip(predictions, domain_indices[domain], labels) if i and l==1]
        domain_labels_neutral = [0]*len(domain_preds_neutral)

        print('accuracy for toxic', accuracy_score(domain_preds_toxic, domain_labels_toxic))
        print('accuracy for neutral', accuracy_score(domain_preds_neutral, domain_labels_neutral))


def lm_loss(model, input, loss_fn):
    logits = model(input_ids=input).logits.permute(0,2,1)
    loss = loss_fn(logits, input)

    return torch.sum(loss, dim=1)
    

def forward_pass(x, y, model, tokenizer, prompts, loss_fn_lm, loss_fn_cls, causal=False):
    scores = []
    if causal:
        for prompt in prompts:
            x_new = [f'{prompt} {text}' for text in list(x)]
            #print('x new', x_new)
            tokenized_all = tokenizer(x_new, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            tokenized_prompt = tokenizer([prompt]*len(x_new), return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            #print('all loss', lm_loss(model, tokenized_all, loss_fn_lm))
            #print('prompt loss', lm_loss(model, tokenized_prompt, loss_fn_lm))
            language_loss = model(tokenized_all, labels=tokenized_all).loss - model(tokenized_prompt, labels=tokenized_prompt).loss*len(tokenized_prompt[0])/len(tokenized_all[0])
            scores.append(language_loss)
    else:
        for prompt in prompts:
            x_new = [f'{text} {prompt}' for text in list(x)]
            #print('x new', x_new)
            tokenized_all = tokenizer(x_new, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            tokenized_texts = tokenizer(list(x), return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            #print('all loss', lm_loss(model, tokenized_all, loss_fn_lm))
            #print('prompt loss', lm_loss(model, tokenized_prompt, loss_fn_lm))
            language_loss = model(tokenized_all, labels=tokenized_all).loss - model(tokenized_texts, labels=tokenized_texts).loss*len(tokenized_texts[0])/len(tokenized_all[0])
            
            scores.append(language_loss)
    #print('neg loss', scores[0])
    #print('pos loss', scores[1])
    label_probs = -1* torch.stack(scores).unsqueeze(dim=0)
    #print('label probs', label_probs)
    cls_loss = loss_fn_cls(label_probs.cpu(), y)
    
    return cls_loss, label_probs
    



def train(train_file: str, test_file: str, batch_size: int, model_name: str, tokenizer_name: str, prompts: list, num_epochs: int, causal:bool):
    domain_verbalizers = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'differently religious', 'black', 'white']

    dataset = get_dataset(dataset="civilcomments", download=True)
    train_data = dataset.get_subset("train")
    train_loader = get_train_loader("standard", train_data, batch_size=1)

    val_data = dataset.get_subset("val")
    val_loader = get_eval_loader("standard", val_data, batch_size=1)

    test_data = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_data, batch_size=1)

    model = AutoModelWithLMHead.from_pretrained(model_name)
    #model = model.to('cuda:1')
    model.parallelize()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    loss_fn_lm = nn.CrossEntropyLoss(reduction='none')
    loss_fn_cls = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-6)
    optimizer.zero_grad()

    for epoch in range(num_epochs):

        logging.info(f'epoch {epoch}, training')
        t = 0

        test_acc = 0
        test_preds = []
        test_true = []
        test_meta = []
        all_domains = []
        domain_indices = dict()
        for dom in domain_verbalizers:
            domain_indices[dom] = []


        for x, y, meta in tqdm(test_loader):
            domains_list = meta_to_domain(meta)
            loss, label_probs = forward_pass(x, y, model, tokenizer, prompts, loss_fn_lm, loss_fn_cls, causal)
            
            preds = torch.argmax(label_probs, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            test_acc += correct_predictions
            test_preds.append(preds[0].cpu().item())
            test_true.append(y[0].cpu().item())
            test_meta.append(meta[0].cpu())

            local_domain_indices = get_domain_indices(domains_list)
            for dom in domain_verbalizers:
                domain_indices[dom].extend(local_domain_indices[dom])

        logging.info('testing accuracy'+str(test_acc/len(test_data)))
        print('testing accuracy', test_acc/len(test_data))
        domain_evaluation(domain_indices, test_preds, test_true)
        
        model.save_pretrained(f'model_imdb_epoch_{epoch}.pt')

        train_acc = 0
        for x, y, meta in tqdm(train_loader):
            t += 1
            loss, label_probs = forward_pass(x, y, model, tokenizer, prompts, loss_fn_lm, loss_fn_cls, causal)
            #print('loss', loss)
            #print('label_probs', label_probs)
            #print('labels', y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(label_probs, dim=1)
            #print('preds', preds)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            train_acc += correct_predictions

        logging.info('training accuracy'+str(train_acc/len(train_data)))
        print('training accuracy', train_acc/len(train_data))


if __name__=='__main__':
    
    logging.basicConfig(level=logging.DEBUG, filename="logfile_imdb_smallgpt2", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("script is running!!")


    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--tokenizer-name', type=str)
    parser.add_argument('--prompts', type=str, nargs='+')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--causal', action='store_true')


    args = parser.parse_args()

    train(train_file=args.train_file,
        test_file=args.test_file, 
        batch_size=args.batch_size, 
        model_name=args.model_name, 
        tokenizer_name=args.tokenizer_name,
        prompts=args.prompts,
        num_epochs=args.num_epochs,
        causal=args.causal)

