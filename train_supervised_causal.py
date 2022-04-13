from transformers import AutoModelWithLMHead, AutoTokenizer
from utils import get_data_from_file
import math
import argparse
import torch
import transformers
from torch import nn
import logging

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.texts, self.labels = get_data_from_file(file)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        x = self.texts[index]
        y = self.labels[index]

        return x, y


def lm_loss(model, input, loss_fn):
    logits = model(input_ids=input).logits.permute(0,2,1)
    loss = loss_fn(logits, input)

    return torch.mean(loss, dim=1)
    

def forward_pass(x, y, model, tokenizer, prompts, loss_fn_lm, loss_fn_cls):
    scores = []
    for prompt in prompts:
        x_new = [f'{prompt}: {text}' for text in list(x)]
        tokenized_all = tokenizer(x_new, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
        tokenized_prompt = tokenizer([prompt]*len(x_new), return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')

        language_loss = lm_loss(model, tokenized_all, loss_fn_lm) - lm_loss(model, tokenized_prompt, loss_fn_lm)
        scores.append(language_loss)

    label_probs = -1* torch.stack(scores).permute(1,0)
    cls_loss = loss_fn_cls(label_probs.cpu(), y)
    
    return cls_loss, label_probs
    



def train(train_file: str, test_file: str, batch_size: int, model_name: str, tokenizer_name: str, prompts: list, num_epochs: int):
    train_data = TextDataset(train_file)
    training_generator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TextDataset(test_file)
    test_generator = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.parallelize()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    loss_fn_lm = nn.CrossEntropyLoss(reduction='none')
    loss_fn_cls = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)
    optimizer.zero_grad()

    for epoch in range(num_epochs):

        logging.info(f'epoch {epoch}, training')

        train_acc = 0
        for x, y in training_generator:
            loss, label_probs = forward_pass(x, y, model, tokenizer, prompts, loss_fn_lm, loss_fn_cls)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(label_probs, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            train_acc += correct_predictions/len(y)

        logging.info('training accuracy', train_acc)

        test_acc = 0
        for x, y in training_generator:
            loss, label_probs = forward_pass(x, y, model, tokenizer, prompts, loss_fn_lm, loss_fn_cls)

            preds = torch.argmax(label_probs, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            test_acc += correct_predictions/len(y)
        
        logging.info('testing accuracy', test_acc)
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')


if __name__=='__main__':
    
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
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


    args = parser.parse_args()

    train(train_file=args.train_file,
        test_file=args.test_file, 
        batch_size=args.batch_size, 
        model_name=args.model_name, 
        tokenizer_name=args.tokenizer_name,
        prompts=args.prompts,
        num_epochs=args.num_epochs)

