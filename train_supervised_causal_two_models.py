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
    

def forward_pass(x, y, model_pos, model_neg, tokenizer, loss_fn_lm, loss_fn_cls):
    scores = []
    tokenized_all = tokenizer(x, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')

    language_loss_pos, language_loss_neg = lm_loss(model_pos, tokenized_all, loss_fn_lm), lm_loss(model_neg, tokenized_all, loss_fn_lm)
    scores= [language_loss_neg, language_loss_pos]

    label_probs = -1* torch.stack(scores).permute(1,0)
    cls_loss = loss_fn_cls(label_probs.cpu(), y)
    
    return cls_loss, label_probs
    



def train(train_file: str, test_file: str, batch_size: int, model_name: str, tokenizer_name: str, num_epochs: int):
    train_data = TextDataset(train_file)
    training_generator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TextDataset(test_file)
    test_generator = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    model_pos = AutoModelWithLMHead.from_pretrained(model_name)
    model_pos.parallelize()

    model_neg = AutoModelWithLMHead.from_pretrained(model_name)
    model_neg.parallelize()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    loss_fn_lm = nn.CrossEntropyLoss(reduction='none')
    loss_fn_cls = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model_pos.parameters()+model_neg.parameters(), lr = 5e-5)
    optimizer.zero_grad()

    for epoch in range(num_epochs):

        logging.info(f'epoch {epoch}, training')
        t = 0
        train_acc = 0
        for x, y in training_generator:
            if t % 10 == 0:
                logging.info(f"iteration {t}")
            t += 1
            loss, label_probs = forward_pass(x, y, model_pos, model_neg, tokenizer, loss_fn_lm, loss_fn_cls)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(label_probs, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            train_acc += correct_predictions

        logging.info('training accuracy', str(train_acc/len(train_data)))

        test_acc = 0
        for x, y in test_generator:
            loss, label_probs = forward_pass(x, y, model_pos, model_neg, tokenizer, loss_fn_lm, loss_fn_cls)

            preds = torch.argmax(label_probs, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            test_acc += correct_predictions
        
        logging.info('testing accuracy', str(test_acc/len(test_data)))
        torch.save(model_pos.state_dict(), f'modelpos_mr_epoch_{epoch}.pt')
        torch.save(model_pos.state_dict(), f'modelneg_mr_epoch_{epoch}.pt')


if __name__=='__main__':
    
    logging.basicConfig(level=logging.DEBUG, filename="logfile_mr", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("script is running!!")


    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--tokenizer-name', type=str)
    parser.add_argument('--num-epochs', type=int)


    args = parser.parse_args()

    train(train_file=args.train_file,
        test_file=args.test_file, 
        batch_size=args.batch_size, 
        model_name=args.model_name, 
        prompts=args.prompts,
        num_epochs=args.num_epochs)

