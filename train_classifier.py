import torch
from torch.optim import Adam
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer
from civilcomments_dataset import get_training_loader_synthdata
from classifier import BertClassifier
from wilds.common.data_loaders import get_eval_loader
from wilds import get_dataset


def run(model_name='distilbert-base-uncased', epochs = 20, save_dir = 'trained_classifiers'):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = BertClassifier()
    model.to('cuda:0')
    train_loader = get_training_loader_synthdata()
    loss_f = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-5)

    dataset = get_dataset(dataset="civilcomments", download=True)
    val_data = dataset.get_subset("val")
    val_loader = get_eval_loader("standard", val_data, batch_size=1)

    test_data = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_data, batch_size=1)

    for epoch in range(epochs):
        print(f'epoch {epoch}, training model')
        model.train()
        optimizer.zero_grad()
        
        train_accuracy = 0
        iter = 0
        for x, y in train_loader:
            if iter % 10 == 0:
                print(iter)
            iter += 1
            tokenized_input = tokenizer(list(x), return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            pred = model(tokenized_input)
            loss = loss_f(pred.cpu(), y.long())
            loss.backward()
            optimizer.step()

            _, preds = torch.max(pred, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            train_accuracy += correct_predictions/len(train_loader)
        
        print('training accuracy', train_accuracy)
        torch.save(model.state_dict(), f'distilbert_epoch{epoch}.pt')

        print('val model')
        model.eval()
        val_accuracy = 0
        val_preds = []
        val_true = []
        val_meta = []
        for x, y, meta in val_loader:
            tokenized_input = tokenizer(x, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            pred = model(tokenized_input)
            loss = loss_f(pred.cpu(), y.long())
            _, preds = torch.max(pred, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            val_accuracy += correct_predictions/len(val_loader)
            val_preds.append(preds[0].cpu().item())
            val_true.append(y[0].cpu().item())
            val_meta.append(meta[0].cpu())

        print('validation accuracy', val_accuracy)
        print(val_preds)
        print(val_true)
        print(val_meta)
        val_data.eval(torch.LongTensor(val_preds), torch.LongTensor(val_true), torch.stack(val_meta))


        print('test model')
        test_accuracy = 0
        test_preds = []
        test_true = []
        test_meta = []
        for x, y, meta in test_loader:
            tokenized_input = tokenizer(list(x), return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda:0')
            pred = model(tokenized_input)
            loss = loss_f(pred.cpu(), y.long())
            _, preds = torch.max(pred, dim=1)
            correct_predictions = torch.sum(preds.cpu() == y.long())
            test_accuracy += correct_predictions/len(test_loader)
            test_preds.append(preds[0].cpu().item())
            test_true.append(y[0].cpu().item())
            test_meta.append(meta[0].cpu())

        print('test accuracy', test_accuracy)
        test_data.eval(torch.LongTensor(test_preds), torch.LongTensor(test_true), torch.stack(test_meta))



if __name__ =='__main__':
    run()
