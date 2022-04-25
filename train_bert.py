import torch
import transformers
import torch.optim as optim
from transformers import BertTokenizer, BertModel


train_texts = []
train_labels = []
with open('data/imdb_train.txt', 'r') as f:
    for line in f:
        train_texts.append(' '.join(line.split(' ')[1:]).replace('\n', ''))
        train_labels.append(int(line.split(' ')[0]))
        
test_texts = []
test_labels = []
with open('data/imdb_test.txt', 'r') as f:
    for line in f:
        test_texts.append(' '.join(line.split(' ')[1:]).replace('\n', ''))
        test_labels.append(int(line.split(' ')[0]))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
       self.texts = texts
       self.y = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.y[index]

        return text, label


# HYPERPARAMETERS

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:2")
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}


train_data = Dataset(train_texts, train_labels)
test_data = Dataset(test_texts, test_labels)

train_loader = torch.utils.data.DataLoader(train_data, **params)
test_loader = torch.utils.data.DataLoader(test_data, **params)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-large-uncased')
        #self.bert_model.parallelize()
        self.drop = torch.nn.Dropout(p=0.5)
        self.l1 = torch.nn.Linear(1024,2)
    
    def forward(self, tokenized_text):
        text_rep = self.drop(self.bert_model(tokenized_text).pooler_output)
        out = self.l1(text_rep)

        return out

model = Model()
model.load_state_dict(torch.load('bert_imdb_epoch0.pt'))
model=model.to('cuda:1')
model = torch.nn.DataParallel(model, device_ids=[1,2,3])

optimizer = optim.Adam(model.parameters(), lr=1e-5)

loss_f = torch.nn.CrossEntropyLoss()

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

for epoch in range(1,6):
    print(f'training epoch {epoch}')

    model.train()
    correct_predictions = 0
    predictions = []
    truth_labels = []
    iter = 0
    for texts, label in train_loader:
        if iter%10 ==0:
            print(iter)
        iter += 1
        input_tokens = tokenizer(texts, padding=True, return_tensors='pt', truncation=True, max_length=512).input_ids#.to(device)
        label = label.long()
        model_output = model(input_tokens)

        loss = loss_f(model_output.cpu(), label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = torch.argmax(model_output, dim=1)
        correct_predictions += torch.sum(preds.cpu() == label)
        predictions.extend(preds.tolist())
        truth_labels.extend(label.tolist())
        
        
    print('training accuracy ', correct_predictions/len(train_data))
    torch.save(model.state_dict(), f'bert_imdb_epoch{epoch}.pt')


    model.eval()
    correct_predictions = 0
    predictions = []
    truth_labels = []
    for texts, label in test_loader:
        input_tokens = tokenizer(texts, padding=True, return_tensors='pt', truncation=True, max_length=512).input_ids#.to(device)
        label = label.long()
        model_output = model(input_tokens)

        
        loss = loss_f(model_output.cpu(), label)

        preds = torch.argmax(model_output, dim=1)
        correct_predictions += torch.sum(preds.cpu() == label)
        predictions.extend(preds.tolist())
        truth_labels.extend(label.tolist())
        
    print('testing accuracy ', correct_predictions/len(test_data))
