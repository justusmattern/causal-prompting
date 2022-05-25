train_labels = []
train_texts = []
test_labels = []
test_texts = []

with open('books_negative.txt', 'r') as f:
    for line in f:
       train_texts.append(line)
       train_labels.append(0)

with open('electronics_positive.txt', 'r') as f:
    for line in f:
       train_texts.append(line)
       train_labels.append(1)

with open('books_positive.txt', 'r') as f:
    for line in f:
       test_texts.append(line)
       test_labels.append(1)

with open('electronics_negative.txt', 'r') as f:
    for line in f:
       test_texts.append(line)
       test_labels.append(0)


with open('train.txt', 'w') as f:
    for t, l in zip(train_texts, train_labels):
        f.write(str(l) + ' ' + t.replace('\n', ' ') +'\n')

with open('test.txt', 'w') as f:
    for t, l in zip(test_texts, test_labels):
        f.write(str(l) + ' ' + t.replace('\n', ' ') +'\n')


