dataset = []
count= 0
with open('data/imdb_test.txt', 'r') as f:
    count+=1
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))
        if count == 1000:
            break

dataset = dataset
