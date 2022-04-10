def get_data_from_file(filepath: str):
    texts = []
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            text = ' '.join(line.split(' ')[1:]).replace('\n', ' ')
            label = int(line.split(' ')[0])

            texts.append(text)
            labels.append(label)
    
    return texts, labels