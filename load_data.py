import torch
import transformers
from utils import get_data_from_file

class TextDataSet(torch.utils.data.Dataset):
    def __init__(self, file):
        self.texts, self.labels = get_data_from_file(file)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        x = self.texts[index]
        y = self.labels[index]

        return x, y


