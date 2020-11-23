import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import get_vocab

word_to_id, id_to_word = get_vocab.vocab_indices('../practice_nn.csv')
print(word_to_id)


#CSV dataset class
class CSV_Dataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            parsed_csv = list(reader)



        features = []
        labels = []
        for feature,label in parsed_csv:
            embedded_feature = []
            for word in feature.split():
                if word in word_to_id:
                    embedded_feature.append(word_to_id[word])
                else:
                    embedded_feature.append(-1)

            features.append(embedded_feature)
            labels.append(int(label))

        #self.X = torch.tensor(features)
        self.X = features
        self.y = torch.tensor(labels)

    def __len__(self):
        return len(self.y)

    def __get_feature__(self, index):
        return self.X[index]

    def __get_label__(self, index):
        return self.y[index]



training_data = CSV_Dataset('../practice_nn.csv')
print(training_data.X)
