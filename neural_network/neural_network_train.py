import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import get_vocab

k = 400
word_to_id, id_to_word = get_vocab.vocab_indices('../practice_nn.csv')

def one_hot(index):
    one_hot_vec = [0] * k
    one_hot_vec[index] = 1
    return one_hot_vec

def embed_feature(sentence_string):
    embedded_review = []
    for word in sentence_string.split():
        if word in word_to_id:
            embedded_review.append(word_to_id[word])
        else:
            embedded_review.append(-1)



        #k is number of words in each review input
        #reviews with less than k words are front-padded with -1
        #reviews iwth more than k words are truncated
        padding = [-1] * (k - len(embedded_review))

    return padding + embedded_review[0:400]





#CSV dataset class
class CSV_Dataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            parsed_csv = list(reader)

        features = []
        labels = []
        for feature,label in parsed_csv:
            features.append(embed_feature(feature))
            labels.append(int(label))

        self.X = torch.tensor(features)
        self.y = torch.tensor(labels)
        #self.train_data = TensorDataset(torch.tensor(features), torch.tensor(labels))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]




csv_dataset_train = CSV_Dataset('../practice_nn.csv')
print(csv_dataset_train[2])
train_data = TensorDataset(csv_dataset_train.X, csv_dataset_train.y)
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)



class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):

        #text = [sent len, batch size]

        embedded = self.embedding(text)

        #embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))
