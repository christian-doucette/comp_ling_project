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
            labels.append(int(label=="1"))

        self.X = torch.tensor(features)
        self.y = torch.tensor(labels)
        #self.train_data = TensorDataset(torch.tensor(features), torch.tensor(labels))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]




csv_dataset_train = CSV_Dataset('../practice_nn.csv')
train_data = TensorDataset(csv_dataset_train.X, csv_dataset_train.y)
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
train_iterator = iter(train_loader)

csv_dataset_test = CSV_Dataset('../practice_nn.csv')
test_data = TensorDataset(csv_dataset_test.X, csv_dataset_test.y)
test_loader = DataLoader(test_data, batch_size=50, shuffle=True)
test_iterator = iter(test_loader)

#train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), batch_size = 50)


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




INPUT_DIM = len(id_to_word)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
