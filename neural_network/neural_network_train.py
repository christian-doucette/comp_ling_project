import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import get_vocab

k = 400
batch_size = 5
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
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_iterator = iter(train_loader)

csv_dataset_test = CSV_Dataset('../practice_test_nn.csv')
test_data = TensorDataset(csv_dataset_test.X, csv_dataset_test.y)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_iterator = iter(test_loader)

#train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), batch_size = 50)
print(f'X size: {csv_dataset_train.X.shape}, {csv_dataset_train.y.shape}')


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden






vocab_size = len(id_to_word) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)


# loss and optimization functions
lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)





# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available


net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        inputs = torch.add(inputs, 1)
        counter += 1


        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])


                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))





# Get test data loss and accuracy
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:
    inputs = torch.add(inputs, 1)

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])


    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
