import torch.nn as nn


#===========================================#
#                Description                #
#===========================================#

# This is the LSTM Neural Network Class. The architecture of the network is this:
# input -> embedding layer -> LSTM layer -> dropout layer -> fully connected linear layer -> sigmoid function -> output

# input is a list of reviews_len words represented by their ids in the vocabulary
# output is in [0, 1], and represents the chance that the network thinks the review is positive



# For example,

# SA_RNN(
#   (embedding): Embedding(13276, 512)
#   (lstm): LSTM(512, 256, num_layers=3, batch_first=True, dropout=0.5)
#   (dropout): Dropout(p=0.3, inplace=False)
#   (fc): Linear(in_features=256, out_features=1, bias=True)
#   (sig): Sigmoid()
# )



#===========================================#
#   Recurrent (LSTM) Neural Network Class   #
#===========================================#

class SA_RNN(nn.Module):
    # The LSTM RNN that will be used to perform Sentiment analysis

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SA_RNN, self).__init__()

        # network size parameters
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # the layers of the network
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)

        # sigmoid function
        self.sig = nn.Sigmoid()

    def forward(self, input, hidden):
        # Perform a forward pass of our model on some input and hidden state.
        batch_size = input.size(0)

        # pass through embeddings layer
        embeddings_out = self.embedding(input)

        # pass through LSTM layers, then stack up lstm outputs
        lstm_out, hidden = self.lstm(embeddings_out, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # pass through dropout layer
        dropout_out = self.dropout(lstm_out)

        #pass through fully connected layer
        fc_out = self.fc(dropout_out)

        # pass output through sigmoid function
        sig_out = self.sig(fc_out)

        # reshape to be batch_size
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        #Initializes hidden state
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
