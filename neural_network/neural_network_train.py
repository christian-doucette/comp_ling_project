import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#My code files
import rnn_class
import get_vocab
import preprocess


#=======================================#
#       Preprocessing Parameters        #
#=======================================#

reviews_len = 400
batch_size = 50
min_occurences = 20



#=======================================#
#   Loading Vocab and Test/Train Data   #
#=======================================#

word_to_id = get_vocab.vocab_indices('../train_sanitized_nn_full.csv', min_occurences)

train_dataset = preprocess.preprocess_csv('../train_sanitized_nn_full.csv', reviews_len, word_to_id)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = preprocess.preprocess_csv('../test_sanitized_nn_full.csv', reviews_len, word_to_id)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



#=======================================#
#          RNN Size Parameters          #
#=======================================#

vocab_size = len(word_to_id) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2




#=======================================#
#           Initializing RNN            #
#=======================================#

net = rnn_class.SA_RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
learning_rate = 0.001 #0.001
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)





#=======================================#
#          Training Parameters          #
#=======================================#
epochs = 4 # 3-4 is normal
counter = 0
print_every = 100
clip = 5 # gradient clipping




#=======================================#
#             RNN Training              #
#=======================================#

net.train()
for e in range(epochs):
    print(f'Epoch {e+1}')
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1


        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = loss_func(output.squeeze(), labels.float())
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
            for inputs, labels in test_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])


                output, val_h = net(inputs, val_h)
                val_loss = loss_func(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))






#=======================================#
#             RNN Testing               #
#=======================================#

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()

for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])


    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = loss_func(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))

    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)





print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
