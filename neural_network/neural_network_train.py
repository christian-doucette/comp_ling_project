import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

#My code files
import rnn_class
import get_vocab
import preprocess


#=======================================#
#       Preprocessing Parameters        #
#=======================================#

reviews_len = 400       # length of each review
batch_size = 50         # size of batches used in train_loader and test_loader
min_occurences = 20     # min number of occurences a word needs to occur in the vocabulary



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

embedding_dim = 512     # size of the word embeddings
hidden_dim = 256        # size of the hidden state
n_layers = 3            # number of LSTM layers



#=======================================#
#           Initializing RNN            #
#=======================================#

net = rnn_class.SA_RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

learning_rate = 0.001
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



#=======================================#
#          Training Parameters          #
#=======================================#
epochs = 4 # 3-4 is normal
clip = 5 # gradient clipping



#=======================================#
#             RNN Training              #
#=======================================#

net.train()
for e in range(epochs):
    print(f'Epoch {e+1}')
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # loop through each batch using loader
    for inputs, labels, _ in train_loader:

        # create new variable for hidden state, to reset training history
        h = tuple([each.data for each in h])

        # sets gradient to zero
        net.zero_grad()

        # get the netowrk output for this batch
        output, h = net.forward(inputs, h)

        # calculate loss gradient with backpropogation
        loss = loss_func(output.squeeze(), labels.float())
        loss.backward()

        # the guide I used said to include this to avoid the "exploding gradient problem"
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()




#=======================================#
#             RNN Testing               #
#=======================================#

misclassifications = [] #examples of reviews that the model misclassifies
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()

# loops through each batch once, so will count each test review once
for inputs, labels, indices in test_loader:

    # create new variable for hidden state, to reset training history
    h = tuple([each.data for each in h])

    # get the netowrk output for this batch
    output, h = net.forward(inputs, h)

    # calculate loss
    test_loss = loss_func(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))

    # collect misclassifications
    incorrect_tensor = torch.logical_not(correct_tensor)
    misclassifications_this_batch = torch.masked_select(indices, incorrect_tensor)
    misclassifications = misclassifications + misclassifications_this_batch.tolist()


    # count number of correct classifications
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)


misclassifications.sort()
print(misclassifications)

print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
