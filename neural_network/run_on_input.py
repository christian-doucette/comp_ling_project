from  collections import Counter
import json
import re
import torch
import torch.nn as nn
import numpy as np
import rnn_class


# Loads in word_to_id and trained network

with open('word_to_id.json') as json_file:
    word_to_id = Counter(json.load(json_file))

net = torch.load('trained_network.pt')
net.eval()





def preprocess_text(review, word_map, reviews_len=400):
    review = re.sub('<.*?>', '', review) #removes all the html tags
    review = re.sub('[\'\"]', '', review) #removes all single/double quotes
    review = re.sub('[^0-9a-zA-Z\']+', ' ', review) #replaces all remaining non-alphanumeric characters with space
    review = review.lower()
    words = review.split()
    ids = [word_map[word] for word in words]

    padding = [0] * (reviews_len - len(ids))
    return padding + ids[0:reviews_len]







review =  "This movie is not too shabby."


test_text = preprocess_text(review, word_to_id)


hidden = net.init_hidden(1)
input = torch.tensor([test_text])
output, hidden = net.forward(input, hidden)
print(output)
