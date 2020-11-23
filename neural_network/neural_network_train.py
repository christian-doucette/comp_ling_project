import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import get_vocab

whitelist_characters = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
word_to_id, id_to_word = get_vocab.vocab_indices('../practice.csv')


#filters everything but letters and numbers from a review string
#returns as an array of words
def preprocess_feature(review):
    filtered_chars = [ch.lower() for ch in review if ch in whitelist_characters]
    return ''.join(filtered_chars).split()



#label of 1 means positive, label of -1 means negative
def preprocess_label(label):
    if (label=="positive"):
        return 1
    else:
        return -1






#CSV dataset class
class CSV_Dataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            parsed_csv = list(reader)[1:2501]



        features = []
        labels = []
        for feature,label in parsed_csv:
            embedded_feature = []
            for word in preprocess_feature(feature):
                if word in word_to_id:
                    embedded_feature.append(word_to_id[word])
                else:
                    embedded_feature.append(-1)

            features.append(embedded_feature)
            labels.append(preprocess_label(label))

        self.X = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __get_feature__(self, index):
        return self.X[index]

    def __get_label__(self, index):
        return self.y[index]



training_data = CSV_Dataset('../practice.csv')
