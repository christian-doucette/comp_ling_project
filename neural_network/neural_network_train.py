import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter

whitelist_characters = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')


#filters everything but letters and numbers from a review string
#returns as an array of words
def preprocess_feature(review: str) -> str:
    filtered_chars = [ch.lower() for ch in review if ch in whitelist_characters]
    return ''.join(filtered_chars).split()



#label of 1 means positive, label of -1 means negative
def preprocess_label(label: str) -> int:
    if (label=="positive"):
        return 1
    else:
        return -1







test_data_words_freqs = Counter()

with open('../train_sanitized.csv', newline='') as f:
    reader = csv.reader(f)
    data = np.array(list(reader))
    for i in data:
        data1 = preprocess_feature(i[0])
        for j in data1:
            test_data_words_freqs[j] += 1

        #filtered = ''.join(filter(whitelist_characters.__contains__, i[0]))

top_words_only = Counter({k: c for k, c in test_data_words_freqs.items() if c >= 10})
print(top_words_only)







"""
#CSV dataset class
class CSV_Dataset(Dataset):

    def __init__(self, file_path):
        parsed_csv = pd.read_csv(file_path)

        features =
        labels = #list(map)
        self.X = torch.tensor(features)
        self.y = torch.tensor(labels)

    def __len__(self):
        return len(self.y)

    def __get_feature__(self, index):
        return self.X[index]

    def __get_label__(self, index):
        return self.y[index]



training_data = CSV_Dataset('practice.csv')
print(len(training_data))



print("Prints out first 10 rows from train.csv")
with open('train.csv', newline='') as f:
    reader = csv.reader(f)
    i = 1
    for row in reader:
        print(row)
        i += 1
        if i==11:
            break
"""
