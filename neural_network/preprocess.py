import csv
import torch
from torch.utils.data import TensorDataset


# takes in a sentence and converts ot to a list of ids, where each id corresponds to a word
# also sets all to length review_lens:
#       if it is shorter, front-pads with 0
#       if it is longer, truncates

def embed_feature(sentence_string, review_lens, vocab):
    embedded_review = []
    for word in sentence_string.split():
        if word in vocab:
            embedded_review.append(vocab[word])
        else:
            embedded_review.append(0)

        padding = [0] * (review_lens - len(embedded_review))

    return padding + embedded_review[0:review_lens]




# takes in a csv filepath and returns a TensorDataset corresponding to that data

def preprocess_csv(file_path, review_lens, vocab):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        parsed_csv = list(reader)

    features = []
    labels = []
    for feature,label in parsed_csv:
        features.append(embed_feature(feature, review_lens, vocab))
        labels.append(int(label=="1"))

    return TensorDataset(torch.tensor(features), torch.tensor(labels), torch.tensor(range(len(parsed_csv))))
