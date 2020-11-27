import csv
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


def embed_feature(sentence_string, review_lens, vocab):
    embedded_review = []
    for word in sentence_string.split():
        if word in vocab:
            embedded_review.append(vocab[word])
        else:
            embedded_review.append(0)



        #review_lens is number of words in each review input
        #reviews with less words are front-padded with 0
        #reviews iwth more words are truncated
        padding = [0] * (review_lens - len(embedded_review))

    return padding + embedded_review[0:review_lens]



def preprocess_csv(file_path, review_lens, vocab):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        parsed_csv = list(reader)

    features = []
    labels = []
    for feature,label in parsed_csv:
        ids_feature = []

        features.append(embed_feature(feature, review_lens, vocab))
        labels.append(int(label=="1"))

    return TensorDataset(torch.tensor(features), torch.tensor(labels), torch.tensor(range(len(parsed_csv))))
