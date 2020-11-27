import csv
from collections import Counter



# returns word_to_id dictionary that maps a word in the vocabulary to its id
# ids start at 1 and lower ids correspond to words that occur less in the training data

def vocab_indices(file_path, min_occurences):
    vocab = Counter()

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        review_label_pairs = list(reader)
        for review,label in review_label_pairs:
            this_review = review.split()
            for word in this_review:
                vocab[word] += 1

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= min_occurences})
    vocab_tuples = vocab_top.most_common(len(vocab_top))


    word_to_id = {word: i+1 for i,(word, c) in enumerate(vocab_tuples)}
    return word_to_id
