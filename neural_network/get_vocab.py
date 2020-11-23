import csv
from collections import Counter



#returns word_to_id dict and id_to_word list
def vocab_indices(file_path):
    vocab = Counter()

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        review_label_pairs = list(reader)
        for review,label in review_label_pairs:
            this_review = review.split()
            for word in this_review:
                vocab[word] += 1

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= 2})
    vocab_tuples = vocab_top.most_common(len(vocab_top))


    word_to_id = {word: i for i,(word, c) in enumerate(vocab_tuples)}
    id_to_word = [word for word, index in word_to_id.items()]
    return (word_to_id, id_to_word)
