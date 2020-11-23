import csv
from collections import Counter

whitelist_characters = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')


#filters everything but letters and numbers from a review string
#returns as an array of words
def preprocess_feature(review):
    filtered_chars = [ch.lower() for ch in review if ch in whitelist_characters]
    return ''.join(filtered_chars).split()





#returns word_to_id dict and id_to_word list
def vocab_indices(file_path):
    vocab = Counter()

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        review_label_pairs = list(reader)
        for review,label in review_label_pairs:
            this_review = preprocess_feature(review)
            for word in this_review:
                vocab[word] += 1

            #filtered = ''.join(filter(whitelist_characters.__contains__, i[0]))

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= 2})
    vocab_tuples = vocab_top.most_common(len(vocab_top))


    word_to_id = {word: i for i,(word, c) in enumerate(vocab_tuples)}
    id_to_word = [word for word, index in word_to_id.items()]
    return (word_to_id, id_to_word)
