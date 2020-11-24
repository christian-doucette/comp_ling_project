import csv
import re
from collections import Counter
import matplotlib.pyplot as plt


#filters everything but letters and numbers from a review string
review_lengths = Counter()

with open('train_sanitized_nn_full.csv', newline='') as f:
    reader_train = csv.reader(f)
    parsed_csv_train = list(reader_train)


for review,label in parsed_csv_train:
    this_len = len(review.split())
    review_lengths[this_len] += 1


with open('test_sanitized_nn_full.csv', newline='') as f:
    reader_test = csv.reader(f)
    parsed_csv_test = list(reader_test)

for review,label in parsed_csv_test:
    this_len = len(review.split())
    review_lengths[this_len] += 1

lens = [k for k,c in review_lengths.items()]
cnts = [c for k,c in review_lengths.items()]



print(f'Max: {max(lens)}')
print(f'Min: {min(lens)}')


average = 0
num_total = 0
for i in range(0, max(lens)+1):
    average += i * review_lengths[i]
    num_total += review_lengths[i]
print(f'Average: {average}/{num_total} = {float(average)/num_total}')

k = 400
num_leq_k = 0
for i in range(0, k+1):
    num_leq_k += review_lengths[i]

num_over_k = 0
for i in range(k+1, max(lens)+1):
    num_over_k += review_lengths[i]

print(f'Number over {k}: {num_over_k}')
print(f'Number leq {k}: {num_leq_k}')


plt.plot(lens, cnts, 'ro')
plt.title('Review lengths histogram')
plt.xlabel('Review lengths (in words)')
plt.ylabel('Count')
plt.show()
