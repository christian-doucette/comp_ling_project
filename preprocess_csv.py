import csv
import re





#filters everything but letters and numbers from a review string
def preprocess_feature(review):
    review = re.sub('<.*?>', '', review) #removes all the html tags
    review = re.sub('[\'\"]', '', review) #removes all single/double quotes
    review = re.sub('[^0-9a-zA-Z\']+', ' ', review) #replaces all remaining non-alphanumeric characters with space
    review = review.lower()
    return review



#label of 1 means positive, label of -1 means negative
def preprocess_label(label):
    if (label=="positive"):
        return 1
    else:
        return -1


with open('train.csv', newline='') as f:
    reader = csv.reader(f)
    parsed_csv = list(reader)[1:100000]


with open('train_sanitized_nn_full.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
    for review,label in parsed_csv:

        writer.writerow([preprocess_feature(review), preprocess_label(label)])
