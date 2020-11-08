import csv


print("Prints out first 10 rows from train.csv")
with open('train.csv', newline='') as f:
    reader = csv.reader(f)
    i = 1
    for row in reader:
        print(row)
        i += 1
        if i==11:
            break
