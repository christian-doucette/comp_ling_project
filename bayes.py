import csv
import numpy 
import math



print("Prints out first 10 rows from train.csv")
with open('train.csv') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        #print(row)
        for word in row[0].split():
        	print(word)
        i += 1
        if i==2:
            break
