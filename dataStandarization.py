import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Paths to csv datasets
datapaths = ['/dataset emotes/Dataset.csv',
            '/dataset emotes/Dataset2.csv',
            '/dataset emotes/Dataset3.csv',
            '/dataset emotes/Dataset4.csv']

# Helper function to scale original 1000 x 1000 drawing by a factor of 0.1
def scale(x):
    a = x.split('; ')
    for num in range(len(a)):
        a[num] = a[num].split(' ')
        for fact in range(len(a[num])):
            if a[num][fact]:
                a[num][fact] = int(round(int(a[num][fact]) * 0.1))
                if a[num][fact] >= 100:
                    a[num][fact] = 99
            else:
                del a[num]
    return a

# Function to preprocess data, delete blank records and scale data
def preprocess(paths):
    temp = []
    for path in paths:
        # Reading data from csv to pandas DataFrame to prepare
        print('Processing path: ' + path)
        path = os.getcwd() + path
        data = pd.read_csv(path)
        # Droping 'id' column and deleting blank rows in data
        data.drop('id', axis=1, inplace=True)
        data.dropna(axis=0, how='any', inplace=True)
        data['dataset_points'] = data['dataset_points'].map(scale)
        # Reading drawing from DataFrame to multi-dimensional list
        for row in data.iterrows():
            index, dt = row
            temp.append(dt.tolist())
    print(f'Total number of records: {len(temp)}')
    return temp

final_data = []
final_class = []
temp = preprocess(datapaths)

for iter in range(len(temp)):
    # Making blank 2D, 100x100 matrix for serialization of drawing
    a = [[0] * 100 for i in range(100)]
    # Applying drawing to a matrix
    for x in temp[iter][1]:
        a[x[0]][x[1]] = 1
    final_data.append(a)
    final_class.append(temp[iter][0])

#ploting "random" image
plt.imshow(final_data[4000])
plt.show()
