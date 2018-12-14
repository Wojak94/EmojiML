import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Function to scale original 1000 x 1000 drawing by a factor of 0.1
def scale(x):
    a = x.split('; ')
    for num in range(len(a)):
        a[num] = a[num].split(' ')
        for fact in range(len(a[num])):
            if a[num][fact]:
                a[num][fact] = int(round(int(a[num][fact]) * 0.1))
            else:
                del a[num]
    return a

# Reading data from csv to pandas DataFrame to prepare
path = os.getcwd() + '/dataset emotes/Dataset.csv'
data = pd.read_csv(path)
# Droping 'id' column and deleting blank rows in data
data.drop('id', axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)
data['dataset_points'] = data['dataset_points'].map(scale) # Scaling drowing to save space

# Reading drawing from DataFrame to multi-dimensional list
temp = []
for row in data.iterrows():
    index, dt = row
    temp.append(dt.tolist())

# Making blank 2D, 100x100 matrix for serialization of drawing
a = [[0] * 100 for i in range(100)]

# Applying drawing to a matrix
for x in temp[100][1]:
    a[x[0]][x[1]] = 1
print(temp[200][0])

# Ploting matrix as a image
plt.imshow(a)
plt.show()
