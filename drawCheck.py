import dataStandarization
import tensorflow as tf
import numpy as np
from numpy import array

datapaths = ['/dataset emotes/single.csv']
labels = ['banana', 'sliceofpizza', 'cloud', 'thumbsupsign', 'cherryblossom',
            'hotbeverage', 'sun', 'redapple', 'heavyblackheart', 'fallenleaf']


all_X = []
all_Y = []
temp = dataStandarization.preprocess(datapaths)

for iter in range(1):
    # Making blank 2D, 100x100 matrix for serialization of drawing
    a = [[0] * 100 for i in range(100)]
    # Applying drawing to a matrix
    for x in temp[iter][1]:
        a[x[0]][x[1]] = 1
    all_X.append(a)
    all_Y.append(temp[iter][0])

np_X = array(all_X)
np_Y = array(all_Y)

nsamples, nx, ny = np_X.shape
np_X_reshape = np_X.reshape((nsamples,nx*ny))

class_names = list(set(np_Y))


model = tf.keras.models.load_model('./my_model.h5')
predict_array = model.predict(np_X_reshape)

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
formatted_list = ["%.2f"%item for item in predict_array.tolist()[0]]
# result = dict(zip(sorted(labels),formatted_list))
print(formatted_list)
