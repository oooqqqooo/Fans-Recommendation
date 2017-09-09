#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 18:32:44 2017

@author: frx
"""

import pandas as pd
import numpy as np

total_movies = 3088 
poster_path = '/home/frx/Test!!!/image/' 



    
    

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import pylab

image = [0]*total_movies
x = [0]*total_movies
print(type(x))

for i in range(total_movies):
    image[i] = kimage.load_img(poster_path + str(i) + ".jpg", target_size=(278, 185))
    x[i] = kimage.img_to_array(image[i])
    x[i] = np.expand_dims(x[i], axis=0)
    x[i] = preprocess_input(x[i]) 
print(x[1].shape,x[1].dtype)
print(x[1])
model = VGG16(include_top=False, weights='imagenet')

prediction = [0]*total_movies
matrix_res = np.zeros([total_movies,20480])
for i in range(total_movies):
    prediction[i] = model.predict(x[i]).ravel()
    matrix_res[i,:] = prediction[i] 

similarity_deep = matrix_res.dot(matrix_res.T)
norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
similarity_deep = similarity_deep / norms / norms.T 

df_id = pd.read_csv('/home/frx/下载/Movielens/ml-latest-small/links.csv', sep=',')

idx_to_movie = {}
for row in df_id.itertuples():
    idx_to_movie[row[1]-1] = row[2]
idx_to_movie    

k = 5  
idx = 9

movies = [ x[i] for i in np.argsort(similarity_deep[idx,:])[:-k-1:-1] ]

print(len(movies))


for i in movies:
    try:
        print(i.shape,i.dtype)
        pylab.imshow(i[0, :, :,:],cmap=pylab.cm.bone)
        pylab.show()
    except Exception as err:
        print(err)
   
        

