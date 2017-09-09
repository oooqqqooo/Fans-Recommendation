#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:45:12 2017

@author: frx
"""
import pandas as pd
import numpy as np
df = pd.read_csv('/home/frx/下载/Movielens/ml-latest-small/ratings.csv', sep=',')
df_id = pd.read_csv('/home/frx/下载/Movielens/ml-latest-small/links.csv', sep=',')
df = pd.merge(df, df_id, on=['movieId'])

n_users = df.userId.unique().shape[0]
n_items = df.movieId.unique().shape[0]
print (str(n_users) + ' users')
print (str(n_items) + ' items')
# 建立我们的评分矩阵如下
rating_matrix = np.zeros((df.userId.unique().shape[0], max(df.movieId)))
for row in df.itertuples():
    rating_matrix[row[1]-1, row[2]-1] = row[3]
rating_matrix = rating_matrix[:,:9066]


def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# 检查我们的评分矩阵的稀疏度如下：

sparsity = float(len(rating_matrix.nonzero()[0]))
sparsity /= (rating_matrix.shape[0] * rating_matrix.shape[1])
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(sparsity))

# 为了训练和测试，我们将评级矩阵分成两个较小的矩阵。 
# 我们从评级矩阵中移除10个评级，并将它们放在测试集中 
train_matrix = rating_matrix.copy()
test_matrix = np.zeros(rating_matrix.shape)

for i in range(rating_matrix.shape[0]):
    rating_idx = np.random.choice(
        rating_matrix[i, :].nonzero()[0], 
        size=10, 
        replace=True)
    train_matrix[i, rating_idx] = 0.0
    test_matrix[i, rating_idx] = rating_matrix[i, rating_idx]

user_similarity = fast_similarity(train_matrix, kind='user')
movie_similarity = fast_similarity(train_matrix, kind='item')

from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

    
def predict_topk(ratings, similarity, kind='user', k=20):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred
    
pred = predict_topk(train_matrix, user_similarity, kind='user', k=40)
print ('Top-k User-based CF MSE: ' + str(get_mse(pred, test_matrix)))

pred = predict_topk(train_matrix, movie_similarity, kind='item', k=40)
print ('Top-k Item-based CF MSE: ' + str(get_mse(pred, test_matrix)))

'''
#where, s(u,v) *is just the cosine similarity measure between user *u and user v.
similarity_user = train_matrix.dot(train_matrix.T) + 1e-9
norms = np.array([np.sqrt(np.diagonal(similarity_user))])
similarity_user = ( similarity_user / (norms * norms.T) )

similarity_movie = train_matrix.T.dot(train_matrix) + 1e-9
norms = np.array([np.sqrt(np.diagonal(similarity_movie))])
similarity_movie = ( similarity_movie / (norms * norms.T) )


#其中用户u到电影i的预测是用户v给予电影i的用户u和v之间的相似度作为
#权重的等级的加权和（归一化）
from sklearn.metrics import mean_squared_error

prediction = similarity_user.dot(train_matrix) / np.array([np.abs(similarity_user).sum(axis=1)]).T
prediction = prediction[test_matrix.nonzero()].flatten()
test_vector = test_matrix[test_matrix.nonzero()].flatten()

mse = mean_squared_error(prediction, test_vector)



print ('MSE = ' + str(mse))
# MSE = 9.82582732089
'''

# 我们使用IMDB ID号码从电影数据库网站 The Movie Database使用其API获取电影海报
import requests
import json

headers = {'Accept': 'application/json'}
payload = {'api_key': '81c676610184e4ff89f8264d49b3889e'} 
response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
response = json.loads(response.text)
base_url = response['images']['base_url'] + 'w185'
print(base_url)

def get_poster(imdb_url, base_url):
    # Get IMDB movie ID
    
    #response = requests.get(imdb_url)
    #movie_id = response.url.split('/')[-2]
    movie_id = 'tt0'+str(imdb_url)
    print('movie_id:'+movie_id)
    
    # Query themoviedb.org API for movie poster path.
    movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(movie_id)
    print('movie_url:'+movie_url)
    headers = {'Accept': 'application/json'}
    payload = {'api_key': '81c676610184e4ff89f8264d49b3889e'} 
    response = requests.get(movie_url, params=payload, headers=headers)
    try:
        file_path = json.loads(response.text)['posters'][0]['file_path']
    except:
        # IMDB movie ID is sometimes no good. Need to get correct one.
        movie_title = imdb_url.split('?')[-1].split('(')[0]
        payload['query'] = movie_title
        response = requests.get('http://api.themoviedb.org/3/search/movie', params=payload, headers=headers)
        movie_id = json.loads(response.text)['results'][0]['id']
        payload.pop('query', None)
        movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(movie_id)
        response = requests.get(movie_url, params=payload, headers=headers)
        file_path = json.loads(response.text)['posters'][0]['file_path']
    print('file_path:'+file_path)
    return base_url + file_path
    

idx_to_movie = {}
for row in df_id.itertuples():
    idx_to_movie[row[1]-1] = row[2]
idx_to_movie    

k = 5  
idx = 3
movies = [ idx_to_movie[x] for x in np.argsort(movie_similarity[idx,:])[:-k-1:-1] ]
print(movies)
movies = filter(lambda imdb: len(str(imdb)) == 6, movies)

n_display = 5
URL = [0]*n_display
IMDB = [0]*n_display
i = 0
for movie in movies:
    print(movie)
    URL[i] = get_poster(movie, base_url)
    print(i)
    print(URL[i])
    i += 1 



import os,urllib,uuid 
#test
#toy_story = 'http://www.imdb.com/title/tt0114709/?ref_=fn_tt_tt_1'
# poster Image url
#url=get_poster(toy_story, base_url)

localPath='/home/frx/Test!!!/1.slmilarity image/2/'
def generateFileName(): 
  return str(uuid.uuid1()) 
  
    
#根据文件名创建文件  
def createFileWithFileName(localPathParam,fileName): 
  totalPath=localPathParam+'\\'+fileName 
  if not os.path.exists(totalPath): 
    file=open(totalPath,'a+') 
    file.close() 
    return totalPath 
    
#保存图片到文件夹
for i in URL:    
    if( len(i)!= 0 ): 
        fileName=generateFileName()+'.jpg'
        urllib.request.urlretrieve(i,createFileWithFileName(localPath,fileName)) 
 




