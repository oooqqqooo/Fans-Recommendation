#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:50:16 2017

@author: frx
"""
import requests
import json
import pandas as pd

headers = {'Accept': 'application/json'}
payload = {'api_key': '81c676610184e4ff89f8264d49b3889e'} 
response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
response = json.loads(response.text)
base_url = response['images']['base_url'] + 'w185'
print(base_url)

file_path = ''
def get_poster(imdb_url, base_url):
    # Get IMDB movie ID
    
    #response = requests.get(imdb_url)
    #movie_id = response.url.split('/')[-2]
    #s
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
    except Exception as err:  
        print(err)
        # IMDB movie ID is sometimes no good. Need to get correct one.
        '''
        movie_title = imdb_url.split('?')[-1].split('(')[0]
        payload['query'] = movie_title
        response = requests.get('http://api.themoviedb.org/3/search/movie', params=payload, headers=headers)
        movie_id = json.loads(response.text)['results'][0]['id']
        payload.pop('query', None)
        movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(movie_id)
        response = requests.get(movie_url, params=payload, headers=headers)
        file_path = json.loads(response.text)['posters'][0]['file_path']
        '''
        file_path = '/lFbBQ55MkBxVxQPwALjzMu3y9rD.jpg'
    print('file_path:'+file_path)
    return base_url + file_path


df_id = pd.read_csv('/home/frx/下载/Movielens/ml-latest-small/links.csv', sep=',')

idx_to_movie = {}
for row in df_id.itertuples():
    idx_to_movie[row[1]-1] = row[2]  
#print(idx_to_movie)

total_movies = 9000

movies = [0]*total_movies
for i in range(len(movies)):
    if i in idx_to_movie.keys() and len(str(idx_to_movie[i])) == 6:
        movies[i] = (idx_to_movie[i])
        #print(movies[i])
#print(type(movies))
movies = list(filter(lambda imdb: imdb != 0, movies))
total_movies  = len(list(movies))
print(total_movies)

URL = [0]*total_movies 
IMDB = [0]*total_movies 
URL_IMDB = {"url":[]}
i = 0
for movie in movies:
    print(movie)
    URL[i] = get_poster(movie, base_url)
    print(URL[i])
    if URL[i] != base_url+"":
        URL_IMDB["url"].append(URL[i])
        #URL_IMDB["imdb"].append(IMDB[i])
    i += 1 
# URL = filter(lambda url: url != base_url+"", URL)
df = pd.DataFrame(data=URL_IMDB) 

total_movies = len(df)  

import urllib


poster_path = '/home/frx/Test!!!/image/'    
for i in range(total_movies):
    print('++++++++'+df.url[i])
    urllib.request.urlretrieve(df.url[i], poster_path + str(i) + ".jpg")

