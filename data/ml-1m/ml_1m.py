#download data from :http://files.grouplens.org/datasets/movielens/ml-1m.zip
import os 
import pandas as pd
import numpy as np
from acgan.data import time_based_split

data_path='.'
names = ['uidx', 'iidx', 'rating', 'ts']
dtype = {'uidx':int, 'iidx':int, 'rating':float, 'ts':float}
ratings = pd.read_csv(os.path.join(data_path, 'ratings.dat'), 
sep='::', 
names=names,
dtype=dtype)
print(ratings.shape)
ratings.uidx = ratings.uidx - 1
ratings.iidx = ratings.iidx - 1
print(ratings.head())
ratings.to_feather(os.path.join(data_path, 'ratings.feather'))
time_based_split(ratings, data_path, 20)

