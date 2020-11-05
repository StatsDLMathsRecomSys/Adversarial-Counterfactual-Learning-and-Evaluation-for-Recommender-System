"""
from http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
"""
import os 
import re
import sys
import gzip
import json
from datetime import datetime
import pandas as pd
import numpy as np
from acgan.data import time_based_split
from sklearn.preprocessing import LabelEncoder

data_path='.'
names = ['user_id', 'item_id', 'tag', 'ts']
dtype = {'user_id':int, 'item_id':int, 'tag':int, 'ts':float}
ratings = pd.read_csv(os.path.join(data_path, 'user_taggedartists-timestamps.dat'), 
sep='\t', 
names=names,
dtype=dtype, skiprows=1)
print(f'frame shape: {ratings.shape}')

# first stage filter on item
valid_ratings = ratings
item_view_count = valid_ratings.groupby('item_id').count().user_id.reset_index()
item_view_count = item_view_count[(item_view_count.user_id > 20)]
item_view_count = item_view_count.item_id.to_frame()
valid_ratings = pd.merge(left=valid_ratings, right=item_view_count, on='item_id')
print(f'frame shape: {valid_ratings.shape}')


# second stage filter on user
user_view_count = valid_ratings.groupby('user_id').count().item_id.reset_index()
user_view_count = user_view_count[(user_view_count.item_id > 20) & (user_view_count.item_id < 1000)]
user_view_count = user_view_count.user_id.to_frame()
valid_ratings = pd.merge(left=valid_ratings, right=user_view_count, on='user_id')
print(f'frame shape: {valid_ratings.shape}')

print(valid_ratings.shape)
valid_ratings['uidx'] = LabelEncoder().fit_transform(valid_ratings.user_id)
valid_ratings['iidx'] = LabelEncoder().fit_transform(valid_ratings.item_id)
print(min(valid_ratings.tag))
valid_ratings['rating'] = (valid_ratings.tag > 0).astype(np.float32)
valid_ratings = valid_ratings[['uidx', 'iidx', 'rating', 'ts']]
print(f'max uidx:{valid_ratings.uidx.max()}, max iidx:{valid_ratings.iidx.max()}')


print(valid_ratings)
valid_ratings.to_feather(os.path.join(data_path, 'ratings.feather'))
time_based_split(valid_ratings, data_path, 20)

