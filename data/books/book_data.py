"""
Steps to download the data:
pip install gdown
gdown 'https://drive.google.com/uc?id=1roQnVtWxVE1tbiXyabrotdZyUY7FA82W'

or go to: https://github.com/MengtingWan/goodreads
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


def load_data(file_name, head = 500):
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            user_id = d['user_id']
            item_id = d['book_id']
            rating = float(d['rating'])
            ts_str = d['date_updated'].strip().split(' ')
            ts_str = ' '.join(ts_str[:4] + ts_str[-1:])
            ts = datetime.strptime(ts_str, '%c').timestamp()
            x = [user_id, item_id, rating, ts]
            data.append(x)
            
            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'ts'])
    return df

data_path = '.'
books_df = load_data('goodreads_reviews_history_biography.json.gz', head=None)
valid_books = books_df
print(valid_books.shape)

# first stage filter on item
item_view_count = valid_books.groupby('item_id').count().user_id.reset_index()
item_view_count = item_view_count[(item_view_count.user_id > 20)]
item_view_count = item_view_count.item_id.to_frame()
valid_books = pd.merge(left=valid_books, right=item_view_count, on='item_id')
print(valid_books.shape)

# second stage filter on user
user_view_count = valid_books.groupby('user_id').count().item_id.reset_index()
user_view_count = user_view_count[(user_view_count.item_id > 20) & (user_view_count.item_id < 1000)]
user_view_count = user_view_count.user_id.to_frame()
valid_books = pd.merge(left=valid_books, right=user_view_count, on='user_id')

print(valid_books.shape)
valid_books['uidx'] = LabelEncoder().fit_transform(valid_books.user_id)
valid_books['iidx'] = LabelEncoder().fit_transform(valid_books.item_id)
valid_books = valid_books[['uidx', 'iidx', 'rating', 'ts']]
print(f'max uidx:{valid_books.uidx.max()}, max iidx:{valid_books.iidx.max()}')
print(valid_books)
valid_books.to_feather(os.path.join(data_path, 'ratings.feather'))
time_based_split(valid_books, data_path, 20)

