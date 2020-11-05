from typing import List
import os
import time
import argparse
from argparse import Namespace
import logging


from scipy import sparse as sp #type: ignore
import numpy as np #type: ignore
from sklearn.utils.extmath import randomized_svd #type: ignore
from tqdm import tqdm #type: ignore
import pandas as pd #type: ignore
from scipy import sparse as sp #type: ignore
import torch #type: ignore

from acgan.module import *
from acgan.recommender import *

from ncf_utils import *

class DuckModel:
    """An adapter class"""
    def __init__(self, model):
        self.model = model 
    
    def predict(self, in_data, batch_size=100, verbose=0):
        users, items = in_data
        scores = self.model.score(users.tolist(), items.tolist())
        return scores

dataset = Dataset('data/ncf_data/ml-1m')
train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

uidx, iidx = train.nonzero()
rating = np.ones_like(uidx).astype(np.float32)
ts = np.arange(rating.shape[0])
train_df = pd.DataFrame({'uidx': uidx, 'iidx': iidx, 'rating': rating, 'ts': ts})
past_hist = train_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
user_num, item_num = train_df.uidx.max() + 1, train_df.iidx.max() + 1

evaluation_threads = 1
factor_num = 32
K = 10

factor = NCFModel(user_num, item_num, factor_num)
recom = ClassRecommender(user_num, item_num, factor)

recom.fit(train_df, 
          num_epochs=20,
          cuda=0,
          decay=1e-7, 
          num_neg=4,
          past_hist=past_hist, batch_size=256,
          lr=0.01)

duck_model = DuckModel(recom)
hit, ndcg = evaluate_model(duck_model, testRatings, testNegatives, K, evaluation_threads)
print(np.mean(hit), np.mean(ndcg))