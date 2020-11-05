from typing import List
from scipy import sparse as sp #type: ignore
import numpy as np #type: ignore
from sklearn.utils.extmath import randomized_svd #type: ignore
from tqdm import tqdm

from acgan.recommender import SVDRecommender, BPRRecommender, eval_test
from acgan.module import FactorModel


# te = sp.load_npz('data/ml-1m/test.npz')
# val = sp.load_npz('data/ml-1m/val.npz')
# tr = sp.load_npz('data/ml-1m/train.npz')

# dim=32

# sv = SVDRecommender(tr.shape[0], tr.shape[1], dim)
# print(f'model with dimension {dim}')
# sv.fit(tr)
# eval_test(te, sv, cut_len=10) 

# factor_model = FactorModel(tr.shape[0], tr.shape[1], dim)
# bpr = BPRRecommender(tr.shape[0], tr.shape[1], factor_model)
# bpr.fit(tr, val, cuda=0, num_neg=4)
# eval_test(te, bpr, cut_len=10)


from scipy import sparse as sp #type: ignore
from acgan.module import FactorModel, BetaModel
from acgan.recommender import ac_train

te = sp.load_npz('data/ml-1m/test.npz')
val = sp.load_npz('data/ml-1m/val.npz')
tr = sp.load_npz('data/ml-1m/train.npz')

dim=32
f = FactorModel(user_num=tr.shape[0], item_num=tr.shape[1], factor_num=dim)
g = FactorModel(user_num=tr.shape[0], item_num=tr.shape[1], factor_num=dim)
beta = BetaModel(user_num=tr.shape[0], item_num=tr.shape[1])

ac_train(f, g, beta, tr, val)
