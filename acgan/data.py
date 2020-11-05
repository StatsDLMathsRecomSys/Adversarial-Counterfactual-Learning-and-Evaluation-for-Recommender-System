import os
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Set

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse as sp  # type: ignore
import torch  # type: ignore
from torch.utils import data  # type: ignore
from numpy.random import RandomState  # type: ignore


def ml_1m(
        data_path: str,
        train_path: str,
        val_path: str,
        test_path: str) -> None:
    ratings = pd.read_csv(
        os.path.join(
            data_path,
            'ratings.dat'),
        sep='::',
        names=[
            'uidx',
            'iidx',
            'rating',
            'ts'],
        dtype={
            'uidx': int,
            'iidx': int,
            'rating': float,
            'ts': float})
    print(ratings.shape)
    ratings.uidx = ratings.uidx - 1
    ratings.iidx = ratings.iidx - 1
    print(ratings.head())
    ratings.to_feather(os.path.join(data_path, 'ratings.feather'))

    user_hist: Dict[int, List[Tuple[int, float]]] = {}
    for row in ratings.itertuples():
        if row.uidx not in user_hist:
            user_hist[row.uidx] = []
        user_hist[row.uidx].append((row.iidx, row.ts))
    # sort by ts in descending order
    # row represents the user, columns represents the item
    train_record: List[Tuple[int, int]] = []
    val_record: List[Tuple[int, int]] = []
    test_record: List[Tuple[int, int]] = []

    for uidx, hist in user_hist.items():
        ord_hist = [x[0] for x in sorted(hist, key=lambda x: x[1])]
        assert(len(ord_hist) >= 20)
        for v in ord_hist[:-2]:
            train_record.append((uidx, v))

        val_record.append((uidx, ord_hist[-2]))
        test_record.append((uidx, ord_hist[-1]))

    train_dat = np.ones(len(train_record))
    val_dat = np.ones(len(val_record))
    test_dat = np.ones(len(test_record))
    train_npy = np.array(train_record)
    val_npy = np.array(val_record)
    test_npy = np.array(test_record)

    mat_shape = (ratings.uidx.max() + 1, ratings.iidx.max() + 1)
    train_csr = sp.csr_matrix((train_dat, (train_npy[:, 0], train_npy[:, 1])),
                              shape=mat_shape)
    val_csr = sp.csr_matrix((val_dat, (val_npy[:, 0], val_npy[:, 1])),
                            shape=mat_shape)
    test_csr = sp.csr_matrix((test_dat, (test_npy[:, 0], test_npy[:, 1])),
                             shape=mat_shape)

    sp.save_npz(train_path, train_csr)
    sp.save_npz(val_path, val_csr)
    sp.save_npz(test_path, test_csr)


def time_based_split(
        ratings: pd.DataFrame,
        data_path: str,
        min_len: int = 20) -> None:
    names = ['uidx', 'iidx', 'rating', 'ts']
    if (ratings.columns == names).min() < 1:
        raise ValueError(
            f"Only support data frame with columns ['uidx', 'iidx', 'rating', 'ts'], the input is {ratings.columns}")
    user_hist: Dict[int, List[Tuple[int, float, float]]] = {}
    for row in ratings.itertuples():
        if row.uidx not in user_hist:
            user_hist[row.uidx] = []
        user_hist[row.uidx].append((row.iidx, row.rating, row.ts))
    # sort by ts in descending order
    train_record = {x: [] for x in names}
    val_record = {x: [] for x in names}
    test_record = {x: [] for x in names}

    def put2record(record, u, obs):
        record['uidx'].append(u)
        record['iidx'].append(obs[0])
        record['rating'].append(obs[1])
        record['ts'].append(obs[2])

    for uidx, hist in user_hist.items():
        ord_hist = [x for x in sorted(hist, key=lambda x: x[-1])]
        assert(len(ord_hist) >= 20)
        for v in ord_hist[:-2]:
            put2record(train_record, uidx, v)
        put2record(val_record, uidx, ord_hist[-2])
        put2record(test_record, uidx, ord_hist[-1])
    train_path = os.path.join(data_path, 'train.feather')
    pd.DataFrame(train_record).to_feather(train_path)
    val_path = os.path.join(data_path, 'val.feather')
    pd.DataFrame(val_record).to_feather(val_path)
    test_path = os.path.join(data_path, 'test.feather')
    pd.DataFrame(test_record).to_feather(test_path)


def ml_1m_v2(data_path: str) -> None:
    names = ['uidx', 'iidx', 'rating', 'ts']
    dtype = {'uidx': int, 'iidx': int, 'rating': float, 'ts': float}
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


class NegSeqData(data.Dataset):
    def __init__(self,
                 features: List[Tuple[int,
                                      int]],
                 num_item: int,
                 num_neg: int = 0,
                 is_training: bool = False,
                 seed: int = 123,
                 past_hist: Optional[Dict[int,
                                          Set[int]]] = None) -> None:
        super(NegSeqData, self).__init__()
        """ Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        self.features = features
        self.num_item = num_item
        self.train_set = set(features)
        self.num_neg = num_neg
        self.is_training = is_training
        self.past_hist = past_hist
        self.prng = RandomState(seed)

    def ng_sample(self) -> None:
        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            j_list = []
            for _ in range(self.num_neg):
                is_dup = True
                while is_dup:
                    j = self.prng.randint(self.num_item)
                    is_dup = (u, j) in self.train_set
                    if self.past_hist is not None:
                        is_dup = is_dup or j in self.past_hist.get(u, [])
                j_list.append(j)
            self.features_fill.append([u, i, j_list])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features
        user = features[idx][0]
        item_i = features[idx][1]
        item_j_list = np.array(features[idx][2]) if \
            self.is_training else features[idx][1]
        return user, item_i, item_j_list


class NegSampleData(data.Dataset):
    def __init__(self,
                 features: List[Tuple[int,
                                      int]],
                 num_item: int,
                 num_neg: int = 0,
                 is_training: bool = False,
                 seed: int = 123) -> None:
        super(NegSampleData, self).__init__()
        """ Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        self.features = features
        self.num_item = num_item
        self.train_set = set(features)
        self.num_neg = num_neg
        self.is_training = is_training
        self.prng = RandomState(seed)

    def ng_sample(self) -> None:
        assert self.is_training, 'no need to sample when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for _ in range(self.num_neg):
                j = self.prng.randint(self.num_item)
                while (u, j) in self.train_set:
                    j = self.prng.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self) -> int:
        return self.num_neg * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j


class RatingData(data.Dataset):
    def __init__(self, features: List[Tuple[int, int, float]]) -> None:
        super(RatingData, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class NegSequenceData(data.Dataset):
    def __init__(self, hist: Dict[int, List[int]], 
                max_len: int, 
                padding_idx: int,
                item_num: int,
                num_neg: int = 0,
                is_training: bool = False,
                past_hist: Optional[Dict[int, Set[int]]] = None,
                seed: int = 123,
                window: bool = True,
                allow_empty: bool =False) -> None:
        super(NegSequenceData, self).__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.num_item = item_num
        self.num_neg = num_neg
        self.past_hist = past_hist
        self.prng = RandomState(seed)
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Build windowed data')
        self.records = []
        for uidx, item_list in hist.items():
            if window:
                for i in range(len(item_list)):
                    item_slice = item_list[max(0, i - max_len):i]
                    if not allow_empty and len(item_slice) == 0:
                        continue
                    self.records.append([uidx, item_list[i], item_slice])
            else:
                if not allow_empty and len(item_list) == 1:
                    continue
                self.records.append([uidx, item_list[-1], item_list[-(max_len + 1):-1]])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        temp_hist = np.zeros(self.max_len, dtype=int) + self.padding_idx
        uidx, pos_item, item_hist =  self.records[idx]
        assert(len(temp_hist) >= len(item_hist))
        if len(item_hist) > 0:
            temp_hist[-len(item_hist):] = item_hist

        negitem_list = np.zeros(self.num_neg, dtype=int)
        for idx in range(self.num_neg):
            is_dup = True
            while is_dup:
                negitem = self.prng.randint(self.num_item)
                is_dup = negitem == pos_item
                if self.past_hist is not None:
                    is_dup = is_dup or negitem in self.past_hist.get(uidx, [])
            negitem_list[idx] = negitem
        return uidx, pos_item, negitem_list, temp_hist

        


if __name__ == '__main__':
    # ml_1m('/mnt/c0r00zy/a()c_gan/data/ml-1m',
    # '/mnt/c0r00zy/ac_gan/data/ml-1m/train.npz',
    # '/mnt/c0r00zy/ac_gan/data/ml-1m/val.npz',
    # '/mnt/c0r00zy/ac_gan/data/ml-1m/test.npz')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    ml_1m_v2(args.data_path)
