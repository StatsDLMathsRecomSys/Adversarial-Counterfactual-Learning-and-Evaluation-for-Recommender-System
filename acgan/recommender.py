from typing import List, Optional, Tuple, Dict, Set
import time
import logging

from tqdm import tqdm  # type: ignore
from scipy import sparse as sp  # type: ignore
import numpy as np  # type: ignore
from sklearn.utils.extmath import randomized_svd  # type: ignore
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.utils import data  # type: ignore
import pandas as pd  # type: ignore
from numpy.random import RandomState  # type: ignore

from acgan.module import PopularModel
from acgan.data import NegSampleData, RatingData, NegSeqData, NegSequenceData


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def build_optimizer(lr, *models):
    # minimizer
    optimizer_list = []
    sparse_weight = []
    dense_weight = []
    for model in models:
        sparse_weight.extend(model.get_sparse_weight())
        dense_weight.extend(model.get_dense_weight())

    if len(sparse_weight) > 0:
        optimizer_list.append(torch.optim.SparseAdam(
            params=sparse_weight, lr=lr))
    if len(dense_weight) > 0:
        optimizer_list.append(torch.optim.Adam(params=dense_weight, lr=lr))

    if len(optimizer_list) < 1:
        raise ValueError('Need at least one dense or sparse weights')
    optimizer = MultipleOptimizer(*optimizer_list)
    return optimizer


class Recommender:
    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        raise NotImplementedError()
        return np.zeros(0)

    def recommend(self, u_s: int, cand_b: List[int], top_k: int) -> List[int]:
        u_b = [u_s] * len(cand_b)
        scores = self.score(u_b, cand_b)
        top_k_ind = scores.argsort()[::-1][:top_k]
        return [cand_b[ind] for ind in top_k_ind]

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()



class PopRecommender(Recommender):
    def __init__(self, pop_module: nn.Module) -> None:
        self.pop_module = pop_module

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        with torch.no_grad():
            device = self.pop_module.get_device()
            self.pop_module.eval()
            u_b_t = torch.LongTensor(u_b).to(device)  # type: ignore
            v_b_t = torch.LongTensor(v_b).to(device)  # type: ignore
            scores = self.pop_module(u_b_t, v_b_t)
            return scores.cpu().numpy()

class RandRecommender(Recommender):
    def __init__(self,  max_u: int, max_v: int) -> None:
        self.max_u = max_u
        self.max_v = max_v

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        return np.random.rand(len(u_b))

class UserBasedKnn(Recommender):
    def __init__(self,  max_u: int, max_v: int) -> None:
        self.max_u = max_u
        self.max_v = max_v
        self.user_item_score = None

    def fit(self, df: pd.DataFrame) -> None:
        row, col = df.uidx, df.iidx
        mat = sp.csr_matrix((df.rating, (row, col)), shape=(self.max_u, self.max_v))
        uu_weight = mat.dot(mat.T) + sp.eye(self.max_u)
        self.user_item_score = uu_weight.dot(mat)

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        return np.asarray(self.user_item_score[u_b, v_b]).reshape(-1)

class PopRecommenderV2(Recommender):
    def __init__(self,  max_u: int, max_v: int) -> None:
        self.max_u = max_u
        self.max_v = max_v
        self.pop_module = None

    def fit(self, df: pd.DataFrame) -> None:
        item_cnt_dict = df.groupby('iidx').count().uidx.to_dict()
        item_cnt = np.array([item_cnt_dict.get(iidx, 0) for iidx in range(self.max_v)])
        self.pop_module = PopularModel(item_cnt)
        
    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        with torch.no_grad():
            device = self.pop_module.get_device()
            self.pop_module.eval()
            u_b_t = torch.LongTensor(u_b).to(device)  # type: ignore
            v_b_t = torch.LongTensor(v_b).to(device)  # type: ignore
            scores = self.pop_module(u_b_t, v_b_t)
            return scores.cpu().numpy()

class SVDRecommender(Recommender):
    def __init__(self, max_u: int, max_v: int, num_factors: int) -> None:
        self.USER_factors = np.zeros((max_u, num_factors))
        self.ITEM_factors = np.zeros((max_v, num_factors))
        self.num_factors = num_factors

    def fit(self, train_mat: sp.csr_matrix) -> None:
        U, Sigma, VT = randomized_svd(train_mat,
                                      n_components=self.num_factors,
                                      # n_iter=5,
                                      random_state=None)

        s_Vt = sp.diags(Sigma) * VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        s = self.USER_factors[u_b] * self.ITEM_factors[v_b]
        return s.sum(1)

class SVDRecommenderV2(Recommender):
    def __init__(self, max_u: int, max_v: int, num_factors: int) -> None:
        self.USER_factors = np.zeros((max_u, num_factors))
        self.ITEM_factors = np.zeros((max_v, num_factors))
        self.max_u = max_u
        self.max_v = max_v
        self.num_factors = num_factors

    def fit(self, df: pd.DataFrame) -> None:
        row, col = df.uidx, df.iidx
        mat = sp.csr_matrix((df.rating, (row, col)), shape=(self.max_u, self.max_v))

        U, Sigma, VT = randomized_svd(mat,
                                      n_components=self.num_factors,
                                      # n_iter=5,
                                      random_state=None)

        s_Vt = sp.diags(Sigma) * VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        s = self.USER_factors[u_b] * self.ITEM_factors[v_b]
        return s.sum(1)

class ContextItemKnn(Recommender):
    def __init__(self, max_u: int, max_v: int, item_embed: np.ndarray) -> None:
        self.max_u = max_u
        self.max_v = max_v
        self.ITEM_factors = item_embed
        self.USER_factors = np.zeros((max_u, item_embed.shape[1]))

    def fit(self, df: pd.DataFrame) -> None:
        for uidx, iidx, rating in zip(df.uidx, df.iidx, df.rating):
            if rating > 0:
                self.USER_factors[uidx, :] += self.ITEM_factors[iidx, :]
        
    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        s = self.USER_factors[u_b] * self.ITEM_factors[v_b]
        return s.sum(1)


class BPRRecommender(Recommender):
    def __init__(self, max_u: int, max_v: int,
                 factor_model: nn.Module,
                 expo_factor: Optional[nn.Module] = None,
                 expo_thresh: float = 0.05,
                 expo_compound: float = 1):
        self.max_u = max_u
        self.max_v = max_v
        self.factor_model = factor_model
        self.expo_factor = expo_factor
        self.expo_thresh = expo_thresh
        self.expo_compound = expo_compound
        self.logger = logging.getLogger(__name__)

    def fit(self,
            train_df: pd.DataFrame,
            test_df: Optional[pd.DataFrame] = None,
            rating_factor: Optional[nn.Module] = None,
            expo_model: Optional[Recommender] = None,
            past_hist: Optional[Dict[int, Set[int]]] = None,
            lr: float = 0.01,
            batch_size: int = 2048,
            num_neg: int = 1,
            num_epochs: int = 50,
            lambda_: float = 0.001,
            decay: float = 0.0,
            delta: float = 10,
            cuda: Optional[int] = None) -> None:

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        model = self.factor_model
        model.to(device)
        if self.expo_factor is not None:
            self.expo_factor.to(device)
            self.expo_factor.eval()
        u, v = train_df.uidx.tolist(), train_df.iidx.tolist()

        optimizer = build_optimizer(lr, model)

        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))
        hist = train_df.groupby('uidx').apply(
            lambda x: list(zip(x.ts, x.iidx))).to_dict()
        for k in hist.keys():
            hist[k] = [x[1] for x in sorted(hist[k])]
        seq_data = NegSequenceData(
            hist,
            1,
            item_num=self.max_v,
            padding_idx=self.max_v,
            num_neg=num_neg,
            window=True,
            past_hist=past_hist,
            allow_empty=True)
        data_loader = data.DataLoader(
            seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        for epoch in tqdm(range(num_epochs)):
            model.train()
            loss_record = []
            for user, item_i, item_j_list, item_hist in data_loader:
                optimizer.zero_grad()
                model.zero_grad()
                # transfer to gpu
                bsz = item_hist.shape[0]
                user = user.to(device).long()  # [B]
                item_i = item_i.to(device).long()  # [B]
                item_j_list = item_j_list.to(device).long()  # [B, num_neg]
                #item_hist = item_hist.to(device).long()  # [B, max_len]

                # reshape
                item_i_list = item_i.view(-1, 1).repeat(1, num_neg)  # [B, num_neg]
                users = user.unsqueeze(1).repeat(
                    1, num_neg)  # [B, num_neg]

                prediction_i = model(users, item_i_list)  # [B, num_neg]
                prediction_j = model(
                    users, item_j_list)  # [B, num_neg]
                g_loss = -(prediction_i - prediction_j).sigmoid().log()
                g_loss = g_loss.mean()
                l2_loss = decay * model.get_l2(users, item_i_list)
                l2_loss += decay * model.get_l2(users, item_j_list)
                target = g_loss + l2_loss
                target.backward()
                optimizer.step()
                loss_record.append(
                    (target.item(), g_loss.item(), l2_loss.item()))
            loss_np = np.array(loss_record)
            #self.logger.debug(
            #    f'target: {np.mean(loss_np[:, 0]):.5f},loss: {np.mean(loss_np[:, 1]):.5f}, l2: {np.mean(loss_np[:, 2]):.5f}')

            if test_df is not None:
                model.eval()
                rating_model = None
                if rating_factor is not None:
                    rating_model = ClassRecommender(
                        self.max_u, self.max_v, rating_factor)
                unbiased_eval(self.max_u, self.max_v, test_df, self,
                              rel_model=rating_model,
                              cut_len=10,
                              expo_model=expo_model,
                              past_hist=past_hist)

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        with torch.no_grad():
            device = self.factor_model.get_device()
            self.factor_model.eval()
            u_b_t = torch.LongTensor(u_b).to(device)  # type: ignore
            v_b_t = torch.LongTensor(v_b).to(device)  # type: ignore
            u_b_t.to(device)  # type: ignore
            v_b_t.to(device)  # type: ignore
            scores = self.factor_model(u_b_t, v_b_t)
            return scores.cpu().numpy()


class ClassRecommender(Recommender):
    def __init__(self, max_u: int, max_v: int,
                 factor_model: nn.Module,
                 expo_factor: Optional[nn.Module] = None,
                 expo_thresh: float = 0.05,
                 expo_compound: float = 1) -> None:

        self.max_u = max_u
        self.max_v = max_v
        self.factor_model = factor_model
        self.expo_factor = expo_factor
        self.expo_thresh = expo_thresh
        self.expo_compound = expo_compound
        self.logger = logging.getLogger(__name__)

    def fit(self,
            train_df: pd.DataFrame,
            test_df: Optional[pd.DataFrame] = None,
            rating_factor: Optional[nn.Module] = None,
            expo_model: Optional[Recommender] = None,
            past_hist: Optional[Dict[int, Set[int]]] = None,
            lr: float = 0.01,
            batch_size: int = 2048,
            num_neg: int = 1,
            num_epochs: int = 50,
            lambda_: float = 0.001,
            decay: float = 0.0,
            delta: float = 10,
            cuda: Optional[int] = None) -> None:

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        model = self.factor_model
        model.to(device)
        if self.expo_factor is not None:
            self.expo_factor.to(device)
            self.expo_factor.eval()
        #u, v = train_df.uidx.tolist(), train_df.iidx.tolist()
        optimizer = build_optimizer(lr, model)
        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        hist = train_df.groupby('uidx').apply(
            lambda x: list(zip(x.ts, x.iidx))).to_dict()
        for k in hist.keys():
            hist[k] = [x[1] for x in sorted(hist[k])]
        seq_data = NegSequenceData(
            hist,
            1,
            item_num=self.max_v,
            padding_idx=self.max_v,
            num_neg=num_neg,
            window=True,
            past_hist=past_hist,
            allow_empty=True)
        data_loader = data.DataLoader(
            seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        for epoch in tqdm(range(num_epochs)):
            model.train()
            loss_record = []
            for user, item_i, item_j_list, item_hist in data_loader:
                optimizer.zero_grad()
                model.zero_grad()
                # transfer to gpu
                bsz = item_hist.shape[0]
                user = user.to(device).long()  # [B]
                item_i = item_i.to(device).long()  # [B]
                item_j_list = item_j_list.to(device).long()  # [B, num_neg]
                #item_hist = item_hist.to(device).long()  # [B, max_len]

                # reshape
                item_i = item_i.view(-1, 1)  # [B, 1]
                items = torch.cat([item_i, item_j_list],
                                    dim=1)  # [B, 1 + num_neg]
                labels = (torch.arange(1 + num_neg).to(device)
                            < 1).float().repeat(bsz).view(bsz, -1)  # [B, 1 + num_neg]
                users = user.unsqueeze(1).repeat(
                    1, 1 + num_neg)  # [B, 1 + num_neg]
                
                g_s = model(users, items)
                g_prob = act_func(g_s)
                if self.expo_factor is not None:
                    expo_score = self.expo_factor(users, items)
                    expo_prob = act_func(expo_score) ** self.expo_compound
                    expo_prob = torch.clamp(expo_prob, min=self.expo_thresh)
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob)) / expo_prob
                else:
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob))
                g_loss = g_loss.mean()
                l2_loss = decay * model.get_l2(user, items)
                target = g_loss + l2_loss
                target.backward()
                optimizer.step()
                loss_record.append(
                    (target.item(), g_loss.item(), l2_loss.item()))
            loss_np = np.array(loss_record)
            #self.logger.debug(
            #    f'target: {np.mean(loss_np[:, 0]):.5f},loss: {np.mean(loss_np[:, 1]):.5f}, l2: {np.mean(loss_np[:, 2]):.5f}')

            if test_df is not None:
                model.eval()
                rating_model = None
                if rating_factor is not None:
                    rating_model = ClassRecommender(
                        self.max_u, self.max_v, rating_factor)
                unbiased_eval(self.max_u, self.max_v, test_df, self,
                              rel_model=rating_model,
                              cut_len=10,
                              expo_model=expo_model,
                              past_hist=past_hist)

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        with torch.no_grad():
            device = self.factor_model.get_device()
            self.factor_model.eval()
            u_b_t = torch.LongTensor(u_b).to(device)  # type: ignore
            v_b_t = torch.LongTensor(v_b).to(device)  # type: ignore
            u_b_t.to(device)  # type: ignore
            v_b_t.to(device)  # type: ignore
            scores = self.factor_model(u_b_t, v_b_t)
            return scores.cpu().numpy()


class RatingEstimator(Recommender):
    def __init__(self, max_u: int, max_v: int, factor_model: nn.Module):
        self.max_u = max_u
        self.max_v = max_v
        self.factor_model = factor_model

    def fit(self,
            features: List[Tuple[int, int, float]],
            lr: float = 0.01,
            batch_size: int = 2048,
            num_neg: int = 1,
            num_epochs: int = 50,
            lambda_: float = 0.001,
            decay: float = 0.0,
            cuda: Optional[int] = None) -> None:

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        rating_data = RatingData(features)
        train_loader = torch.utils.data.DataLoader(
            rating_data, batch_size=batch_size, shuffle=True, num_workers=2)

        model = self.factor_model
        model.to(device)
        # minimizer
        sp_minimizer = torch.optim.SparseAdam(
            params=model.get_sparse_weight(), lr=lr)
        ds_minimizer = torch.optim.Adam(params=model.get_dense_weight(), lr=lr)
        optimizer = MultipleOptimizer(sp_minimizer, ds_minimizer)
        loss_func = torch.nn.MSELoss()

        for epoch in tqdm(range(num_epochs)):
            model.train()
            loss_metric = []
            for user, item, rating in train_loader:
                optimizer.zero_grad()
                model.zero_grad()

                user = user.to(device).long()
                item = item.to(device).long()
                rating = rating.to(device).float()
                pred_rating = model(user, item)
                loss = loss_func(pred_rating, rating)
                l2_loss = decay * model.get_l2(user, item)
                target = loss + l2_loss
                target.backward()
                optimizer.step()
                loss_metric.append(loss.item())

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        with torch.no_grad():
            device = self.factor_model.embed_item.weight.device
            self.factor_model.eval()
            u_b_t = torch.LongTensor(u_b).to(device)  # type: ignore
            v_b_t = torch.LongTensor(v_b).to(device)  # type: ignore
            u_b_t.to(device)  # type: ignore
            v_b_t.to(device)  # type: ignore
            scores = self.factor_model(u_b_t, v_b_t)
            return scores.cpu().numpy()


class DeepRecommender(Recommender):
    def __init__(self, max_u: int, max_v: int,
                 seq_model: nn.Module,
                 expo_factor: Optional[nn.Module] = None,
                 expo_thresh: float = 0.05,
                 expo_compound: float = 1,
                 expo_isdeep:bool = False):
        self.max_u = max_u
        self.max_v = max_v
        self.seq_model = seq_model
        self.max_len = self.seq_model.max_len
        self.padding_idx = self.seq_model.padding_idx
        self.expo_factor = expo_factor
        self.expo_thresh = expo_thresh
        self.expo_compound = expo_compound
        self.logger = logging.getLogger(__name__)
        self.user_records = None
        self.expo_isdeep = expo_isdeep

    def set_user_record(self, user_record: Dict[int, List[int]]):
        self.user_records = user_record

    def fit(self,
            train_df: pd.DataFrame,
            test_df: Optional[pd.DataFrame] = None,
            rating_factor: Optional[nn.Module] = None,
            expo_model: Optional[Recommender] = None,
            past_hist: Optional[Dict[int, Set[int]]] = None,
            lr: float = 0.01,
            batch_size: int = 2048,
            num_neg: int = 1,
            num_epochs: int = 50,
            lambda_: float = 0.001,
            decay: float = 0.0,
            delta: float = 10,
            window: bool = True,
            cuda: Optional[int] = None) -> None:

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        model = self.seq_model
        model.to(device)
        if self.expo_factor is not None:
            self.expo_factor.to(device)
            self.expo_factor.eval()
        
        optimizer = build_optimizer(lr, model)

        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        hist = train_df.groupby('uidx').apply(
            lambda x: list(zip(x.ts, x.iidx))).to_dict()
        for k in hist.keys():
            hist[k] = [x[1] for x in sorted(hist[k])]

        self.set_user_record(hist)

        seq_data = NegSequenceData(
            hist, self.max_len,
            item_num=self.max_v,
            padding_idx=self.padding_idx,
            num_neg=num_neg,
            window=window,
            past_hist=past_hist)

        train_loader = data.DataLoader(
            seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        for epoch in tqdm(range(num_epochs)):
            model.train()
            loss_record = []
            for user, item_i, item_j_list, item_hist in train_loader:
                optimizer.zero_grad()
                model.zero_grad()
                bsz = item_hist.shape[0]
                user = user.to(device).long()
                item_i = item_i.to(device).long()
                item_j_list = item_j_list.to(device).long()
                item_hist = item_hist.to(device).long()
                item_i = item_i.view(-1, 1)  # [B, 1]
                items = torch.cat([item_i, item_j_list],
                                    dim=1)  # [B, 1 + num_neg]
                labels = (torch.arange(1 + num_neg).to(device)
                            < 1).float().repeat(bsz).view(bsz, -1)  # [B, 1 + num_neg]
                users = user.unsqueeze(1).repeat(
                    1, 1 + num_neg)  # [B, 1 + num_neg]
                
                g_s = model(items, item_hist)
                g_prob = act_func(g_s)

                if self.expo_factor is not None:
                    if self.expo_isdeep:
                        expo_score = self.expo_factor(items, item_hist)
                    else:
                        expo_score = self.expo_factor(users, items)
                    expo_prob = act_func(expo_score) ** self.expo_compound
                    expo_prob = torch.clamp(expo_prob, min=self.expo_thresh)
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob)) / expo_prob
                else:
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                (1 - labels) * torch.log(1 - g_prob))
                g_loss = g_loss.mean()
                l2_loss = decay * g_loss * 0  # model.get_l2(user, items)
                target = g_loss + l2_loss
                target.backward()
                optimizer.step()
                loss_record.append(
                    (target.item(), g_loss.item(), l2_loss.item()))
            loss_np = np.array(loss_record)
            #self.logger.debug(
            #    f'target: {np.mean(loss_np[:, 0]):.5f},loss: {np.mean(loss_np[:, 1]):.5f}, l2: {np.mean(loss_np[:, 2]):.5f}')

            if test_df is not None:
                model.eval()
                unbiased_eval(self.max_u, self.max_v, test_df, self,
                              rel_model=None,
                              cut_len=10,
                              expo_model=None,
                              past_hist=past_hist)

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        assert(self.user_records is not None)
        temp_hist = np.zeros(self.max_len, dtype=int) + self.padding_idx
        item_hist = self.user_records[u_b[0]]
        if len(item_hist) == 0:
            return np.zeros(len(v_b))

        temp_hist[-len(item_hist):] = item_hist[-self.max_len:]
        temp_hist = temp_hist.reshape(1, -1)
        with torch.no_grad():
            device = self.seq_model.get_device()
            self.seq_model.eval()
            v_b_t = torch.LongTensor(v_b).to(device)  # [num_item]
            v_b_t = v_b_t.view(1, -1)  # [1, num_item]
            temp_hist = torch.from_numpy(temp_hist).to(device)  # [1, max_len]
            scores = self.seq_model(v_b_t, temp_hist).flatten()
            return scores.cpu().numpy()


def unbiased_eval(num_user: int, num_item: int, dat_df: pd.DataFrame,
                  recom: Recommender, rel_model: Optional[Recommender] = None,
                  expo_model: Optional[Recommender] = None,
                  past_hist: Optional[Dict[int, Set[int]]] = None, expo_compound: float = 1.0,
                  epsilon: float = 1.0, num_neg: int = 100, cut_len: int = 10, seed: int = 886):
    logger = logging.getLogger(__name__)
    # this is to make sure comparision between models is fair yet not affect the negative sampling's variation
    prng = RandomState(seed)

    row, col = dat_df.uidx, dat_df.iidx
    def sigmoid(x): return np.exp(x) / (1 + np.exp(x))
    recall_cnt = 0
    ndcg_sum = 0
    for u, i in list(zip(row, col)):
        if past_hist is None:
            neg = prng.randint(0, num_item, num_neg)
            neg = neg[neg != i]
        else:
            neg = prng.randint(0, num_item, num_neg)
            for idx in range(num_neg):
                if int(neg[idx]) in past_hist.get(u, []) or i == neg[idx]:
                    while int(
                            neg[idx]) not in past_hist.get(
                            u, []) and i != neg[idx]:
                        neg[idx] = prng.randint(0, num_item)
        item_list: List[int] = neg.tolist()
        item_list.append(i)
        user_list = [u] * len(item_list)
        scores = recom.score(user_list, item_list)
        if rel_model is not None:
            rel_score = rel_model.score(user_list, item_list)
            rel_prob = sigmoid(rel_score - epsilon)
        else:
            rel_prob = np.ones(len(scores))

        expo_score = 1
        if expo_model is not None:
            expo_score = sigmoid(expo_model.score([u], [i])[0]) ** expo_compound

        rank = scores.argsort()[::-1]
        item_npy = np.array(item_list)
        top_items = item_npy[rank][:cut_len]
        top_item_rel_prob = rel_prob[rank][:cut_len]
        #recall_cnt += int(i in top_items)
        for pos, (top_i, top_rel) in enumerate(
                zip(top_items, top_item_rel_prob)):
            if i == top_i:
                recall_cnt += (top_rel / expo_score)
                ndcg_sum += np.log(2) / np.log(2 + pos) * \
                    (top_rel / expo_score)
    logger.info(
        f'Recall@{cut_len} = {recall_cnt / len(row):.5f}; NDCG@{cut_len} = {ndcg_sum / len(row):.5f}')
    return recall_cnt / len(row)


def ac_train_v2(f_model: torch.nn.Module,
                g_model: torch.nn.Module,
                beta_model: torch.nn.Module,
                tr_df: pd.DataFrame,
                user_num: int,
                item_num: int,
                val_df: Optional[pd.DataFrame] = None,
                rating_model: Optional[Recommender] = None,
                expo_model: Optional[Recommender] = None,
                past_hist: Optional[Dict[int, Set[int]]] = None,
                num_epochs: int = 50,
                batch_size: int = 2048,
                min_prob: float = 0.1,
                num_neg: int = 1,
                cuda_idx: int = 0,
                min_delta: float = 0.1,
                lr: float = 0.01,
                f_round_ahead: int = 1,
                g_round_ahead: int = 1,
                decay: float = 0.0):
    logger = logging.getLogger(__name__)
    with torch.cuda.device(cuda_idx):

        f_recommender = ClassRecommender(user_num, item_num, f_model)
        g_recommender = ClassRecommender(user_num, item_num, g_model)

        u, v = tr_df.uidx.tolist(), tr_df.iidx.tolist()

        minimizer = build_optimizer(lr, f_model, beta_model)
        maximizer = build_optimizer(lr, g_model)

        loss_func = torch.nn.BCELoss(reduction='none')
        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        #device_cuda = torch.device(f'cuda:{cuda_idx}')
        f_model.cuda()
        g_model.cuda()
        beta_model.cuda()

        def train_epoch(optimizer, data_loader, flag='g_train'):
            f_loss_record, g_loss_record = [], []
            # train the g_model for one epoch
            for c_round in range(g_round_ahead):
                for user, item_pos, item_neg_list in data_loader:

                    f_model.zero_grad()
                    g_model.zero_grad()
                    beta_model.zero_grad()
                    optimizer.zero_grad()

                    f_model.train()
                    g_model.train()
                    beta_model.train()

                    user = user.long().cuda()
                    item_pos = item_pos.long().cuda()
                    item_neg_list = item_neg_list.cuda().long()
                    item_neg = item_neg_list.flatten()

                    user_for_neg = user.reshape(
                        1, -1).repeat(num_neg, 1).t().flatten()
                    user = torch.cat([user, user_for_neg], dim=0).long()
                    items = torch.cat([item_pos, item_neg], dim=0).long()
                    labels = torch.cat([torch.ones(len(item_pos)).cuda(
                    ), torch.zeros(len(item_neg)).cuda()], dim=0).float()

                    f_s = f_model(user, items)
                    g_s = g_model(user, items)
                    q_s = beta_model(user, items, g_s, labels)

                    f_prob = torch.clamp(act_func(f_s), min=0.01, max=1)
                    g_prob = torch.clamp(act_func(g_s), min=0.01, max=1)
                    q_prob = torch.clamp(act_func(q_s), min=min_prob, max=1)

                    f_loss = -1 * (labels * torch.log(f_prob) +
                                   (1 - labels) * torch.log(1 - f_prob)) / q_prob
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob))

                    if flag == 'g_train':
                        target = (
                            torch.clamp(
                                min_delta + g_loss - f_loss,
                                min=0)).mean()  # g wants to maximize the gap
                        target += decay * g_model.get_l2(user, items)
                        target.backward()
                    elif flag == 'f_train':
                        target = f_loss.mean()
                        target += decay * \
                            f_model.get_l2(user, items) + decay * \
                            beta_model.get_l2(user, items)
                        target.backward()
                    else:
                        raise ValueError('use g_train or f_train')
                    optimizer.step()

                    with torch.no_grad():
                        f_loss = f_loss.mean()
                        g_loss = g_loss.mean()
                        f_loss_record.append(f_loss.item())
                        g_loss_record.append(g_loss.item())

                logger.info(
                    f'{flag} at {c_round} round -- f_loss: {np.mean(f_loss_record)} g_loss: {np.mean(g_loss_record)}')

        # pre-fit the g without adjusting
        g_recommender.fit(tr_df,
                          num_epochs=0,
                          cuda=cuda_idx,
                          decay=decay)

        neg_data = NegSeqData(list(zip(u, v)), item_num,
                              num_neg=num_neg, past_hist=past_hist)
        neg_data.is_training = True
        for epoch in range(num_epochs):
            neg_data.ng_sample()
            data_loader = data.DataLoader(
                neg_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True)

            logger.info(f'Epoch -- {epoch}')
            minimizer.zero_grad()
            maximizer.zero_grad()

            train_epoch(minimizer, data_loader, 'f_train')
            train_epoch(maximizer, data_loader, 'g_train')

            if val_df is not None:
                logger.info('f_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    f_recommender,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist)

                logger.info('g_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    g_recommender,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist)


def ac_train_v3(f_model: torch.nn.Module,
                is_f_seq: bool,
                g_model: torch.nn.Module,
                is_g_seq: bool,
                beta_model: torch.nn.Module,
                tr_df: pd.DataFrame,
                user_num: int,
                item_num: int,
                val_df: Optional[pd.DataFrame] = None,
                rating_model: Optional[Recommender] = None,
                expo_model: Optional[Recommender] = None,
                past_hist: Optional[Dict[int, Set[int]]] = None,
                g_weight: float = 1.0,
                num_epochs: int = 50,
                batch_size: int = 2048,
                min_prob: float = 0.1,
                num_neg: int = 1,
                cuda_idx: int = 0,
                min_delta: float = 0.1,
                lr: float = 0.01,
                decay: float = 0.0,
                expo_compound: float = 1.0,
                epsilon: float = 1.0):
    logger = logging.getLogger(__name__)
    with torch.cuda.device(cuda_idx):

        if is_f_seq:
            f_recommender = DeepRecommender(user_num, item_num, f_model)
        else:
            f_recommender = ClassRecommender(user_num, item_num, f_model)

        if is_g_seq:
            g_recommender = DeepRecommender(user_num, item_num, g_model)
        else:
            g_recommender = ClassRecommender(user_num, item_num, g_model)

        minimizer = build_optimizer(lr, f_model, beta_model)
        maximizer = build_optimizer(lr, g_model)

        loss_func = torch.nn.BCELoss(reduction='none')
        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        #device_cuda = torch.device(f'cuda:{cuda_idx}')
        f_model.cuda()
        g_model.cuda()
        beta_model.cuda()

        def train_epoch(optimizer, data_loader, flag, is_f_seq, is_g_seq, round_repeat=1):
            f_loss_record, g_loss_record = [], []
            q_prob_record = []
            # train the g_model for one epoch
            for c_round in range(round_repeat):
                for user, item_i, item_j_list, item_hist in data_loader:

                    f_model.zero_grad()
                    g_model.zero_grad()
                    beta_model.zero_grad()
                    optimizer.zero_grad()

                    f_model.train()
                    g_model.train()
                    beta_model.train()

                    # transfer to gpu
                    bsz = item_hist.shape[0]
                    user = user.cuda().long()  # [B]
                    item_i = item_i.cuda().long()  # [B]
                    item_j_list = item_j_list.cuda().long()  # [B, num_neg]
                    item_hist = item_hist.cuda().long()  # [B, max_len]

                    # reshape
                    item_i = item_i.view(-1, 1)  # [B, 1]
                    items = torch.cat([item_i, item_j_list],
                                      dim=1)  # [B, 1 + num_neg]
                    labels = (torch.arange(1 + num_neg).cuda()
                              < 1).float().repeat(bsz).view(bsz, -1)  # [B, 1 + num_neg]
                    users = user.unsqueeze(1).repeat(
                        1, 1 + num_neg)  # [B, 1 + num_neg]

                    f_s = f_model(items, item_hist) if is_f_seq else f_model(
                        users, items)
                    g_s = g_model(items, item_hist) if is_g_seq else g_model(
                        users, items)
                    q_s = beta_model(users, items, g_s, labels)

                    f_prob = torch.clamp(act_func(f_s), min=0.01, max=1)
                    g_prob = torch.clamp(act_func(g_s), min=0.01, max=1)
                    q_prob = torch.clamp(act_func(q_s), min=min_prob, max=1)

                    f_loss = -1 * (labels * torch.log(f_prob) +
                                   (1 - labels) * torch.log(1 - f_prob)) / q_prob
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob))

                    if flag == 'g_train':
                        target = (
                            torch.clamp(
                                min_delta + g_weight * g_loss - f_loss,
                                min=0)).mean()  # g wants to maximize the gap
                        target += decay * g_model.get_l2(user, items)
                        target.backward()
                    elif flag == 'f_train':
                        target = f_loss.mean()
                        target += decay * \
                            f_model.get_l2(user, items) + decay * \
                            beta_model.get_l2(user, items)
                        target.backward()
                    else:
                        raise ValueError('use g_train or f_train')
                    optimizer.step()

                    with torch.no_grad():
                        f_loss = f_loss.mean()
                        g_loss = g_loss.mean()
                        f_loss_record.append(f_loss.item())
                        g_loss_record.append(g_loss.item())
                        q_prob_record.append(q_prob.mean().item())

                logger.info(
                    f'{flag} at {c_round} round -- f_loss: {np.mean(f_loss_record)} g_loss: {np.mean(g_loss_record)}, q_prob: {np.mean(q_prob_record)}')

        hist = tr_df.groupby('uidx').apply(
            lambda x: list(zip(x.ts, x.iidx))).to_dict()
        for k in hist.keys():
            hist[k] = [x[1] for x in sorted(hist[k])]
        if is_f_seq:
            f_recommender.set_user_record(hist)
        if is_g_seq:
            g_recommender.set_user_record(hist)

        padding_idx = item_num + 1
        max_len = 1
        if is_f_seq:
            max_len = f_model.max_len
        elif is_g_seq:
            max_len = g_model.max_len
        f_seq_data = NegSequenceData(
            hist,
            max_len,
            item_num=item_num,
            padding_idx=padding_idx,
            num_neg=num_neg,
            window=True,
            past_hist=past_hist,
            allow_empty=not is_f_seq)

        f_train_loader = data.DataLoader(
            f_seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        g_seq_data = NegSequenceData(
            hist,
            max_len,
            item_num=item_num,
            padding_idx=padding_idx,
            num_neg=num_neg,
            window=True,
            past_hist=past_hist,
            allow_empty=not is_g_seq)

        g_train_loader = data.DataLoader(
            g_seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        for epoch in range(num_epochs):
            logger.info(f'Epoch -- {epoch}')
            minimizer.zero_grad()
            maximizer.zero_grad()

            train_epoch(minimizer, f_train_loader,
                        'f_train', is_f_seq, is_g_seq)
            train_epoch(maximizer, g_train_loader,
                        'g_train', is_f_seq, is_g_seq)

            logger.info(f'beta_model: {beta_model.alpha.item()}, {beta_model.beta.item()}, {beta_model.label_coef.item()}')
            if val_df is not None:
                logger.info('f_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    f_recommender,
                    epsilon=epsilon,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist,
                    expo_compound=expo_compound)

                logger.info('g_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    g_recommender,
                    epsilon=epsilon,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist,
                    expo_compound=expo_compound)
