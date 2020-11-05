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

def frame2mat(df, num_u, num_i):
    row, col = df.uidx, df.iidx
    data = np.ones(len(row))
    mat = sp.csr_matrix((data, (row, col)), shape=(num_u, num_i))
    return mat

def main(args: Namespace):

    ratings = pd.read_feather(os.path.join(args.data_path, args.data_name))
    user_num, item_num = ratings.uidx.max() + 1, ratings.iidx.max() + 1

    tr_df = pd.read_feather(os.path.join(args.data_path, 'train.feather'))
    val_df = pd.read_feather(os.path.join(args.data_path, 'val.feather'))
    te_df = pd.read_feather(os.path.join(args.data_path, 'test.feather'))

    if not args.tune_mode:
        tr_df = pd.concat([tr_df, val_df])
        te_df = te_df
    else:
        tr_df = tr_df
        te_df = val_df

    past_hist = tr_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
    item_cnt_dict = tr_df.groupby('iidx').count().uidx.to_dict()
    item_cnt = np.array([item_cnt_dict.get(iidx, 0) for iidx in range(item_num)])
    hist = tr_df.groupby('uidx').apply(
        lambda x: list(zip(x.ts, x.iidx))).to_dict()
    for k in hist.keys():
        hist[k] = [x[1] for x in sorted(hist[k])]

    logger.info(f'test data size: {te_df.shape}')

    rating_model = None
    tr_mat = frame2mat(tr_df, user_num, item_num)
    choices = args.models
    logging.info(f'Running {choices}')

    acgan_config = [args.f_model == 'seq', args.g_model == 'seq']

    pop_factor = PopularModel(item_cnt)
    logging.info('-------The Popularity model-------')
    pop_model = PopRecommender(pop_factor)
    logger.info('biased eval for plian popular model on test')
    unbiased_eval(user_num, item_num, te_df, pop_model, past_hist=past_hist)

    logger.info('-------The SVD model---------')
    sv = SVDRecommender(tr_mat.shape[0], tr_mat.shape[1], args.dim)
    logger.info(f'model with dimension {args.dim}')
    sv.fit(tr_mat)
    logger.info('biased eval for SVD model on test')
    unbiased_eval(user_num, item_num, te_df, sv, past_hist=past_hist)
    #unbiased_eval(user_num, item_num, te_df, sv)

    def get_model(model_str, user_num, item_num, factor_num, max_len=50, num_layer=2):
        if model_str == 'mlp':
            return MLPRecModel(user_num, item_num, factor_num)
        elif model_str == 'gmf':
            return FactorModel(user_num, item_num, factor_num)
        elif model_str == 'ncf':
            return NCFModel(user_num, item_num, factor_num)
        elif model_str == 'seq':
            return AttentionModel(user_num, item_num, args.dim, max_len=max_len, num_layer=num_layer)
        else:
            raise NotImplementedError(f'{model_str} is not implemented')

    def complete_experiment(model_str, user_num, item_num, dim, is_deep):
        logging.info(f'-------The {model_str} model-------')
        base_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        if is_deep:
            base_model = DeepRecommender(user_num, item_num, base_factor)
        else:
            base_model = ClassRecommender(user_num, item_num, base_factor)
        base_model.fit(tr_df, test_df=te_df,
                        num_epochs=args.epoch,
                        cuda=args.cuda_idx,
                        decay=args.decay, 
                        num_neg=args.num_neg,
                        batch_size=args.batch_size,
                        past_hist=past_hist,
                        lr=args.lr)
        logger.info(f'eval for {model_str}  model on test')
        unbiased_eval(user_num, item_num, te_df, base_model, past_hist=past_hist)

        logging.info(f'-------The {model_str} Pop Adjust model-------')
        pop_adjust_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        if is_deep:
            pop_adjust_model = DeepRecommender(user_num, item_num, pop_adjust_factor, pop_factor, expo_thresh=0.1)
        else:
            pop_adjust_model = ClassRecommender(user_num, item_num, pop_adjust_factor, pop_factor, expo_thresh=0.1)
        pop_adjust_model.fit(tr_df, test_df=te_df,
                        num_epochs=args.epoch,
                        cuda=args.cuda_idx,
                        decay=args.decay,
                        num_neg=args.num_neg,
                        batch_size=args.batch_size,
                        past_hist=past_hist,
                        lr=args.lr)
        logger.info(f'eval for adjust {model_str} with popular model on test')
        unbiased_eval(user_num, item_num, te_df, pop_adjust_model, past_hist=past_hist)
        del pop_adjust_factor


        logging.info(f'-------The {model_str} Mirror Adjust model-------')
        adjust_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        if is_deep:
            adjust_model = DeepRecommender(user_num, item_num, adjust_factor, base_factor, expo_thresh=0.1, expo_isdeep=True)
        else:
            adjust_model = ClassRecommender(user_num, item_num, adjust_factor, base_factor, expo_thresh=0.1)
        adjust_model.fit(tr_df, test_df=te_df,
                        num_epochs=args.epoch, 
                        cuda=args.cuda_idx,
                        num_neg=args.num_neg,
                        batch_size=args.batch_size,
                        past_hist=past_hist,
                        decay=args.decay,
                        lr=args.lr)

        logger.info(f'eval for {model_str} mirror adjusted model')
        unbiased_eval(user_num, item_num, te_df, adjust_model, past_hist=past_hist)
        del adjust_factor

    for model_str in choices:
        if model_str != 'acgan':
            complete_experiment(model_str, user_num, item_num, args.dim, model_str == 'seq')

    if 'acgan' in choices:
        logger.info(f'-------The AC GAN model with {args.f_model} / {args.g_model}---------')
        if acgan_config[0]:
            f = AttentionModel(user_num=user_num, item_num=item_num, factor_num=args.dim, max_len=50, num_layer=2)
            f_recommender = DeepRecommender(max_u=user_num, max_v=item_num, seq_model=f)
            f_recommender.set_user_record(hist)
        else:
            f = get_model(args.f_model, user_num=user_num, item_num=item_num, factor_num=args.dim)
            f_recommender = ClassRecommender(user_num, item_num, f)

        if acgan_config[1]:
            g = AttentionModel(user_num=user_num, item_num=item_num, factor_num=args.dim, max_len=50, num_layer=2)
            g_recommender = DeepRecommender(max_u=user_num, max_v=item_num, seq_model=g)
            g_recommender.set_user_record(hist)
        else:
            g = get_model(args.g_model, user_num=user_num, item_num=item_num, factor_num=args.dim)
            g_recommender = ClassRecommender(user_num, item_num, g)
        beta = BetaModel(user_num=user_num, item_num=item_num)

        g_recommender.fit(tr_df,
                        num_epochs=args.g_round_head, 
                        cuda=args.cuda_idx,
                        num_neg=args.num_neg,
                        batch_size=args.batch_size,
                        past_hist=past_hist,
                        decay=args.decay,
                        lr=args.lr)
        ac_train_v3(f, acgan_config[0], g, acgan_config[1], beta, tr_df,
                    user_num=user_num, 
                    item_num=item_num, 
                    val_df=te_df, 
                    rating_model=rating_model, 
                    num_epochs=args.epoch,
                    decay=args.decay, 
                    cuda_idx=args.cuda_idx, 
                    num_neg=args.num_neg,
                    batch_size=args.batch_size,
                    past_hist=past_hist,
                    g_weight=0.5,
                    lr=args.lr)

        logger.info(f'--final eval for AC GAN {args.f_model} / {args.g_model}--')
        unbiased_eval(user_num, item_num, te_df, f_recommender, past_hist=past_hist)
        unbiased_eval(user_num, item_num, te_df, g_recommender, past_hist=past_hist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--decay', type=float, default=1e-7)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='ratings.feather')
    parser.add_argument('--prefix', type=str, default='ml_1m_real')
    parser.add_argument('--num_neg', type=str, default=4)
    parser.add_argument('--tune_mode', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--models',
                                default=['ncf', 'mlp', 'gmf', 'acgan', 'seq'],
                                nargs='+',
                                help = "input a list from ['ncf', 'mlp', 'gmf', 'acgan', 'seq']")
    parser.add_argument('--f_model', type=str, default='mlp', choices=['ncf', 'mlp', 'gmf', 'seq'])
    parser.add_argument('--g_model', type=str, default='mlp', choices=['ncf', 'mlp', 'gmf', 'seq'])
    parser.add_argument('--g_round_head', type=int, default=5)

    args = parser.parse_args()

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'log/{args.prefix}-{str(time.time())}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    main(args)