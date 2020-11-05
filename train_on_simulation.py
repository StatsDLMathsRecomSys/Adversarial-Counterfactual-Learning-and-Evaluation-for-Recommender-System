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

    ratings = pd.read_feather(os.path.join(args.data_path, args.data_name + '_smaple'))
    user_num, item_num = ratings.uidx.max() + 1, ratings.iidx.max() + 1

    #df = pd.read_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_full.feather'))
    tr_df = pd.read_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_train.feather'))
    val_df = pd.read_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_val.feather'))
    te_df = pd.read_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_test.feather'))

    if args.tune_mode:
        tr_df = pd.concate([tr_df, val_df])
        te_df = te_df
    else:
        tr_df = tr_df
        te_df = val_df

    past_hist = tr_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
    item_cnt_dict = tr_df.groupby('iidx').count().uidx.to_dict()
    item_cnt = np.array([item_cnt_dict.get(iidx, 0) for iidx in range(item_num)])

    logger.info(f'test data size: {te_df.shape}')

    dim=args.dim

    rel_factor = FactorModel(user_num, item_num, dim)
    PATH = os.path.join(args.sim_path, f'{args.prefix}_rel.pt')
    rel_factor.load_state_dict(torch.load(PATH))
    rel_factor.eval()

    train_expo_factor = FactorModel(user_num, item_num, dim)
    PATH = os.path.join(args.sim_path, f'{args.prefix}_expo.pt')
    train_expo_factor.load_state_dict(torch.load(PATH))
    train_expo_factor.eval()

    train_expo_factor = NoiseFactor(train_expo_factor, args.dim)
    train_expo_factor = train_expo_factor.to(torch.device(f'cuda:{args.cuda_idx}'))
    train_expo_factor.load_state_dict(torch.load(os.path.join(args.sim_path, f'{args.prefix}_expo_noise.pt')))
    train_expo_factor.eval()

    expo_factor = FactorModel(user_num, item_num, dim)
    PATH = os.path.join(args.sim_path, f'{args.prefix}_expo_bs.pt')
    expo_factor.load_state_dict(torch.load(PATH))
    expo_factor.eval()

    rating_model = RatingEstimator(user_num, item_num, rel_factor)
    expo_model = ClassRecommender(user_num, item_num, expo_factor)
    tr_mat = frame2mat(tr_df, user_num, item_num)
    val_mat = frame2mat(val_df, user_num, item_num)

    choices = args.models
    logging.info(f'Running {choices}')
    def get_model(model_str, user_num, item_num, factor_num):
        if model_str == 'mlp':
            return MLPRecModel(user_num, item_num, factor_num)
        elif model_str == 'gmf':
            return FactorModel(user_num, item_num, factor_num)
        elif model_str == 'ncf':
            return NCFModel(user_num, item_num, factor_num)
        else:
            raise NotImplementedError(f'{model_str} is not implemented')

    logging.info('-------The Popularity model-------')
    pop_factor = PopularModel(item_cnt)
    pop_model = PopRecommender(pop_factor)
    logger.info('unbiased eval for plian popular model on test')
    unbiased_eval(user_num, item_num, te_df, pop_model, epsilon=args.epsilon,
    rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)

    logger.info('-------The SVD model---------')
    sv = SVDRecommender(tr_mat.shape[0], tr_mat.shape[1], dim)
    logger.info(f'model with dimension {dim}')
    sv.fit(tr_mat)
    logger.info('un-biased eval for SVD model on test')
    unbiased_eval(user_num, item_num, te_df, sv, epsilon=args.epsilon,
    rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)


    def complete_experiment(model_str, user_num, item_num, dim):
        logging.info(f'-------The {model_str} model-------')
        base_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        base_model =ClassRecommender(user_num, item_num, base_factor)
        base_model.fit(tr_df,
                        num_epochs=args.epoch,
                        cuda=args.cuda_idx,
                        decay=1e-8, 
                        num_neg=args.num_neg,
                        past_hist=past_hist,
                        lr=args.lr)
        logger.info(f'unbiased eval for {model_str}  model on test')
        unbiased_eval(user_num, item_num, te_df, base_model, epsilon=args.epsilon,
        rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)

        logging.info(f'-------The {model_str} Pop Adjust model-------')
        pop_adjust_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        pop_adjust_model = ClassRecommender(user_num, item_num, pop_adjust_factor, pop_factor, expo_thresh=0.1)
        pop_adjust_model.fit(tr_df,
                        num_epochs=args.epoch,
                        cuda=args.cuda_idx,
                        decay=args.decay,
                        num_neg=args.num_neg,
                        past_hist=past_hist,
                        lr=args.lr)
        logger.info(f'unbiased eval for adjust {model_str} with popular model on test')
        unbiased_eval(user_num, item_num, te_df, pop_adjust_model, epsilon=args.epsilon,
        rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)
        del pop_adjust_factor


        logging.info(f'-------The {model_str} Mirror Adjust model-------')
        adjust_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        adjust_model = ClassRecommender(user_num, item_num, adjust_factor, base_factor, expo_thresh=0.1)
        adjust_model.fit(tr_df,
                        num_epochs=args.epoch, 
                        cuda=args.cuda_idx,
                        num_neg=args.num_neg,
                        past_hist=past_hist,
                        decay=args.decay,
                        lr=args.lr)

        logger.info(f'un-biased eval for {model_str} mirror adjusted model')
        unbiased_eval(user_num, item_num, te_df, adjust_model, epsilon=args.epsilon,
        rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)
        del adjust_factor

        logger.info(f'-------The {model_str} Oracle Adjust model---------')
        oracle_factor = get_model(model_str, user_num=user_num, item_num=item_num, factor_num=dim)
        oracle_model = ClassRecommender(user_num, 
            item_num, oracle_factor, train_expo_factor, expo_thresh=0.1, expo_compound=args.p)

        oracle_model.fit(tr_df,
                        num_epochs=args.epoch, 
                        cuda=args.cuda_idx,
                        num_neg=args.num_neg,
                        past_hist=past_hist,
                        decay=args.decay,
                        lr=args.lr)

        logger.info('un-biased eval for oracle model on test')
        unbiased_eval(user_num, item_num, te_df, oracle_model, epsilon=args.epsilon,
        rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)
        del oracle_factor

    for model_str in choices:
        if model_str != 'acgan':
            complete_experiment(model_str, user_num, item_num, dim)


    if 'acgan' in choices:
        logger.info('-------The AC GAN model---------')
        f = get_model(args.f_model, user_num, item_num, dim)
        g = get_model(args.g_model, user_num, item_num, dim)
        beta = BetaModel(user_num=user_num, item_num=item_num)
        f_recommender = ClassRecommender(user_num, item_num, f)
        g_recommender = ClassRecommender(user_num, item_num, g)
        g_recommender.fit(tr_df,
                        num_epochs=args.g_round_head, 
                        cuda=args.cuda_idx,
                        num_neg=args.num_neg,
                        past_hist=past_hist,
                        decay=args.decay,
                        lr=args.lr)
        ac_train_v3(f, False, g, False, beta, tr_df,
                    user_num=user_num,
                    item_num=item_num, 
                    num_neg=args.num_neg, 
                    past_hist=past_hist,
                    val_df=te_df, 
                    rating_model=rating_model, 
                    expo_model=expo_model,
                    num_epochs=args.epoch, 
                    decay=args.decay, 
                    cuda_idx=args.cuda_idx,
                    lr=args.lr,
                    g_weight=0.5,
                    expo_compound=args.p,
                    epsilon=args.epsilon)

        logger.info(f'eval on test with f_model ({args.f_model})')
        unbiased_eval(user_num, item_num, te_df, f_recommender, epsilon=args.epsilon,
        rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)
        logger.info(f'eval on test with g_model ({args.g_model})')
        unbiased_eval(user_num, item_num, te_df, g_recommender, epsilon=args.epsilon,
        rel_model=rating_model, past_hist=past_hist, expo_model=expo_model, expo_compound=args.p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=4)
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--epoch', type=float, default=10)
    parser.add_argument('--decay', type=float, default=1e-7)
    parser.add_argument('--sim_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='ratings.feather')
    parser.add_argument('--prefix', type=str, default='ml_1m_mf')
    parser.add_argument('--tune_mode', action='store_true')
    parser.add_argument('--num_neg', type=str, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--models',
                                default=['ncf', 'mlp', 'gmf', 'acgan'],
                                nargs='+',
                                help = "input a list of ['ncf', 'mlp', 'gmf', 'acgan']")
    parser.add_argument('--f_model', type=str, default='mlp')
    parser.add_argument('--g_model', type=str, default='mlp')
    parser.add_argument('--g_round_head', type=int, default=5)
    args = parser.parse_args()


    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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