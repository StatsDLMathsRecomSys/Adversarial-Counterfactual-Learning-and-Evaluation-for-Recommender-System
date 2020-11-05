"""Script to generate recommendation data from simulation"""
import argparse
from argparse import Namespace
import os

import pandas as pd #type: ignore
import torch #type: ignore
import numpy as np #type: ignore
from scipy import sparse as sp #type: ignore
from tqdm import tqdm #type: ignore
from acgan.data import RatingData
from acgan.module import FactorModel, NoiseFactor
from acgan.recommender import ClassRecommender, RatingEstimator, BPRRecommender
from sklearn.model_selection import train_test_split

torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args: Namespace):
    ratings = pd.read_feather(os.path.join(args.data_path, args.data_name))
    u_limit, i_limit = args.u_limit, args.i_limit
    ratings = ratings[(ratings.uidx < u_limit) & (ratings.iidx < i_limit)]
    ratings.reset_index(inplace=True)
    ratings.to_feather(os.path.join(args.data_path, args.data_name + '_smaple'))
    u_num, i_num = ratings.uidx.max() + 1, ratings.iidx.max() + 1

    print(f'u: {u_num}, i: {i_num}')
    #
    print('train rel model')
    rel_factor = FactorModel(u_num, i_num, args.dim)
    rating_features = list(zip(ratings.uidx, ratings.iidx, ratings.rating))
    rating_model = RatingEstimator(u_num, i_num, rel_factor)
    rating_model.fit(rating_features, cuda=0, num_epochs=args.epoch)

    #
    print('train expo model')
    expo_factor = FactorModel(u_num, i_num, args.dim)
    #expo_model = BPRRecommender(u_num, i_num, expo_factor)
    expo_model = ClassRecommender(u_num, i_num, expo_factor)
    full_mat = sp.csr_matrix((ratings.rating, (ratings.uidx, ratings.iidx)), shape=(u_num, i_num))
    print(full_mat.shape)
    expo_model.fit(ratings, cuda=0, num_epochs=args.epoch, decay=args.decay)

    torch.save(rel_factor.state_dict(), os.path.join(args.sim_path, f'{args.prefix}_rel.pt'))
    torch.save(expo_factor.state_dict(), os.path.join(args.sim_path, f'{args.prefix}_expo.pt'))

    print('get noise added expo model')
    expo_factor = NoiseFactor(expo_factor, args.dim, noise_ratio=args.noise_ratio)
    expo_factor = expo_factor.cuda()
    torch.save(expo_factor.state_dict(), os.path.join(args.sim_path, f'{args.prefix}_expo_noise.pt'))
    # re-assign the expo model
    expo_model = ClassRecommender(u_num, i_num, expo_factor)
    sigmoid = lambda x: np.exp(x) / (1 + np.exp(x))

    u_all = np.arange(u_num).repeat(i_num)
    i_all = np.arange(i_num).repeat(u_num).reshape(i_num, u_num).reshape(-1, order='F')
    est_rel = rating_model.score(u_all, i_all)
    est_click_prob = sigmoid(est_rel - args.epsilon)
    est_logits = expo_model.score(u_all, i_all)
    est_expo_prob = sigmoid(est_logits) ** args.p
    simu_size = len(est_click_prob)
    click_event = np.random.random(simu_size) < est_click_prob
    expo_event = np.random.random(simu_size) < est_expo_prob
    valid = click_event * expo_event
    train_valid = valid

    print(f'total size: {len(valid)}, valid size: {valid.sum()}')
    out = {}
    out['uidx'] = u_all[valid]
    out['iidx'] = i_all[valid]
    out['click_prob'] = est_click_prob[valid]
    out['expo_prob'] = est_expo_prob[valid]
    # placeholder variable to train the testing exposure model
    out['rating'] = np.ones(out['click_prob'].size)
    out['ts'] = np.random.rand(out['click_prob'].size)
    train_df = pd.DataFrame(out)

    new_expo_factor = FactorModel(u_num, i_num, args.dim).cuda()
    new_expo_model = ClassRecommender(u_num, i_num, new_expo_factor)
    new_expo_model.fit(train_df, cuda=0, num_epochs=args.epoch, decay=args.decay)
    torch.save(new_expo_factor.state_dict(), os.path.join(args.sim_path, f'{args.prefix}_expo_bs.pt'))

    est_rel = rating_model.score(u_all, i_all)
    est_click_prob = sigmoid(est_rel - args.epsilon)
    est_logits = new_expo_model.score(u_all, i_all)
    expo_prob = sigmoid(est_logits) ** args.p
    simu_size = len(est_click_prob)
    click_event = np.random.random(simu_size) < est_click_prob
    expo_event = np.random.random(simu_size) < est_expo_prob
    valid = click_event * expo_event * (~train_valid)

    robu_out = {}
    robu_out['uidx'] = u_all[valid]
    robu_out['iidx'] = i_all[valid]
    robu_out['click_prob'] = est_click_prob[valid]
    robu_out['expo_prob'] = est_expo_prob[valid]
    print(valid.sum())
    size = valid.sum()
    # placeholder variable to train the testing exposure model
    robu_out['rating'] = np.ones(size)
    robu_out['ts'] = np.random.rand(size)
    robu_df = pd.DataFrame(robu_out)
    val_df, test_df = train_test_split(robu_df, test_size=0.5)

    train_df = train_df.reset_index(drop=True)
    print(f'train shape: {train_df.shape}')
    val_df = val_df.reset_index(drop=True)
    print(f'val shape: {val_df.shape}')
    test_df = test_df.reset_index(drop=True)
    print(f'test shape: {test_df.shape}')
    print(train_df.head())

    train_df.to_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_train.feather'))
    val_df.to_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_val.feather'))
    test_df.to_feather(os.path.join(args.sim_path, f'{args.prefix}_sim_test.feather'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=3)
    parser.add_argument('--p', type=float, default=2)
    parser.add_argument('--epoch', type=float, default=10)
    parser.add_argument('--decay', type=float, default=1e-8)
    parser.add_argument('--sim_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='ratings.feather')
    parser.add_argument('--prefix', type=str, default='ml_1m_mf')
    parser.add_argument('--sample_sim', action='store_true')
    parser.add_argument('--item_sample_size', type=int, default=2000)
    parser.add_argument('--noise_ratio', type=float, default=1.0)
    parser.add_argument('--u_limit', type=int, default=500)
    parser.add_argument('--i_limit', type=int ,default=1000)
    args = parser.parse_args()
    main(args)