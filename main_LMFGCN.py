
""" A Multi-view GCN workflow with intermediate integration strategy for Autism MRI data Classification.
Implementation of the paper: (https://www.biorxiv.org/content/10.1101/2023.06.26.546426v1.full.pdf)
Author of this code: Simon Wang (https://github.com/DLDLCQJ/ASD-trajectory-and-classification)
Created : 07/20/2023
"""

import os
import argparse 
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from utils import *
from training_scripts import train_test

set_seed(123)
def parse_arguments():
    parser = argparse.ArgumentParser(description='LMFGCN')
    parser.add_argument('--path', type=str, default='/Users/simon/KY/code_jupyter/GCN/mydata-shen-ABIDE-I-II', help='Path of data files')
    parser.add_argument('--view_list', nargs='+', type=int, default=[1, 2, 3], help='Number of views')
    parser.add_argument('--lr_e', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr_c', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr_e_pretrain', type=float, default=0.005, help='Learning rate for pretraining')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--num_epoch_pretrain', type=int, default=200, help='Number of epochs to pretrain')
    parser.add_argument('--latent_dim_list', nargs='+', type=int, default=[476, 68, 32], help='Dimension of latent space')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--k_order', type=int, default=2, help='Chebychev filters order')
    parser.add_argument('--rank', type=int, default=2, help='Number of rank')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN dropout rate')
    parser.add_argument('--parameter', type=int, default=8, help='Parameter for weighted adjacency matrix')
    parser.add_argument('--model_type', type=str, default='ChebConv', help='Type of GCN model')
    return parser.parse_args()

def load_data(path, view_list):
    labels = np.loadtxt(os.path.join(path, "labels.csv"), skiprows=1, delimiter=',')
    data_list = [scale(np.loadtxt(os.path.join(path, f"{i}_feature.csv"), skiprows=1, delimiter=',')) for i in view_list]
    y = np.array([int(label) for label in labels]).reshape(-1, 1)
    return labels, data_list, y

def run_training(args, labels, data_list, y):
    #scores = Parallel(n_jobs=10)(delayed(train_test)(
    scores =[train_test(
        trval_idx, tr_idx, te_idx, labels, data_list, args.view_list, args.num_class, args.num_layers, 
        args.model_type, args.lr_e, args.lr_e_pretrain, args.lr_c, args.num_epoch, args.patience, args.checkpoints,
        args.num_epoch_pretrain, args.parameter, args.latent_dim_list, args.gcn_dropout, args.k_order, args.rank)
        for trval_idx, tr_idx, te_idx in reversed(list(StratifiedKFoldn(n_splits=10).split(np.zeros(len(labels)), y.squeeze())))]

    return scores

if __name__ == "__main__":
    args = parse_arguments()

    # Ensure checkpoints directory exists
    os.makedirs(args.checkpoints, exist_ok=True)

    # Load and preprocess data
    labels, data_list, y = load_data(args.path, args.view_list)

    # Run training and collect scores
    scores = run_training(args, labels, data_list, y)
