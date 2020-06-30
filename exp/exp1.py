import numpy as np
import pandas as pd
import collections
import cv2
import datetime
import gc
import glob
import h5py
import logging
import math
import operator
import os 
import pickle
import random
import re
import sklearn
import scipy.signal
import scipy.stats as stats
import seaborn as sns
import string
import sys
import time
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from contextlib import contextmanager
from collections import Counter, defaultdict, OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from skimage import measure
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from torch.utils import model_zoo
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

sys.path.append("/usr/src/app/kaggle/panda-challenge")

EXP_ID = 'exp1'
import configs.config as config
import src.engine as engine
from src.model import resnet10, resnet34
from src.machine_learning_util import seed_everything, prepare_labels, timer, to_pickle, unpickle


SEED = 718
seed_everything(SEED)


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


class PANDADataset:
    def __init__(self, df, target_cols, indices, map_path):
        self.df = df.iloc[indices]
        self.target = df.iloc[indices][target_cols+['Id']]
        self.target = self.target.fillna(self.target.mean())
        self.map_path = map_path
        self.target_cols = target_cols

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):

        IDX = self.df.iloc[item].Id        
        path = self.map_path + str(IDX)
        
        all_maps = h5py.File(path + '.mat', 'r')['SM_feature'][()]
        targets = self.target[self.target.Id==IDX][self.target_cols].values

        return {
            'idx': torch.tensor(IDX, dtype=torch.float32)
            'features': torch.tensor(all_maps, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
        }


def run_one_fold(fold_id):

    fnc_df = pd.read_csv(config.FNC_PATH)
    loading_df = pd.read_csv(config.LOADING_PATH)
    labels_df = pd.read_csv(config.TRAIN_SCORES_PATH)


    fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
    df = fnc_df.merge(loading_df, on="Id")
    labels_df["is_train"] = True

    df = df.merge(labels_df, on="Id", how="left")

    df['bin_age'] = pd.cut(df['age'], [i for i in range(0, 100, 10)], labels=False)

    df_test = df[df["is_train"] != True].copy()
    df_train = df[df["is_train"] == True].copy()


    num_folds = config.NUM_FOLDS
    kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
    splits = list(kf.split(X=df_train, y=df_train[['bin_age']]))

    train_idx = splits[fold_id][0]
    val_idx = splits[fold_id][1]

    target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

    print(len(train_idx), len(val_idx))


    train_dataset = PANDADataset(df=df_train, target_cols=target_cols, indices=train_idx, map_path=config.TRAIN_MAP_PATH)
    train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, 
                batch_size=config.TRAIN_BATCH_SIZE,
                num_workers=0, pin_memory=True)

    val_dataset = PANDADataset(df=df_train, target_cols=target_cols, indices=val_idx, map_path=config.TRAIN_MAP_PATH)
    val_loader = torch.utils.data.DataLoader(
                val_dataset, shuffle=False, 
                batch_size=config.VALID_BATCH_SIZE,
                num_workers=0, pin_memory=True)

    del train_dataset, val_dataset
    gc.collect()


    device = config.DEVICE
    model = resnet10()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-5) 

    patience = config.PATIENCE
    p = 0
    min_loss = 999
    best_score = -999

    for epoch in range(1, config.EPOCHS + 1):

        print("Starting {} epoch...".format(epoch))

        engine.train_fn(train_loader, model, optimizer, device, scheduler)
        score, val_loss, val_ids, val_preds = engine.eval_fn(val_loader, model, device)
        scheduler.step()
        # scheduler.step(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config.OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
            print("save model at score={} on epoch={}".format(best_score, best_epoch))
            p = 0 

        if p > 0: 
            print(f'val loss is not updated while {p} epochs of training')
        p += 1
        if p > patience:
            print(f'Early Stopping')
            break

    to_pickle(os.path.join(config.OUT_DIR, '{}_fold{}.pkl'.format(EXP_ID, fold_id)), [val_ids, val_preds])
    print("best score={} on epoch={}".format(best_score, best_epoch))


if __name__ == '__main__':

    fold0_only = config.FOLD0_ONLY

    for fold_id in range(5):

        LOGGER.info("Starting fold {} ...".format(fold_id))

        run_one_fold(fold_id)

        if fold0_only:
            LOGGER.info("This is fold0 only experiment.")
            break
