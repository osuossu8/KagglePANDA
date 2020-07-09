import numpy as np
import pandas as pd
import albumentations
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
import scipy as sp
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
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
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

import skimage.io
from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

sys.path.append("/usr/src/app/kaggle/panda-challenge")

EXP_ID = 'exp6'
import configs.config6 as config
import src.engine6 as engine
from src.model import CustomSEResNeXt
from src.efficient_model import Efficient
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


# https://albumentations.readthedocs.io/en/latest/api/augmentations.html
'''
data_transforms = albumentations.Compose([
    # albumentations.Flip(p=0.3),  
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=(15,30), p=0.3),
    albumentations.Cutout(p=0.3),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    # ToTensorV2(),
    ])
'''

data_transforms = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])

data_transforms_test = albumentations.Compose([])


def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
    img3 = img2.reshape(
            img2.shape[0] // tile_size,
            tile_size,
            img2.shape[1] // tile_size,
            tile_size,
            3
    )

    img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
    n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
    return result, n_tiles_with_info >= n_tiles


def tile(img, sz=128, N=16):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img


def get_conc_tile(image):
    conc_image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2]]),
                             cv2.vconcat([image[3], image[4], image[5]]),
                             cv2.vconcat([image[6], image[7], image[8]])])
    return conc_image


tile_mode = 0
tile_size = 256 # 144 # 256
image_size = 256 # 144 # 256
n_tiles = 36
idxes = list(range(n_tiles))


class PANDADataset:
    def __init__(self, df, indices, transform=None):
        self.df = df.iloc[indices]
        self.data_provider = df.iloc[indices]['data_provider'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):

        level = 1 # 2# 一番小さかった data
        file_name = self.df['image_id'].values[item]
        file_path = config.TRAIN_IMG_PATH + f'{file_name}_{level}.jpeg'
        image = cv2.imread(file_path) 

        tiles, OK = get_tiles(image, tile_mode)

        n_row_tiles = int(np.sqrt(n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255
                # this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1+image_size, w1:w1+image_size] = this_img
                
        if self.transform is not None:
            augmented = self.transform(image=images)
            images = augmented['image']

        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1) 

        # targets = self.df['isup_grade'].values[item]
        targets = np.zeros(5).astype(np.float32)
        targets[:self.df['isup_grade'].values[item]] = 1.
        
        return {
            'file_names': file_name,
            'images': torch.tensor(images, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
        }


def run_one_fold(fold_id):

    df_train = pd.read_csv(config.TRAIN_PATH)
    LOGGER.info(df_train.shape)

    DEBUG = 0
    if DEBUG:
        df_train = df_train.head(25)

    TARGETS = 'isup_grade'


    kf = StratifiedKFold(n_splits = config.NUM_FOLDS, random_state = SEED)
    splits = list(kf.split(X=df_train, y=df_train[TARGETS].values))

    train_idx = splits[fold_id][0]
    val_idx = splits[fold_id][1]

    train_dataset = PANDADataset(df=df_train, indices=train_idx, 
                                 transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(
                   train_dataset, shuffle=True, 
                   batch_size=config.TRAIN_BATCH_SIZE,
                   num_workers=0, pin_memory=True)

    val_dataset = PANDADataset(df=df_train, indices=val_idx, 
                               transform=data_transforms_test)

    val_loader = torch.utils.data.DataLoader(
                 val_dataset, shuffle=False, 
                 batch_size=config.VALID_BATCH_SIZE,
                 num_workers=0, pin_memory=True)

    del train_dataset, val_dataset
    gc.collect()

    device = config.DEVICE
    model = Efficient(5, encoder='efficientnet-b0', pool_type="gem")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    patience = config.PATIENCE
    p = 0
    min_loss = 999
    best_score = -999

    for epoch in range(1, config.EPOCHS + 1):

        LOGGER.info("Starting {} epoch...".format(epoch))

        engine.train_fn(train_loader, model, optimizer, device, scheduler)
        score, val_loss, val_ids, val_preds = engine.eval_fn(val_loader, model, device)
        scheduler.step()

        if val_loss < min_loss:
            min_loss = val_loss
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config.OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
            LOGGER.info(f'min valid loss : {min_loss}')
            LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))
            p = 0 

        if p > 0: 
            LOGGER.info(f'val loss is not updated while {p} epochs of training')
        p += 1
        if p > patience:
            LOGGER.info(f'Early Stopping')
            break

    to_pickle(os.path.join(config.OUT_DIR, '{}_fold{}.pkl'.format(EXP_ID, fold_id)), [val_ids, val_preds])
    LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))


def calc_overall_kappa(EXP_ID):
    df_train = pd.read_csv(config.TRAIN_PATH)

    val_idices0, val_preds0 = unpickle(f'models/{EXP_ID}_fold0.pkl')
    val_idices1, val_preds1 = unpickle(f'models/{EXP_ID}_fold1.pkl')
    val_idices2, val_preds2 = unpickle(f'models/{EXP_ID}_fold2.pkl')
    # val_idices3, val_preds3 = unpickle(f'models/{EXP_ID}_fold3.pkl')

    all_idx = np.concatenate([np.concatenate(val_idices0), 
                              np.concatenate(val_idices1),
                              np.concatenate(val_idices2),
                              # np.concatenate(val_idices3),
    ])

    all_val_preds = np.concatenate([val_preds0,
                                    val_preds1,
                                    val_preds2,
                                    # val_preds3,
    ]) 

    df_val = pd.DataFrame(columns=['image_id', 'isup_grade'])
    df_val['image_id'] = all_idx
    df_val['isup_grade'] = all_val_preds
    print(df_val.head(3))

    optimized_rounder = engine.OptimizedRounder()
    optimized_rounder.fit(df_val.isup_grade, df_train.isup_grade)
    coefficients = optimized_rounder.coefficients()
    final_preds = optimized_rounder.predict(df_val.isup_grade, coefficients)
    LOGGER.info(f'Counter preds: {Counter(final_preds)}')
    LOGGER.info(f'coefficients: {coefficients}')
    kappa = engine.quadratic_weighted_kappa(df_train.isup_grade, final_preds)
    LOGGER.info(f'overall kappa score: {kappa}')
    

if __name__ == '__main__':

    fold0_only = config.FOLD0_ONLY
 
    LOGGER.info(f'{EXP_ID} : exp4 (LB 0.85) + apply img256 efficientb0')
    
    for fold_id in range(config.NUM_FOLDS):

        LOGGER.info("Starting fold {} ...".format(fold_id))

        run_one_fold(fold_id)

        if fold0_only:
            LOGGER.info("This is fold0 only experiment.")
            break
    
    calc_overall_kappa(EXP_ID)

    LOGGER.info('all process done!')

