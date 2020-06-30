import logging
import os
import random
import numpy as np
import pandas as pd
import pickle
import time
import torch
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def seed_everything(seed=1129):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p    


class DownSampler(object):
    def __init__(self, random_states, frac=None, n=None):
        self.random_states = random_states
        self.frac = frac
        self.n = n

    def transform(self, data, target):
        if self.frac is not None:
            positive_data = data[data[target] == 1].sample(frac=self.frac)
        elif self.n is not None:
            positive_data = data[data[target] == 1].sample(n=self.n)
        else:
            positive_data = data[data[target] == 1]
        positive_ratio = len(positive_data) / len(data)
        negative_data = data[data[target] == 0].sample(
            frac=positive_ratio / (1 - positive_ratio), random_state=self.random_states)
        return pd.concat([positive_data, negative_data], sort=True).reset_index(drop=True)
