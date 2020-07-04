import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import rankdata

import scipy as sp
from functools import partial
from collections import defaultdict, Counter
from sklearn.metrics import cohen_kappa_score

# LOGGER = logging.getLogger()


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loss_fn(logits, targets):
    loss_fct = RMSELoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    y_true = []
    y_pred = []
    for bi, d in enumerate(tk0):

        images = d["images"].to(device, dtype=torch.float32)
        # images_lv1 = d["images_lv1"].to(device, dtype=torch.float32)
        targets = d["targets"].to(device, dtype=torch.float32)
        model.zero_grad()
        outputs = model(images).view(-1)
        # outputs = model(images, images_lv1).view(-1)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        y_true.append(targets.cpu().detach().numpy())
        y_pred.append(outputs.cpu().detach().numpy())
        # y_pred.append(torch.sigmoid(outputs).cpu().detach().numpy())
        losses.update(loss.item(), images.size(0))
        tk0.set_postfix(loss=losses.avg)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)
    print()


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    y_true = []
    y_pred = []
    val_ids = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            images = d["images"].to(device, dtype=torch.float32)
            # images_lv1 = d["images_lv1"].to(device, dtype=torch.float32)
            targets = d["targets"].to(device, dtype=torch.float32)
            outputs = model(images).view(-1)
            # outputs = model(images, images_lv1).view(-1)
            loss = loss_fn(outputs, targets)
            y_true.append(targets.cpu().detach().numpy())
            y_pred.append(outputs.cpu().detach().numpy())
            # y_pred.append(torch.sigmoid(outputs).cpu().detach().numpy())
            val_ids.append(d["file_names"])
            losses.update(loss.item(), images.size(0))
            tk0.set_postfix(loss=losses.avg)
    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    # rank 化する
    # y_pred = rankdata(y_pred) / len(y_pred)

    optimized_rounder = OptimizedRounder()
    optimized_rounder.fit(y_pred, y_true)
    coefficients = optimized_rounder.coefficients()
    final_preds = optimized_rounder.predict(y_pred, coefficients)
    print(f'Counter preds: {Counter(final_preds)}')
    print(f'coefficients: {coefficients}')
    kappa = quadratic_weighted_kappa(y_true, final_preds)

    print('kappa score:', kappa)
    return kappa, losses.avg, val_ids, final_preds   
