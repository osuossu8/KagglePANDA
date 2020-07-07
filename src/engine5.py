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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    new_data = data.clone()

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


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


def cutmix_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = RMSELoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    y_true = []
    y_pred = []
    for bi, d in enumerate(tk0):

        images = d["images"].to(device, dtype=torch.float32)
        targets = d["targets"].to(device, dtype=torch.float32)
        model.zero_grad()


        if np.random.rand()<0.5:
            new_images, new_targets = cutmix(images, targets, 0.4)
            outputs = model(new_images).view(-1)
            loss = cutmix_criterion(outputs, new_targets)

        else:
            outputs = model(images).view(-1)
            loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        y_true.append(targets)
        y_pred.append(outputs)
        losses.update(loss.item(), images.size(0))
        tk0.set_postfix(loss=losses.avg)

    y_true = torch.cat(y_true).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().detach().numpy()
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
            targets = d["targets"].to(device, dtype=torch.float32)
            outputs = model(images).view(-1)
            loss = loss_fn(outputs, targets)

            y_true.append(targets)
            y_pred.append(outputs)
            val_ids.append(d["file_names"])
            losses.update(loss.item(), images.size(0))
            tk0.set_postfix(loss=losses.avg)
    y_true = torch.cat(y_true).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().detach().numpy()

    optimized_rounder = OptimizedRounder()
    optimized_rounder.fit(y_pred, y_true)
    coefficients = optimized_rounder.coefficients()
    final_preds = optimized_rounder.predict(y_pred, coefficients)
    print(f'Counter preds: {Counter(final_preds)}')
    print(f'coefficients: {coefficients}')
    kappa = quadratic_weighted_kappa(y_true, final_preds)

    print('kappa score:', kappa)
    return kappa, losses.avg, val_ids, final_preds   
