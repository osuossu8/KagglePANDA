import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def weighted_nae(inp, targ):
    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])
    return torch.mean(torch.matmul(torch.abs(inp - targ), W.cuda()/torch.mean(targ, axis=0)))


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))


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

        features = d["features"].to(device, dtype=torch.float32)
        targets = d["targets"].to(device, dtype=torch.float32).view(-1, 5)

        model.zero_grad()
        outputs = model(features)

        loss = weighted_nae(outputs, targets)
        loss.backward()
        optimizer.step()

        y_true.append(targets.cpu().detach().numpy())
        y_pred.append(outputs.cpu().detach().numpy())

        losses.update(loss.item(), features.size(0))
        tk0.set_postfix(loss=losses.avg)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)
    print()


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            features = d["features"].to(device, dtype=torch.float32)
            targets = d["targets"].to(device, dtype=torch.float32).view(-1, 5)
            outputs = model(features)
            loss = weighted_nae(outputs, targets)
            y_true.append(targets.cpu().detach().numpy())
            y_pred.append(outputs.cpu().detach().numpy())
            losses.update(loss.item(), features.size(0))
            tk0.set_postfix(loss=losses.avg)
    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    domain = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    w = [0.3, 0.175, 0.175, 0.175, 0.175]

    m_all = 0
    for i in range(5):
        m = metric(y_true[:,i], y_pred[:,i])
        print(domain[i],'metric:', m)
        m_all += m*w[i]

    print('all_metric:', m_all)
    return m_all, losses.avg        
