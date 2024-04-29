import numpy as np
from sklearn import linear_model, model_selection

import torch
import torch.nn as nn


def get_losses(model, dataloader, device):
    model.eval()
    loss_f = nn.CrossEntropyLoss(reduction="none")
    
    losses = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            
            loss_arr = loss_f(pred, targets).cpu().detach().numpy()
            losses.append(loss_arr)

    return np.concatenate(losses)

def mia_score(a_losses, b_losses, seed, n_splits=5):
    min_n = min(len(a_losses), len(b_losses))
    
    a_losses = a_losses[:min_n]
    b_losses = b_losses[:min_n]

    samples = np.concatenate((a_losses, b_losses)).reshape((-1, 1))
    labels = [0] * min_n + [1] * min_n

    attack_model = linear_model.LogisticRegression(random_state=seed)
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=seed
    )
    
    mia_score = model_selection.cross_val_score(
        attack_model, samples, labels, cv=cv, scoring="accuracy"
    ).mean()
    
    return float(mia_score)
