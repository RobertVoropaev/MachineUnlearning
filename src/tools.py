import random, os
import numpy as np
import torch

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_forget_retain_mask(train_size, forget_shape, seed=0):
    forget_size = int(train_size * forget_shape)
    retain_size = train_size - forget_size

    mask_forget = np.full(train_size, False)
    mask_forget[:forget_size] = True
    
    np.random.seed(seed)
    np.random.shuffle(mask_forget)
    mask_retain = ~mask_forget
    return mask_forget, mask_retain