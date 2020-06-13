import numpy as np
import os
import torch
import we
import random

def seed_everything(seed=42):
    """source: somewhere on kaggle"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_g(E):
    definitional = [['woman', 'man'],
                    ['girl', 'boy'],
                    ['she', 'he'],
                    ['mother', 'father'],
                    ['daughter', 'son'],
                    ['gal', 'guy'],
                    ['female', 'male'],
                    ['her', 'his'],
                    ['herself', 'himself']] #Source: tolga
    g = we.doPCA(definitional, E).components_[0]
    return g    