import os
import numpy as np
import pickle
import torch.nn.functional as F
import torch.nn as nn
import torch
import we
from tqdm import tqdm
from copy import deepcopy
import copy
import pickle
import time
import argparse
import gc
from multiprocessing import Process
import time
import configparser
import utils
utils.seed_everything() #Reproducibility

def parse_float(s):
    """to read the config file properly"""
    try: 
        return float(s)
    except:
        n, d = s.split("/")
        return float(n)/float(d)   

def get_ns_idb(word, N):
    """Quick reference from the pre-computed neighbour dictionary see `../GIPE/make_neighbours.py`"""
    return N[word]

def init_vector(word, E):
    """Initializing the vector with original embedding makes converges faster"""
    v = deepcopy(E.v(word)) 
    return torch.FloatTensor(v)

def torch_cosine_similarity(X, vectors):
    return torch.matmul(vectors, X) / (vectors.norm(dim=1) * X.norm(dim=0))

def ran_objective(X, sel, desel, g, ws):
    """The heart of Repulsion-Attraction-Neutralization"""
    w1, w2, w3 = ws
    A = torch.abs(torch_cosine_similarity(X, sel) - 1).mean(dim=0)/2
    if not isinstance(desel, bool):
        R = torch.abs(torch_cosine_similarity(X, desel)).mean(dim=0)
    else:
        R = 0 #nothing to repel
    N = torch.abs(X.dot(g)).mean(dim=0)    
    J = w1*R + w2*A + w2*N
    return J

#CPU
class RANDebias(nn.Module):
    def __init__(self, E, word, X, N, g, ws=[0.33, 0.33, 0.33], ripa=False):
        super(RANDebias, self).__init__()
        sel_max = 1
        desel_max = 100
        self.sel = N[word]['selected'][:sel_max]
        self.desel = N[word]['deselected'][:desel_max]
        self.sel = torch.FloatTensor(
                [E.v(l) for l in self.sel]).requires_grad_(True)
        
        if len(self.desel) == 0:
            self.desel = False 
        else:
            self.desel = torch.FloatTensor(
                    [E.v(l) for l in self.desel]).requires_grad_(True)
        self.X = nn.Parameter(X)
        self.E = E
        self.g = g
        self.word = word
        self.ws = ws

    def forward(self):
        return  ran_objective(self.X, self.sel, self.desel, 
                    self.g, self.ws)

def minimize(E, word, X, lr, max_epochs, *args, **kwargs):
    m = RANDebias(E, word, X, *args, **kwargs)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        out = m.forward()
        out.backward()
        optimizer.step()
    return m.X

def get_new_word_embs(E, word,*args, **kwargs):
    X = init_vector(word, E).requires_grad_(True) 
    debiased_X = minimize(E, word, X, *args, **kwargs)
    return debiased_X/torch.norm(debiased_X)

def get_N_info(words, N, sel_max=1, desel_max=100):
    """there's a sel_max arg because earlier I tried 
    to attract to more words than the original word itself.
    you can try experimenting for `sel_max` > 1, for me
    it yields worse results in gender bias tests and almost 
    no improvement in semantic and analogy tests."""

    thresh, N_info = 0.05, {} 
    for w in words:
        sel, desel = [], []
        try:
            for key in N[w]:
                if N[w][key] >= thresh: desel.append(key)
                else: sel.append(key)
        except:
            print(f"Problem for word: {w}")
        N_info[w] = dict(zip(['selected', 'deselected'], [sel[:sel_max], desel[:desel_max]]))  
    return N_info


def get_embedding_and_g(filename):
    E = we.WordEmbedding(filename)
    g = utils.get_g()
    g = torch.Tensor(g)
    return E, g

def debias_part(start, end, conf_file, sleep_t):
    config = configparser.ConfigParser()
    config.read(conf_file)[0]
    config = config['RAN-GloVe']

    learning_rate = parse_float(config['lr'])
    lambda_weights = [parse_float(config['lambda_1']), parse_float(config['lambda_2']), parse_float(config['lambda_3'])]
    n_epochs = int(config['n_epochs'])
    emb_file = config['emb_file']
    out_emb_name = config['out_emb_name']
    deb_file = config['deb_file']
    ns_file = config['ns_file']
    op_directory = config['op_directory']

    time.sleep(sleep_t) #memory gets full if I load all processes at once, gc.collect() takes some time to reach so wait.
    
    E, g = get_embedding_and_g(emb_file)
    with open(deb_file, 'rb') as handle:
        biased_list = pickle.load(handle)

    with open(ns_file, 'rb') as handle:
        N = pickle.load(handle)
    print(f"Loaded N for: {start} to {end}")

    wrds_to_debias = biased_list[start:end]

    N_info = get_N_info(wrds_to_debias, N)
    del N; gc.collect()

    new_embs = {}
    for word in tqdm(wrds_to_debias):
        try:
            new_embs[word] = get_new_word_embs(E, word,
                                learning_rate, n_epochs, N_info, g=g,
                                ws=lambda_weights).detach().numpy()
        except:
            print(f"Failed for word: {word}")

    out_emb_name = f"{out_emb_name}-{start}-to-{end}.dict.pickle"
    fname = os.path.join(op_directory, out_emb_name)

    if not os.path.exists(op_directory):
        os.makedirs(op_directory)

    with open(fname, "wb") as handle:
        print(f"Saving... {fname}")
        pickle.dump(new_embs, handle)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_name', default="config.ini", type=str,
                    help="config file. See `config.ini` for an example.")
    parser.add_argument('--deb_list_file', default="../data/debias.pkl", type=str,
                    help="pickled list of words to debias.")
    parser.add_argument('--n', default=4, type=int,
                    help="number of processes, usually set this equal to the number of physical cores.")
    args = parser.parse_args()
    conf_file = args.conf_name
    deb_list_file = args.deb_list_file
    n = args.n
    with open(deb_list_file, "rb") as handle:
        biased_list = pickle.load(handle)
    l = len(biased_list)
    width = l // n
    arguments = []
    sleeps = [30*i for i in range(n)]
    for i in range(n):
        start = int(i*width)
        end = int((i+1)*width)
        arguments.append([start, end, conf_file, sleeps[i]])
    ps = [Process(target=debias_part, args=arguments[i]) for i in range(n)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
