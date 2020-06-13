import os
import we
import json
import pickle
import argparse
import numpy as np
from numba import jit
from tqdm import tqdm
import utils
utils.seed_everything() #Reproducibility


def get_nbs(E, word, k=100):
    return np.argsort(E.vecs.dot(E.v(word)))[-k:][::-1]

@jit(nopython=False)
def get_pair_idb(w, v, g):
    w_orth = w - (np.dot(w, g)) * g
    v_orth = v - (np.dot(v, g)) * g
    dots = np.dot(w, v)
    orth_dots = np.dot(w_orth, v_orth)
    I = (dots - orth_dots / (np.linalg.norm(w_orth) * np.linalg.norm(v_orth))) / (dots)
    return I

def get_ns_idb(E, word, g):
    tops = get_nbs(E, word, 200) #We only need 100 neighbours, I am storing 200 anyway.
    wv = E.v(word)
    d = dict(zip([E.words[v] for v in tops], [get_pair_idb(E.vecs[v], wv, g) for v in tops]))
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default="../embeddings/glove", type=str)
    parser.add_argument('--o', default="../data/_glove_ns.pkl", type=str)
    parser.add_argument('--data_path', default="../data/", type=str)
    args = parser.parse_args()
    
    E = we.WordEmbedding(args.f)
    words = E.words
    g = utils.get_g(E)
    
    neighbour_idb_dict = dict(zip([w for w in tqdm(words)], 
                            [get_ns_idb(E, w, g) for w in tqdm(words)]))

    with open(args.o, 'wb') as handle:
        pickle.dump(neighbour_idb_dict, handle)
