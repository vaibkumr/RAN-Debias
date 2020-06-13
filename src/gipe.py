import numpy as np
import os
import json
import codecs
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import pickle
import we
import argparse
from collections import defaultdict
import utils
utils.seed_everything() #Reproducibility


def get_neighbors(N, word, k):
    return list(N[word].keys())[1:k+1]

def get_pair_idb(w, v, g, E):
    if isinstance(w, str): w = E.v(w)
    if isinstance(v, str): v = E.v(v)
    w_orth = w - (np.dot(w, g)) * g
    v_orth = v - (np.dot(v, g)) * g
    dots = np.dot(w, v)
    orth_dots = np.dot(w_orth, v_orth)
    idb = (dots - orth_dots / (np.linalg.norm(w_orth) * np.linalg.norm(v_orth))) / (dots )
    return idb

def prox_bias(vals, l, thresh):
    return len(vals[vals>thresh]) / l

def get_words(bias_listPath):
    with open(bias_listPath, 'rb') as handle:
        biased_words = pickle.load(handle)
    return biased_words

def gipe(biased_words, E_new, E_orig, g, dict_file,thresh = 0.05, n=100):
    total = 0
    neighbours = {}
    incoming_edges = defaultdict(list)
    etas = {}
    with open(dict_file, 'rb') as handle:
        N = pickle.load(handle)
    for word in tqdm(biased_words): #Creating BBN
        try:
            neighbours[word] = get_neighbors(N, word, n) #Neighbours according to current embedding
            l = len(neighbours[word])
        except:
            print(f"{word} is weird.")
            continue
        values = []

        for i, element in enumerate(neighbours[word]):
            value = float(get_pair_idb(word, element, g, E_orig))  #Beta according to original (same in case of non-debiased) embedding
            values.append(value)
            incoming_edges[element].append(value)
        etas[word] = prox_bias(np.array(values), l, thresh)

    eps = np.finfo(float).eps
    weights = defaultdict(int)
    for key in incoming_edges:
        idbs = np.array(incoming_edges[key])
        weights[key] = 1 + (len(idbs[idbs>thresh])/(len(idbs) + eps))
    
    return etas, weights

def score(vals, weights):
    score = 0
    sum = 0
    for v in vals:
        try:
            score += weights[v] * vals[v]
            sum += weights[v]
        except:
            aux_w = 1  #By default, the weight is 1 (1 is the lowest possible weight, means lowest "penalty")
            score += vals[v] * aux_w
            sum += aux_w
    score /= sum
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default="../embeddings/glove", type=str,
                        help = "embeddings (npy format) to perform the test for.")
    parser.add_argument('--idb_thresh', default=0.03, type=float,
                        help = "threshold for indirect bias.")
    parser.add_argument('--n', default=100, type=int,
                        help = "number of neighbours to consider.")
    parser.add_argument('--orig_emb', default="../embeddings/glove", type=str,
                        help = "original embedding (npy format), same in case of non-debiased embedding\
                        idb is calculated using original debiased therefore you need to provide it here.")
    parser.add_argument('--dictf', default="../data/glove_ns.pkl", type=str,
                        help = "pkl file for neighbour dict (see `make_neighbours.py` for this)")
    parser.add_argument('--bias_list', default="../data/debias.pkl", type=str,
                        help = "pkl file for list of words to debias (V_d, debias set in paper).")
    args = parser.parse_args()

    E_orig = we.WordEmbedding(args.orig_emb)
    X = we.WordEmbedding(args.f)
    g = utils.get_g(E_orig)

    biased_words = get_words(args.bias_list)

    vals, wts = gipe(biased_words, X, E_orig, g, args.dictf, args.idb_thresh, args.n)
    gipe_score = score(vals, wts)
    print(gipe_score)

    fname = "gipe-results.txt"
    with open(fname, 'a+') as handle:
        handle.write("=========================================================\n")
        handle.write(f"emb: {args.f} | idb_thresh: {args.idb_thresh} \n{gipe_score}\n")
        print(f"Written results in file {fname}")
