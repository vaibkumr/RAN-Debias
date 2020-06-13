import os
import we
from tqdm import tqdm
import numpy as np
import copy
import pickle
import glob
import argparse
import utils
utils.seed_everything() #Reproducibility

def read(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', default="../embeddings/glove", type=str)
    parser.add_argument('--dict_dir', default="output", type=str)
    parser.add_argument('--out_dir', default="finals", type=str)
    parser.add_argument('--out_fname', default="RAN-GloVe", type=str)
    args = parser.parse_args()

    emb_file = args.emb_file
    dict_dir = args.dict_dir
    out_fname = args.out_fname
    out_dir = args.out_dir

    E = we.WordEmbedding(emb_file)

    fnames = glob.glob(f'{dict_dir}/*.dict.pickle')

    D = {}
    for fname in fnames:
        print(f"Reading {fname}...")
        d = read(fname)
        for k in d:
            D[k] = d[k]        

    new_embs = copy.deepcopy(E.vecs)
    for w in tqdm(D):
        new_embs[E.index[w]] = D[w]


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    vec_f = os.path.join(out_dir, f"{out_fname}.wv.npy")
    vocab_f = os.path.join(out_dir, f"{out_fname}.vocab")
    
    with open(vec_f, "wb") as handle:
        np.save(handle, new_embs)      

    with open(vocab_f, "w") as handle:
        handle.write('\n'.join(E.words))     

    print(f"Embeddings saved in npy format at {vec_f} & {vocab_f}")    
