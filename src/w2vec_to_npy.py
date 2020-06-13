import numpy as np
import pickle
import we
from tqdm import tqdm
import argparse
import random


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default='embeddings/glove.txt', type=str)
    parser.add_argument('--o', default='my_emb', type=str)
    parser.add_argument('--dim', default=300, type=int)
    parser.add_argument('--d', default=False, type=bool)
    args = parser.parse_args()

    if args.o == 'my_emb': #Dont overwrite by mistake (I hope to not get unlucky and I like to try my luck so... ROLL!)
        yolo_factor = 1000 #lets not get too cocky
        output_path = f'my_emb_{random.randint(1, yolo_factor)}'

    dim = args.dim
    debug = args.d
    with open(args.f, "r") as f:
        lines = f.readlines()
        words = []
        vecs = []
        for line in tqdm(lines):
            tokens = line.split()
            v = np.array([float(x) for x in tokens[-dim:]])
            w = " ".join([str(x) for x in tokens[:-dim]])
            if len(v) != dim or w == " ":
                print(f"Weird line: {tokens} | {len(v)}")
                continue
            if w in words and debug:
                print(f"Two vecs for {w}??.. Skipping")
                continue
            words.append(w)
            vecs.append(v)

    vecs = np.array(vecs, dtype='float32')
    print(vecs.shape)

    with open(f"{args.o}.wv.npy", "wb") as handle:
        np.save(handle, vecs)

    with open(f"{args.o}.vocab", "w") as handle:
        handle.write('\n'.join(words))

    print(f"Saved in: {args.o}.wv.npy and {args.o}.vocab")
