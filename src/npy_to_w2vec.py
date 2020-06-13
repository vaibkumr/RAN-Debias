import numpy as np
import we
from tqdm import tqdm
import argparse
import random


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default='embeddings/glove', type=str)
    parser.add_argument('--o', default='my_emb', type=str)
    parser.add_argument('--dim', default=300, type=int)

    args = parser.parse_args()

    if args.o == 'my_emb':
        output_path = f'my_emb_{random.randint(1,10000)}.txt'
    else:
        output_path = args.o

    print(f"Converting: {args.f}")

    E = we.WordEmbedding(args.f)
    lines = [f'{len(E.words)} {args.dim}']
    for word in tqdm(E.words):
        vector = ' '.join([str(v) for v in E.v(word)])
        word = ' '.join([w for w in word.split()])
        line = word + ' ' + vector
        lines.append(line)

    with open(output_path, 'w') as handle:
        handle.write('\n'.join(lines))
        
    print(f"File saved in {output_path}")
