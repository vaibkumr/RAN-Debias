# RAN-Debias: Repulsion-Attraction-Neutralization Debias (TACL)
This is the code for the TACL paper titled ["Nurse is Closer to Woman than Surgeon? Mitigating Gender-Biased Proximities in Word Embeddings"](https://arxiv.org/abs/2006.01938) by Me (Vaibhav Kumar), Vaibhav Kumar, Tenzin Bhotia and Tanmoy Chakraborty. 

# Abstract
Word embeddings are the standard model for semantic and syntactic representations of words. Unfortunately, these models have been shown to exhibit undesirable word associations resulting from gender, racial, and religious biases. Existing post-processing methods for debiasing word embeddings are unable to mitigate gender bias hidden in the spatial arrangement of word vectors. In this paper, we propose RAN-Debias, a novel gender debiasing methodology which not only eliminates the bias present in a word vector but also alters the spatial distribution of its neighbouring vectors, achieving a bias-free setting while maintaining minimal semantic offset. We also propose a new bias evaluation metric - Gender-based Illicit Proximity Estimate (GIPE), which measures the extent of undue proximity in word vectors resulting from the presence of gender-based predilections. Experiments based on a suite of evaluation metrics show that RAN-Debias significantly outperforms the state-of-the-art in reducing proximity bias (GIPE) by at least 42.02\%. It also reduces direct bias, adding minimal semantic disturbance, and achieves the best performance in a downstream application task (coreference resolution).


# Pretrained embedding
- Our pretrained word embedding (trained on wikidump dataset; 322636 tokens), RAN-GloVe can be found [here](https://drive.google.com/drive/folders/14yebEnP4kXHsTisfbeWxzo0J42O54QtD?usp=sharing).


# Requirements
Python 3.6 or above with the following packages:
```
torch==1.3.1
numpy==1.17.2
tqdm==4.36.1
numba==0.45.0
```

# Instructions
## Files:
- `data/debias.pkl`: Pickled python list of words subjected to debiasing procedure (created using the KBC algorithm in the paper). 
- `src/make_neighbours.py`: Code to create neighbourhood dictionary needed for both GIPE and RAN-Debias.
- `src/gipe.py`: Code for Gender-based Illicit Proximity Estimate.
- `src/gipe.sh`: Reproduce results for Gender-based Illicit Proximity Estimate as mentioned the in paper. 
- `src/npy_to_w2vec.py`: Convert word embeddings from numpy format to the w2vec txt format. In numpy format the vectors are stored as a pickled numpy array and vocabulary is stored as a seperate text file.
- `src/w2vec_to_npy.py`: Convert word embeddings from w2vec txt format to the numpy format.
- `src/RANDebias.py`: Code for the RAN Debiasing prodecure.
- `src/combine.py`: Combine multi-processing output from `src/RANDebias.py` to create the final embedding.
- `src/RANDebias.sh`: Code to reproduce the RAN-Debiasing procedure.
- `src/utils.py`: Some utility files.
- `src/we.py`: Modifed version of `we.py` provided used [here](https://github.com/tolga-b/debiaswe).
- `src/config.ini`: Config files with data paths and hyperparams for RAN-Debias.

## RAN-Debias
To debias any word embedding, follow these steps:
1. Convert the embedding to numpy format using `src/w2vec_to_npy.py`.
2. Create the neighbours dictionary using `src/make_neighbours.py` (This is basically a word-list pair where a list of neighbours is precomputed and stored every word, it makes the procedure much faster).
3. Edit the location of these files in `config.ini`.
4. Run `python RANDebias.py --conf_name config.ini --deb_list_file ../data/debias.pkl`.

## GIPE
To compute GIPE for your word embeddings:
1. Convert the embedding to numpy format using `src/w2vec_to_npy.py`.
2. Create the neighbours dictionary using `src/make_neighbours.py` (this is basically a word-list pair where a list of nearest neighbours is precomputed and stored for every word, it makes the procedure much faster).
4. Run `python gipe.py --f <embedding_path> --dictf <neighbour_dict_path> --idb_thresh 0.03`.

**Note**: We provide the neighbourhood dictionary for RAN-Debias as a reference [here](https://drive.google.com/drive/folders/14yebEnP4kXHsTisfbeWxzo0J42O54QtD?usp=sharing).

## Reproducing results
- Sembias: The details, code and dataset for this test can be found on [this repository by Zhao et al.](https://github.com/uclanlp/gn_glove/).
- Word Semantic Similarity Tests and Analogy Tests: We use the [Word Embeddings Benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks/) for these tests.
- Coreference Resolution: The End-to-end Neural Coreference Resolution model used is from [this repository by Lee et al. 2017](https://github.com/kentonl/e2e-coref/tree/e2e).

# Cite
todo
