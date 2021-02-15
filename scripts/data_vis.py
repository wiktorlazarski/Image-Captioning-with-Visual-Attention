from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import scripts.data_processing as dp
import rest_api.utils

def plot_embeddings(embeddings: np.array, words: List[str], vocab: dp.Vocabulary = dp.Vocabulary()) -> None:
    E_reduced = reduce_to_k_dim(embeddings)

    x = []
    y = []
    for word in words:
        word_x, word_y = E_reduced[vocab.word2idx(word.lower())]

        x.append(word_x)
        y.append(word_y)

        plt.annotate(word, (word_x, word_y), color="blue", fontsize=12)

    plt.scatter(x, y, c="red")
    plt.show()


def reduce_to_k_dim(embeddings: np.array, k: int = 2) -> np.array:
    E_reduced = None

    pca = PCA(n_components=k)
    E_reduced = pca.fit_transform(embeddings)

    return E_reduced
