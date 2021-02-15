from typing import List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import rest_api.utils
from sklearn.decomposition import PCA

import scripts.data_processing as dp


def plot_embeddings(
    embeddings: np.array, words: List[str], vocab: dp.Vocabulary = dp.Vocabulary()
) -> None:
    E_reduced = reduce_to_k_dim(embeddings)

    x = []
    y = []
    for word in words:
        word_x, word_y = E_reduced[vocab.word2idx(word.lower())]

        x.append(word_x)
        y.append(word_y)

        plt.annotate(word, (word_x + 0.01, word_y + 0.01), color="blue", fontsize=12)

    plt.scatter(x, y, c="red")
    plt.show()


def reduce_to_k_dim(embeddings: np.array, k: int = 2) -> np.array:
    E_reduced = None

    pca = PCA(n_components=k)
    E_reduced = pca.fit_transform(embeddings)

    return E_reduced


def paint_attention(
    original_image: np.array, encoder_input_dim: Tuple[int, int], context: np.array, beta: float
) -> np.array:
    resized = cv.resize(original_image, encoder_input_dim)

    context = cv.resize(context, encoder_input_dim)
    context = cv.GaussianBlur(context, (5, 5), 0, borderType=cv.BORDER_DEFAULT)
    context = cv.normalize(context, context, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    context = context.astype(np.float64)
    context *= beta
    context = context.astype(np.uint8)
    context = cv.applyColorMap(context, cv.COLORMAP_JET)

    context_vis = cv.addWeighted(resized, 0.5, context, 0.5, 0)

    return context_vis


def plot_betas(betas: List[float], words: List[str]) -> None:
    plt.ylim([0, 1.1])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    xs = range(len(words))
    plt.xticks(xs, words, rotation=20)

    plt.plot(xs, betas, "ro--")

    for x, y in zip(xs, betas):
        plt.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="blue",
        )
