import numpy as np


def recall_at_k(rank, plabels=None, glabels=None, k=1):
    """Compute R@K: Recall@K (K=1, 5, 10) represents the percentage of the queries
    where at least one ground-truth is retrieved among the top-K results.
    ---------------------------------------------------
    Inputs:
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        probe sample and j-th gallery sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
    ---------------------------------------------------
    Outputs:
    out : numpy.ndarray, The recall@k accuracy
    ---------------------------------------------------
    """
    n_probe, n_gallery = rank.shape
    match = 0
    for i in range(n_probe):
        match += int(sum(glabels[rank[i, :k]] == plabels[i]) > 0)

    score = match * 1.0 / n_probe

    return score


def average_precision_at_k(rank, plabels=None, glabels=None, k=50):
    """Compute AP@K: We report the AP@K, the percent of top-K scoring images whose
    class matches that of the text query, averaged over all the test classes.
    ---------------------------------------------------
    Inputs:
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        probe sample and j-th gallery sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
    ---------------------------------------------------
    Outputs:
    out : numpy.ndarray, The AP@K accuracy
    ---------------------------------------------------
    """
    n_probe, n_gallery = rank.shape
    match = 0
    average_precision = 1.0 * np.zeros_like(plabels)

    for i in range(n_probe):
        relevant_size = sum(glabels == plabels[i])
        hit_index = np.where(glabels[rank[i, :k]] == plabels[i])
        precision = 1.0 * np.zeros_like(hit_index[0])
        for j in range(hit_index[0].shape[0]):
            hitid = max(1, hit_index[0][j])
            precision[j] = sum(glabels[rank[i, :hitid]] == plabels[i]) * 1.0 / (hit_index[0][j] + 1)
        average_precision[i] = np.sum(precision) * 1.0 / relevant_size

    score = np.mean(average_precision)

    return score


def mean_average_precision(rank, plabels=None, glabels=None):
    """Compute the Mean Average Precision.
    ---------------------------------------------------
    Inputs:
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        probe sample and j-th gallery sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
    ---------------------------------------------------
    Outputs:
    out : numpy.ndarray, The MAP result
    ---------------------------------------------------
    """
    n_probe, n_gallery = rank.shape
    average_precision = 1.0 * np.zeros_like(plabels)

    for i in range(n_probe):
        relevant_size = sum(glabels == plabels[i])
        hit_index = np.where(glabels[rank[i, :]] == plabels[i])
        precision = 1.0 * np.zeros_like(hit_index[0])
        assert relevant_size == hit_index[0].shape[0]
        for j in range(relevant_size):
            hitid = max(1, hit_index[0][j])
            precision[j] = sum(glabels[rank[i, :hitid]] == plabels[i]) * 1.0 / (hit_index[0][j] + 1)
        average_precision[i] = np.sum(precision) * 1.0 / relevant_size

    score = np.mean(average_precision)

    return score
