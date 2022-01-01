import numpy as np


def get_triu_off_diag_flat(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]
