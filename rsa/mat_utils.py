import numpy as np


def get_triu_off_diag_flat(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def triu_off_diag_to_mat(triu1vec):
    nf = 0.5 * (1 + np.sqrt(8 * triu1vec.size + 1))
    n = int(nf)
    assert (nf == float(n))
    mat = np.zeros((n, n))
    idx = 0
    for r_idx in range(n):
        for c_idx in range(r_idx + 1, n):
            mat[r_idx, c_idx] = triu1vec[idx]
            idx += 1
    return mat
