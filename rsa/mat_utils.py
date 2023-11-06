import numpy as np


def get_triu_off_diag_flat(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def num_mat_rows(num_triu1_elements):
    nf = 0.5 * (1 + np.sqrt(8 * num_triu1_elements + 1))
    n = int(nf)
    assert (nf == float(n))
    return n


def triu_off_diag_to_mat(triu1vec):
    n = num_mat_rows(triu1vec.size)
    mat = np.zeros((n, n))
    idx = 0
    for r_idx in range(n):
        for c_idx in range(r_idx + 1, n):
            mat[r_idx, c_idx] = triu1vec[idx]
            idx += 1
    return mat
