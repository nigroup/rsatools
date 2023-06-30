from itertools import permutations
import numpy as np
from rsa.rdm_utils import triu_off_diag_vec_to_rdm


def permute_rdm(rdm):
    if rdm.ndim == 1:
        rdm = triu_off_diag_vec_to_rdm(rdm)
    num_rows, _ = rdm.shape
    if num_rows == 2:
        r_new = np.copy(rdm)
    else:
        itr = permutations(range(num_rows))
        non_perm = np.arange(num_rows)
        is_perm = False
        while not is_perm:
            perm = next(itr)
            is_perm = num_rows > 2 and not np.all(perm == non_perm)
        r_new = np.zeros_like(rdm)
        r_new[:, :] = rdm[perm, :]
        r_new[:, :] = r_new[:, perm]
    return r_new
