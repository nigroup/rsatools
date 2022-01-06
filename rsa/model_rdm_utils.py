import multiprocessing as mp
import tqdm
import numpy as np
from scipy.stats import spearmanr
import rsa.mat_utils as mutils

ENTRY_EMPTY = -999


def get_input_rdm_flat_from_file(fpath):
    rdm = np.load(fpath)
    if rdm.ndim == 1:
        return rdm
    elif rdm.ndim == 2 and 1 in rdm.shape:
        return rdm.flatten()
    elif rdm.ndim > 2:
        raise ValueError("File does not contain a 2D matrix (%s)" % fpath)
    else:
        return mutils.get_triu_off_diag_flat(rdm)


def calc_spearman_rank_corr_from_files(fpath_in_i, fpath_in_j, i, j):
    rdm_i = np.load(fpath_in_i)
    rdm_j = np.load(fpath_in_j)
    return i, j, spearmanr(rdm_i, rdm_j)


def calc_model_rdm(fpath_in_rdms_all_1, fpath_in_rdms_all_2, processes=1):
    if fpath_in_rdms_all_1 == fpath_in_rdms_all_2:
        model_rdm = calc_model_rdm_symmetric(fpath_in_rdms_all_1, processes=processes)
    else:
        raise NotImplementedError
    return model_rdm


def calc_model_rdm_symmetric(fpath_in_all, processes=1):
    if processes < 2:
        model_rdm = _calc_model_rdm_symmetric_sequential(fpath_in_all)
    else:
        model_rdm = _calc_model_rdm_symmetric_parallel(fpath_in_all, processes)
    return model_rdm


def _calc_model_rdm_symmetric_sequential(fpath_in_all):
    nrows = len(fpath_in_all)
    ncols = nrows
    model_rdm = np.zeros((nrows, ncols)) + ENTRY_EMPTY

    for r, fpath_in_rdm_1 in enumerate(fpath_in_all):
        in_rdm_1 = get_input_rdm_flat_from_file(fpath_in_rdm_1)  # .flatten()
        for c, fpath_in_rdm_2 in enumerate(fpath_in_all):
            if r == c:
                # dissimilarity with itself is zero
                model_rdm[r, c] = 0.
            elif model_rdm[r, c] == ENTRY_EMPTY:
                in_rdm_2 = get_input_rdm_flat_from_file(fpath_in_rdm_2)  # .flatten()
                # assumes symmetry, essentially list 1 == list 2...
                model_rdm[r, c] = model_rdm[c, r] = 1 - spearmanr(in_rdm_1, in_rdm_2)[0]
    return model_rdm


def _calc_model_rdm_symmetric_parallel(fpath_in_all, processes):
    nrows = len(fpath_in_all)
    ncols = nrows
    up_triangle_idxs = [(i, j) for i in range(nrows) for j in range(i + 1, ncols)]
    with mp.Pool(processes=processes) as pool:
        result = pool.starmap(calc_spearman_rank_corr_from_files,
                              tqdm.tqdm(
                                  [(fpath_in_all[i],
                                    fpath_in_all[j], i, j)
                                   for (i, j) in up_triangle_idxs], total=len(up_triangle_idxs)),
                              chunksize=10,
                              )

    model_rdm = np.zeros((nrows, ncols)) + ENTRY_EMPTY
    for r in range(nrows):
        model_rdm[r, r] = 0.  # set diagonal to zero dissimilarity
    for r, c, result_spearman in result:
        model_rdm[r, c] = model_rdm[c, r] = 1 - result_spearman.correlation
    return model_rdm

