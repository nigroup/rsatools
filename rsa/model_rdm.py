import multiprocessing as mp
import errno
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from rsa.model_rdm_utils import calc_spearman_rank_corr_from_files, ENTRY_EMPTY
from rsa.rdm_loader import RDMLoaderNPY
import rsa.mat_utils as mutils


class ModelRDM:

    def __init__(self, fpath_list):
        # Oflloaded to loader, but should still check before apply if applicable with loader TODO
        # for fp in fpath_list:
        #     if not Path(fp).is_file():
        #         raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fp)
        self.fp_list = fpath_list
        self.num_rows = len(fpath_list)
        self.loader = RDMLoaderNPY()

    def set_loader(self, loader):
        self.loader = loader

    def dissimilarity(self, fp_row, fp_col, idx):

        if self.model_rdm_triu[idx] == ENTRY_EMPTY:
            idx, _, _, spearman = calc_spearman_rank_corr_from_files(fp_row, fp_col, -1, -1, idx, loader=self.loader)
            return idx, 1 - spearman.correlation

    def apply(self, processes=1, do_disable_tqdm=False):

        triu_rows, triu_cols = np.triu_indices(self.num_rows, k=1)
        self.model_rdm_triu = np.zeros((triu_rows.size,)) + ENTRY_EMPTY

        with mp.get_context("spawn").Pool(processes=processes) as pool:
            result = pool.starmap(self.dissimilarity,
                                  tqdm(
                                      [(self.fp_list[row],
                                        self.fp_list[col], idx)
                                       for idx, (row, col) in enumerate(zip(triu_rows, triu_cols))],
                                      total=len(triu_rows),
                                      disable=do_disable_tqdm),
                                  chunksize=10,
                                  )

        for idx, dissimilarity in result:
            self.model_rdm_triu[idx] = dissimilarity
        return self.model_rdm_triu

