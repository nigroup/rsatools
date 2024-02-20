import multiprocessing as mp
import errno
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from rsa.model_rdm_utils import calc_spearman_rank_corr_from_files, ENTRY_EMPTY
from rsa.rdm_loader import RDMLoaderNPY
import rsa.mat_utils as mutils
from rsa.cache.rdm_cache import RDMCache
from rsa.model_rdm import ModelRDM


class ModelRDMCached(ModelRDM):

    def __init__(self, fpath_list, fp_cache):
        super().__init__(fpath_list)
        self.cache = None
        self.fp_cache = fp_cache
        self.load_cache(self.fp_cache)

    def load_cache(self, fp_cache):
        self.cache = RDMCache()
        self.cache.load_from_file(fp_cache)

    def save_cache(self, fp_cache):

        self.cache.save_to_file(fp_cache)

    def _init_model_rdm_triu(self):
        triu_rows, triu_cols = self.get_triu_rows_cols()
        self.model_rdm_triu = np.zeros((triu_rows.size,)) + ENTRY_EMPTY

        for idx, (row, col) in enumerate(zip(triu_rows, triu_cols)):
            fp_row = self.fp_list[row]
            fp_col = self.fp_list[col]

            self.model_rdm_triu[idx] = self.cache.get(fp_row, fp_col, ENTRY_EMPTY)

    def apply(self, processes=1, do_disable_tqdm=False):

        self.model_rdm_triu = super().apply(processes=processes, do_disable_tqdm=do_disable_tqdm)

        triu_rows, triu_cols = self.get_triu_rows_cols()
        for idx, (row, col) in enumerate(zip(triu_rows, triu_cols)):
            fp_row = self.fp_list[row]
            fp_col = self.fp_list[col]
            dissimilarity_value = self.model_rdm_triu[idx]
            self.cache.add(fp_row, fp_col, float(dissimilarity_value))

        self.cache.save_to_file(self.fp_cache)

        return self.model_rdm_triu

