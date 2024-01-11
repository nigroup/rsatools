import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
import rsa.mat_utils as mutils


ENTRY_EMPTY_INPUT_RDM = -999
class InputRDM:

    def __init__(self, acts):
        # Oflloaded to loader, but should still check before apply if applicable with loader TODO
        # for fp in fpath_list:
        #     if not Path(fp).is_file():
        #         raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fp)
        self.acts = acts
        self.num_rows = len(self.acts)

    def set_loader(self, loader):
        self.loader = loader

    def dissimilarity(self, act_row, act_col, dst_idx):

        if self.input_rdm_triu[dst_idx] == ENTRY_EMPTY_INPUT_RDM:
            rho, _ = pearsonr(act_row, act_col)
            return dst_idx, 1 - rho

    def apply(self, processes=1, do_disable_tqdm=False):

        triu_rows, triu_cols = np.triu_indices(self.num_rows, k=1)
        self.input_rdm_triu = np.zeros((triu_rows.size,)) + ENTRY_EMPTY_INPUT_RDM

        with mp.get_context("spawn").Pool(processes=processes) as pool:
            result = pool.starmap(self.dissimilarity,
                                  tqdm(
                                      [(self.acts[row],
                                        self.acts[col], idx)
                                       for idx, (row, col) in enumerate(zip(triu_rows, triu_cols))],
                                      total=len(triu_rows),
                                      disable=do_disable_tqdm),
                                  chunksize=10,
                                  )

        for idx, dissimilarity in result:
            self.input_rdm_triu[idx] = dissimilarity
        return self.input_rdm_triu

