from nose import tools
from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import os
import tempfile
import shutil
import numpy as np

import rsa.model_rdm_utils as model_rdm_utils
from rsa.mat_utils import get_triu_off_diag_flat

def rand_rdm(n):
    rng = np.random.default_rng()
    rdm = np.zeros((n, n))
    rows, cols = np.triu_indices_from(rdm, k=1)
    vals = rng.uniform(0, 2, size=n*(n-1)//2)
    for idx, val in enumerate(vals):
        rdm[rows[idx], cols[idx]] = vals[idx]
    # rdm += rdm.T
    return get_triu_off_diag_flat(rdm)

class TestModelRDMUtils:

    @classmethod
    def setup_class(cls):
        cls.dir_tmp = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.dir_tmp)
        pass

    def setup(self):
        self.in_rdms = np.array([[[1, 2, 3],
                                      [4, 5, 6]],
                                     [[11, 12, 13],
                                      [14, 15, 16]],
                                     [[16, 15, 14],
                                      [13, 12, 11]]])
        self.fpath_in1 = os.path.join(self.dir_tmp, 'in1.npy')
        self.fpath_in2 = os.path.join(self.dir_tmp, 'in2.npy')
        self.fpath_in3 = os.path.join(self.dir_tmp, 'in3.npy')

    def test_calc_model_rdm_size(self):
        for sz_in_rdm in range(3,7):
            for num_rdms in range(2,5):
                fp_in_rdms = []
                for rdm_idx in range(num_rdms):
                    rdm = rand_rdm(sz_in_rdm)
                    # print(rdm.shape)
                    fp = os.path.join(self.dir_tmp, 'rand_inrdm_%d-%d-%d.npy' % (sz_in_rdm, num_rdms, rdm_idx))
                    np.save(fp, rdm)
                    fp_in_rdms.append(fp)
                mrdm = model_rdm_utils.calc_model_rdm(fp_in_rdms, fp_in_rdms, do_disable_tqdm=True)
                # print(mrdm.shape)
                assert_equal(mrdm.size, num_rdms*(num_rdms-1)//2)
