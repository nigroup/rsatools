from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal

import numpy as np

import rsa.rdm_utils as rdm_utils


def rand_rdm(n):
    mat = np.random.uniform(0, 2, size=(n, n))
    mat = mat.T + mat
    for r in range(n):
        mat[r, r] = 0
    return mat


class TestRDMUtils:

    def test_triu_off_diag_vec_to_rdm(self):
        for n in range(2, 15):
            rdm = rand_rdm(n)
            triu1vec = rdm[np.triu_indices(rdm.shape[0], k=1)]
            # print(triu1vec.shape)
            rdm_new = rdm_utils.triu_off_diag_vec_to_rdm(triu1vec)
            assert_true(np.all(rdm == rdm_new))
            # print(in_rdm_new)

