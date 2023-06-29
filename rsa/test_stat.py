from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal

import numpy as np
import rsa.rdm_utils as rdm_utils
import rsa.stat as st


def rand_rdm(n):
    mat = np.random.uniform(0, 2, size=(n, n))
    mat = (mat.T + mat) / 2.
    for r in range(n):
        mat[r, r] = 0
    return mat


class TestStat:

    def test_permutate_rdm_2_cols(self):
        for i in range(5):
            rdm = rand_rdm(2)
            triu1vec = rdm[np.triu_indices(rdm.shape[0], k=1)]
            rdm_new = st.permute_rdm(triu1vec)
            assert_true(np.all(rdm == rdm_new))

    def test_permutate_rdm(self):
        for n in range(3, 5):
            rdm = rand_rdm(n)
            # print(rdm)
            triu1vec = rdm[np.triu_indices(rdm.shape[0], k=1)]
            # print(triu1vec.shape)
            rdm_new = st.permute_rdm(triu1vec)

            # print('new')
            # print(rdm_new)
            assert_true(np.all(rdm.shape == rdm_new.shape))

            assert_false(np.all(rdm == rdm_new))
