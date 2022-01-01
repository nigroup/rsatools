from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import numpy as np

import rsa.mat_utils as mat_utils


class TestMatUtils:

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup(self):
        pass

    def test_get_triu_off_diag_flat_num_elements(self):
        for n in range(2,15):
            triu = mat_utils.get_triu_off_diag_flat(np.ones((n, n)))
            assert_equal(len(triu.shape), 1)
            assert_equal(triu.shape[0], n*(n-1)/2)


    def test_get_triu_off_diag_flat_vals(self):
        for n in range(2,15):
            mat = np.random.rand(n, n)
            triu = mat_utils.get_triu_off_diag_flat(mat)
            idx = 0
            for r_idx in range(0, n):
                for c_idx in range(r_idx+1, n):
                    assert_equal(mat[r_idx, c_idx], triu[idx])
                    idx += 1



