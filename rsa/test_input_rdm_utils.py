from nose import tools
from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import os
import tempfile
import shutil
import numpy as np

import rsa.input_rdm_utils as input_rdm_utils


@tools.nottest
class BaseTestInputRDMUtils:

    @classmethod
    def setup_class(cls):
        cls.dir_tmp = tempfile.mkdtemp()
        pass

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.dir_tmp)
        pass

    def test_calc_input_rdm_shape(self):
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts)
        # print(in_rdm)
        assert_equal(in_rdm.shape[0], self.num_samples)
        assert_equal(in_rdm.shape[1], self.num_samples)

    def test_calc_input_rdm_values_diag(self):
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts)
        # print(in_rdm)
        for r_idx, row in enumerate(in_rdm):
            for c_idx, el in enumerate(row):
                if r_idx == c_idx:
                    assert_equal(el, 0.0)


@tools.istest
class TestInputRDMUtils(BaseTestInputRDMUtils):

    def setup(self):
        self.num_samples = 4
        self.activations = np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [3, 2, 1],
                                     [6, 5, 4]])
        self.fpath_acts = os.path.join(self.dir_tmp, 'a.npy')
        np.save(self.fpath_acts, self.activations)

    def test_calc_and_save_input_rdm_return_path(self):
        for idx in range(5):
            fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm_%d.npy' % idx)
            fpath_dst_ret = input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg)
            assert_equal(fpath_dst_ret, fpath_dst_arg)

    def test_calc_input_rdm_values_offdiag(self):
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts)
        assert_equal(in_rdm[0, 1], 0)
        assert_equal(in_rdm[0, 2], 2)
        assert_equal(in_rdm[0, 3], 2)
        assert_equal(in_rdm[1, 2], 2)
        assert_equal(in_rdm[1, 3], 2)


@tools.istest
class TestInputRDMUtils2D(BaseTestInputRDMUtils):

    def setup(self):
        self.num_samples = 3
        self.activations = np.array([[[1, 2, 3],
                                      [4, 5, 6]],
                                     [[11, 12, 13],
                                      [14, 15, 16]],
                                     [[16, 15, 14],
                                      [13, 12, 11]]])
        self.fpath_acts = os.path.join(self.dir_tmp, 'a2d.npy')
        np.save(self.fpath_acts, self.activations)

    def test_calc_input_rdm_values_offdiag_2D_input(self):
        activations = np.array([[[1, 2, 3],
                                 [4, 5, 6]],
                                [[11, 12, 13],
                                 [14, 15, 16]],
                                [[16, 15, 14],
                                 [13, 12, 11]]])
        fpath_acts_2d = os.path.join(self.dir_tmp, 'a2d.npy')
        np.save(fpath_acts_2d, activations)

        in_rdm = input_rdm_utils.calc_input_rdm(fpath_acts_2d)
        assert_equal(in_rdm[0, 1], 0)
        assert_equal(in_rdm[0, 2], 2)
        assert_equal(in_rdm[1, 2], 2)
