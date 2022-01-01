from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import os
import tempfile
import shutil
import numpy as np

import rsa.input_rdm_utils as input_rdm_utils


class TestInputRDMUtils:

    @classmethod
    def setup_class(cls):
        cls.dir_tmp = tempfile.mkdtemp()
        pass

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.dir_tmp)
        pass

    def setup(self):
        self.activations = np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [3, 2, 1]])
        self.fpath_acts = os.path.join(self.dir_tmp, 'a.npy')
        np.save(self.fpath_acts, self.activations)

    def test_calc_and_save_input_rdm_return_path(self):
        for idx in range(5):
            fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm_%d.npy' % idx)
            fpath_dst_ret = input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg)
            assert_equal(fpath_dst_ret, fpath_dst_arg)

    def test_calc_input_rdm_return_path(self):
        fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm.npy')
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts)
        assert_equal(in_rdm.shape[0], 3)
        assert_equal(in_rdm.shape[1], 3)
