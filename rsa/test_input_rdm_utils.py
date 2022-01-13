from nose import tools
from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import os
import tempfile
import shutil
import numpy as np
import tables as tb

import rsa.input_rdm_utils as input_rdm_utils


def save_to_h5(fpath_dst, values, key):
    with tb.File(fpath_dst, "w") as f:
        f.create_earray(f.root, key,
                        atom=tb.Atom.from_dtype(values.dtype),
                        obj=values,
                        title="activation values")


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
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts, key=self.key)
        # print(in_rdm)
        assert_equal(in_rdm.shape[0], self.num_samples)
        assert_equal(in_rdm.shape[1], self.num_samples)

    def test_calc_input_rdm_values_diag(self):
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts, key=self.key)
        # print(in_rdm)
        for r_idx, row in enumerate(in_rdm):
            for c_idx, el in enumerate(row):
                if r_idx == c_idx:
                    assert_equal(el, 0.0)

    def test_calc_and_save_input_rdm_return_path(self):
        for idx in range(5):
            fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm_%d.npy' % idx)
            fpath_dst_ret = input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg, key=self.key)
            assert_equal(fpath_dst_ret, fpath_dst_arg)

    def test_calc_and_save_input_rdm_default_dims(self):

        fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg, key=self.key)
        in_rdm = np.load(fpath_dst_arg)
        assert_equal(in_rdm.ndim, 1)
        assert_equal(len(in_rdm.shape), 1)
        assert_equal(in_rdm.shape[0], self.num_samples * (self.num_samples - 1) / 2)
        assert_equal(in_rdm.size, self.num_samples * (self.num_samples - 1) / 2)

    def test_calc_and_save_input_rdm_do_triu_true_dims(self):

        fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg, key=self.key, do_triu=True)
        in_rdm = np.load(fpath_dst_arg)
        assert_equal(in_rdm.ndim, 1)
        assert_equal(len(in_rdm.shape), 1)
        assert_equal(in_rdm.shape[0], self.num_samples * (self.num_samples - 1) / 2)
        assert_equal(in_rdm.size, self.num_samples * (self.num_samples - 1) / 2)

    def test_calc_and_save_input_rdm_do_triu_false_dims(self):

        fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg, key=self.key, do_triu=False)
        in_rdm = np.load(fpath_dst_arg)
        assert_equal(in_rdm.ndim, 2)
        assert_equal(in_rdm.shape[0], self.num_samples)
        assert_equal(in_rdm.shape[1], self.num_samples)

    def test_calc_and_save_input_rdm_do_triu_false_values_diag(self):

        fpath_dst_arg = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst_arg, key=self.key, do_triu=False)
        in_rdm = np.load(fpath_dst_arg)
        for r_idx, row in enumerate(in_rdm):
            for c_idx, el in enumerate(row):
                if r_idx == c_idx:
                    assert_equal(el, 0.0)


@tools.istest
class TestInputRDMUtils(BaseTestInputRDMUtils):

    def setup(self):
        self.key = ""
        self.num_samples = 4
        self.activations = np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [3, 2, 1],
                                     [6, 5, 4]])
        self.fpath_acts = os.path.join(self.dir_tmp, 'a.npy')
        np.save(self.fpath_acts, self.activations)

    def test_calc_input_rdm_values_offdiag(self):
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts, key=self.key)
        assert_equal(in_rdm[0, 1], 0)
        assert_equal(in_rdm[0, 2], 2)
        assert_equal(in_rdm[0, 3], 2)
        assert_equal(in_rdm[1, 2], 2)
        assert_equal(in_rdm[1, 3], 2)

    def test_calc_and_save_input_rdm_do_triu_false_values_offdiag(self):
        fpath_dst = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst, key=self.key, do_triu=False)
        in_rdm = np.load(fpath_dst)
        assert_equal(in_rdm[0, 1], 0)
        assert_equal(in_rdm[0, 2], 2)
        assert_equal(in_rdm[0, 3], 2)
        assert_equal(in_rdm[1, 2], 2)
        assert_equal(in_rdm[1, 3], 2)

    def test_calc_and_save_input_rdm_do_triu_true_values_offdiag(self):
        fpath_dst = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst, key=self.key, do_triu=True)
        in_rdm = np.load(fpath_dst)

        assert_equal(in_rdm.size, self.num_samples * (self.num_samples - 1) / 2)
        assert_equal(in_rdm[0], 0)
        assert_equal(in_rdm[1], 2)
        assert_equal(in_rdm[2], 2)
        assert_equal(in_rdm[3], 2)
        assert_equal(in_rdm[4], 2)


@tools.istest
class TestInputRDMUtilsHDF5(TestInputRDMUtils):

    def setup(self):
        self.key = 'values'
        self.num_samples = 4
        self.activations = np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [3, 2, 1],
                                     [6, 5, 4]])
        self.fpath_acts = os.path.join(self.dir_tmp, 'a.h5')
        save_to_h5(self.fpath_acts, self.activations, self.key)


@tools.istest
class TestInputRDMUtils2D(BaseTestInputRDMUtils):

    def setup(self):
        self.key = ""
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
        in_rdm = input_rdm_utils.calc_input_rdm(self.fpath_acts, key=self.key)
        assert_equal(in_rdm[0, 1], 0)
        assert_equal(in_rdm[0, 2], 2)
        assert_equal(in_rdm[1, 2], 2)

    def test_calc_and_save_input_rdm_do_triu_false_values_offdiag(self):
        fpath_dst = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst, key=self.key, do_triu=False)
        in_rdm = np.load(fpath_dst)
        assert_equal(in_rdm[0, 1], 0)
        assert_equal(in_rdm[0, 2], 2)
        assert_equal(in_rdm[1, 2], 2)

    def test_calc_and_save_input_rdm_do_triu_true_values_offdiag(self):
        fpath_dst = os.path.join(self.dir_tmp, 'inrdm.npy')
        input_rdm_utils.calc_and_save_input_rdm(self.fpath_acts, fpath_dst, key=self.key, do_triu=True)
        in_rdm = np.load(fpath_dst)

        assert_equal(in_rdm.size, self.num_samples * (self.num_samples - 1) / 2)
        assert_equal(in_rdm[0], 0)
        assert_equal(in_rdm[1], 2)
        assert_equal(in_rdm[2], 2)


@tools.istest
class TestInputRDMUtils2DHDF5(TestInputRDMUtils2D):

    def setup(self):
        self.key = "values"
        self.num_samples = 3
        self.activations = np.array([[[1, 2, 3],
                                      [4, 5, 6]],
                                     [[11, 12, 13],
                                      [14, 15, 16]],
                                     [[16, 15, 14],
                                      [13, 12, 11]]])
        self.fpath_acts = os.path.join(self.dir_tmp, 'a2d.h5')
        save_to_h5(self.fpath_acts, self.activations, self.key)
