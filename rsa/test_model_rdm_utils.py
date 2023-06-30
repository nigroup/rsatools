from nose import tools
from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rsa.model_rdm_utils as model_rdm_utils
from rsa.mat_utils import get_triu_off_diag_flat, triu_off_diag_to_mat
from rsa.rdm_loader import RDMLoaderNPZ


def rand_rdm(n):
    rng = np.random.default_rng()
    rdm = np.zeros((n, n))
    rows, cols = np.triu_indices_from(rdm, k=1)
    vals = rng.uniform(0, 2, size=n * (n - 1) // 2)
    for idx, val in enumerate(vals):
        rdm[rows[idx], cols[idx]] = vals[idx]
    # rdm += rdm.T
    return get_triu_off_diag_flat(rdm)


class TestModelRDMUtilsInput2DMat:

    @classmethod
    def setup_class(cls):
        cls.dir_tmp = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.dir_tmp)
        pass

    @staticmethod
    def assert_rdm_shape(rdm):
        assert_equal(rdm.ndim, 2)
        assert_equal(rdm.shape[0], 3)
        assert_equal(rdm.shape[1], 3)

    def setup(self):
        in_rdm1 = np.array([[0, 1, 2],
                            [1, 0, 0.5],
                            [2, 0.5, 0]])
        self.fpath_in1 = os.path.join(self.dir_tmp, 'in1.npy')
        np.save(self.fpath_in1, in_rdm1)
        in_rdm2 = in_rdm1
        self.fpath_in2 = os.path.join(self.dir_tmp, 'in2.npy')
        np.save(self.fpath_in2, in_rdm2)
        in_rdm3 = np.array([[0, 2, 1],
                            [2, 0, 0.5],
                            [1, 0.5, 0]])
        self.fpath_in3 = os.path.join(self.dir_tmp, 'in3.npy')
        np.save(self.fpath_in3, in_rdm3)
        in_rdm4 = in_rdm3
        self.fpath_in4 = os.path.join(self.dir_tmp, 'in4.npy')
        np.save(self.fpath_in4, in_rdm4)

    def helper_calc_model_rdm(self, fp_in_rdms_1, fp_in_rdms_2):
        return model_rdm_utils.calc_model_rdm(fp_in_rdms_1,
                                              fp_in_rdms_2,
                                              do_disable_tqdm=True)

    def test_calc_model_rdm_size(self):
        for sz_in_rdm in range(3, 7):
            for num_rdms in range(2, 5):
                fp_in_rdms = []
                for rdm_idx in range(num_rdms):
                    rdm = rand_rdm(sz_in_rdm)
                    # print(rdm.shape)
                    fp = os.path.join(self.dir_tmp, 'rand_inrdm_%d-%d-%d.npy' % (sz_in_rdm, num_rdms, rdm_idx))
                    np.save(fp, rdm)
                    fp_in_rdms.append(fp)
                mrdm = self.helper_calc_model_rdm(fp_in_rdms, fp_in_rdms)
                # print(mrdm.shape)
                assert_equal(mrdm.size, num_rdms * (num_rdms - 1) // 2)

    def test_calc_model_rdm_identical(self):
        fp_list = [self.fpath_in1,
                   self.fpath_in2,
                   self.fpath_in3,
                   self.fpath_in4]
        mrdm = self.helper_calc_model_rdm(fp_list, fp_list)
        mrdm = triu_off_diag_to_mat(mrdm)
        for r, fp_r in enumerate(fp_list):
            for c, fp_c in enumerate(fp_list):
                if fp_r == fp_c:
                    assert_equal(mrdm[r, c], 0)
                    assert_equal(mrdm[c, r], 0)
                elif np.all(np.load(fp_r) == np.load(fp_c)):
                    # print(r,c)
                    assert_equal(mrdm[r, c], 0)
                    assert_equal(mrdm[c, r], 0)
                else:
                    # print(mrdm[r, c], r, c)
                    assert_true(0 <= mrdm[r, c] <= 2)

    def test_calc_model_rdm_values(self):
        fp_list = [self.fpath_in1,
                   self.fpath_in2,
                   self.fpath_in3,
                   self.fpath_in4]
        mrdm = self.helper_calc_model_rdm(fp_list, fp_list)
        mrdm = triu_off_diag_to_mat(mrdm)
        mrdm += mrdm.T
        for r, fp_r in enumerate(fp_list):
            rdm_r = np.load(fp_r)
            self.assert_rdm_shape(rdm_r)
            rdm_r = get_triu_off_diag_flat(rdm_r) if rdm_r.ndim > 1 else rdm_r
            for c, fp_c in enumerate(fp_list):
                rdm_c = np.load(fp_c)
                self.assert_rdm_shape(rdm_c)
                rdm_c = get_triu_off_diag_flat(rdm_c) if rdm_c.ndim > 1 else rdm_c
                corr = spearmanr(rdm_r, rdm_c).correlation
                assert_equal(mrdm[r, c], 1 - corr)


class TestModelRDMUtilsInputTriuVec(TestModelRDMUtilsInput2DMat):

    @staticmethod
    def assert_rdm_shape(rdm):
        assert_equal(rdm.ndim, 1)
        assert_equal(rdm.shape[0], 3 * (3 - 1) // 2)

    def setup(self):
        TestModelRDMUtilsInput2DMat.setup(self)
        in_rdm1 = np.array([[0, 1, 2],
                            [1, 0, 0.5],
                            [2, 0.5, 0]])
        for idx in range(1, 5):
            fp = os.path.join(self.dir_tmp, 'in%d.npy' % idx)
            in_rdm = np.load(fp)
            np.save(fp, get_triu_off_diag_flat(in_rdm))


class TestModelRDMUtilsInput2DMatNPZ(TestModelRDMUtilsInput2DMat):

    def helper_calc_model_rdm(self, fp_in_rdms_1, fp_in_rdms_2):
        # switch from npy to npz
        flist_npz_1 = []
        for fp in fp_in_rdms_1:
            fp_new = os.path.splitext(fp)[0] + '.npz'
            np.savez(fp_new, my_in_rdm=np.load(fp))
            flist_npz_1.append(fp_new)
        flist_npz_2 = []
        for fp in fp_in_rdms_2:
            fp_new = os.path.splitext(fp)[0] + '.npz'
            np.savez(fp_new, my_in_rdm=np.load(fp))
            flist_npz_2.append(fp_new)
        
        my_loader = RDMLoaderNPZ()
        my_loader.set_key('my_in_rdm')
        return model_rdm_utils.calc_model_rdm(flist_npz_1,
                                              flist_npz_2,
                                              do_disable_tqdm=True,
                                              loader=my_loader)


class TestModelRDMUtilsModelRDM2DF:

    def test_mrdm2df_rows(self):

        for n in range(2, 10):
            my_rdm = triu_off_diag_to_mat(rand_rdm(n))
            frames = [pd.DataFrame({'A': k, 'B': k * 10}, index=[0]) for k in range(n)]
            df = pd.concat(frames).reset_index(inplace=False, drop=True)
            df_rdm = model_rdm_utils.mrdm2df(my_rdm.copy(), df)
            assert_equal(len(df_rdm), n * (n - 1) // 2)

    def test_mrdm2df_col_names(self):

        for num_cols in range(1, 5):
            for n in range(2, 10):

                my_rdm = triu_off_diag_to_mat(rand_rdm(n))
                frames = [pd.DataFrame({'col%02d' % (col_idx + 1): col_idx * 10 + k for col_idx in range(num_cols)},
                                       index=[0]) for k in range(n)]
                df_meta = pd.concat(frames).reset_index(inplace=False, drop=True)
                df_rdm = model_rdm_utils.mrdm2df(my_rdm.copy(), df_meta)

                col_names_rdm = list(df_rdm.columns)
                assert_equal(len(col_names_rdm), len(list(df_meta.columns)) * 2 + 1)

                for col_name_meta in list(df_meta.columns):
                    assert_true(col_name_meta + '_x' in col_names_rdm)
                    assert_true(col_name_meta + '_y' in col_names_rdm)

                assert_true('dissim' in col_names_rdm)

    def test_mrdm2df_frames(self):

        for num_cols in range(1, 5):
            for n in range(2, 10):

                frames = [pd.DataFrame({'col%02d' % (col_idx + 1): col_idx * 10 + k for col_idx in range(num_cols)},
                                       index=[0]) for k in range(n)]
                df_meta = pd.concat(frames).reset_index(inplace=False, drop=True)
                my_rdm_matrix = triu_off_diag_to_mat(rand_rdm(n))
                df_rdm = model_rdm_utils.mrdm2df(my_rdm_matrix.copy(), df_meta)

                rows, cols = np.triu_indices_from(my_rdm_matrix, k=1)
                for idx, (r, c) in enumerate(zip(rows, cols)):
                    for col_name_meta in df_meta.columns:
                        assert_equal(df_meta.iloc[r][col_name_meta], df_rdm.iloc[idx][col_name_meta + '_x'])
                        assert_equal(df_meta.iloc[c][col_name_meta], df_rdm.iloc[idx][col_name_meta + '_y'])

    def test_mrdm2df_values_triu(self):

        for num_cols in range(1, 5):
            for n in range(2, 10):

                my_rdm_triu = rand_rdm(n)
                frames = [pd.DataFrame({'col%02d' % (col_idx + 1): col_idx * 10 + k for col_idx in range(num_cols)},
                                       index=[0]) for k in range(n)]
                df_meta = pd.concat(frames).reset_index(inplace=False, drop=True)
                df_rdm = model_rdm_utils.mrdm2df(triu_off_diag_to_mat(my_rdm_triu), df_meta)

                for idx, df_row in df_rdm.iterrows():
                    assert_equal(my_rdm_triu[idx], df_row['dissim'])

    def test_mrdm2df_values_mat(self):

        for num_cols in range(1, 5):
            for n in range(2, 10):

                my_rdm_matrix = triu_off_diag_to_mat(rand_rdm(n))
                frames = [pd.DataFrame({'col%02d' % (col_idx + 1): col_idx * 10 + k for col_idx in range(num_cols)},
                                       index=[0]) for k in range(n)]
                df_meta = pd.concat(frames).reset_index(inplace=False, drop=True)
                df_rdm = model_rdm_utils.mrdm2df(my_rdm_matrix.copy(), df_meta)

                rows, cols = np.triu_indices_from(my_rdm_matrix, k=1)
                for idx, df_row in df_rdm.iterrows():
                    assert_equal(my_rdm_matrix[rows[idx], cols[idx]], df_row['dissim'])
