from nose import tools
from nose.tools import assert_equal, \
    assert_true, assert_false, \
    assert_raises, assert_list_equal
import os
import yaml

from rsa.model_rdm_cached import ModelRDMCached

from rsa.test_model_rdm import TestModelRDMInput2DMat, \
    TestModelRDMInput2DMatNPZ, \
    TestModelRDMInput2DMaInMemory


def helper_compare_mrdm_cache(flist, mrdm, cache):
    assert_equal(len(cache.cache_dict.keys()), mrdm.size)

    idx = 0
    for row in range(len(flist)):
        for col in range(row + 1, len(flist)):
            fp_row = flist[row]
            fp_col = flist[col]
            assert_equal(cache.get(fp_row, fp_col, 123), mrdm[idx])
            idx += 1


def helper_calc_model_rdm_with_cache(flist, fp_cache):
    with open(fp_cache, 'w') as h:
        h.write(yaml.dump({}))

    m = ModelRDMCached(flist, fp_cache)
    mrdm = m.apply(do_disable_tqdm=True)

    with open(fp_cache, 'r') as h:
        cache_dict = yaml.safe_load(h)

    helper_compare_mrdm_cache(flist, mrdm, m.cache)

    return mrdm


class TestModelRDMCachedInput2DMatEmptyCache(TestModelRDMInput2DMat):

    def helper_calc_model_rdm(self, flist):
        fp_cache = os.path.join(self.dir_tmp, 'my_cache.yml')
        mrdm = helper_calc_model_rdm_with_cache(flist, fp_cache)

        return mrdm


class TestModelRDMCachedInput2DMatNPZEmptyCache(TestModelRDMInput2DMatNPZ):

    def helper_calc_model_rdm(self, flist):
        fp_cache = os.path.join(self.dir_tmp, 'my_cache.yml')
        mrdm = helper_calc_model_rdm_with_cache(flist, fp_cache)

        return mrdm


class TestModelRDMCachedInput2DMaInMemoryEmptyCache(TestModelRDMInput2DMaInMemory):

    def helper_calc_model_rdm(self, flist):
        fp_cache = os.path.join(self.dir_tmp, 'my_cache.yml')
        mrdm = helper_calc_model_rdm_with_cache(flist, fp_cache)

        return mrdm
