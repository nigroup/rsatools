import os
import numpy as np

from rsa.mat_utils import get_triu_off_diag_flat
from rsa.rdm_loader import RDMLoaderNPY
from rsa.corr.pearson_corrcoef import PearsonCorrcoef
from rsa.input_rdm import InputRDM


def calc_input_rdm(fpath_src_activations, key="",
                   do_keep_mem_low=False,
                   num_processes=None):
    """
    Calculate Input RDM

    :param fpath_src_activations: path to .npy or HDF5 file with activations
    :param key: key or field (for HDF5)
    :param do_keep_mem_low: use a different routine for calculating the pearson linear correlation, slower but with lower memory footprint
    :param num_processes: count of threads for parallel processing, only used in the case of do_keep_mem_low==True
    :return: Input RDM
    """
    _, ext = os.path.splitext(fpath_src_activations)
    if ext.endswith('npy'):
        acts = np.load(fpath_src_activations)
    elif ext.endswith('h5'):
        import tables as tb
        with tb.File(fpath_src_activations, "r") as hf:
            acts = hf.get_node('/')[key][:]

    num_samples = acts.shape[0]
    if do_keep_mem_low:
        ir = InputRDM(acts.reshape(num_samples, -1))
        in_rdm = ir.apply(processes=1 if num_processes is None else num_processes,
                          do_disable_tqdm=True)
        from rsa.rdm_utils import triu_off_diag_vec_to_rdm
        in_rdm = triu_off_diag_vec_to_rdm(in_rdm)
    else:
        # acts = acts.reshape(num_samples, -1)
        # in_rdm = (1 - np.corrcoef(acts, acts, rowvar=True))[:num_samples, num_samples:]
        p = PearsonCorrcoef(acts.shape)
        in_rdm = 1 - p.calculate(acts.reshape(num_samples, -1))

    return in_rdm


def calc_and_save_input_rdm(fpath_src_activations, fpath_dst, key="", do_triu=True,
                            do_keep_mem_low=False,
                            num_processes=None):
    in_rdm = calc_input_rdm(fpath_src_activations, key=key,
                            do_keep_mem_low=do_keep_mem_low,
                            num_processes=num_processes)
    if do_triu:
        in_rdm = get_triu_off_diag_flat(in_rdm)
        # print(in_rdm.shape)
        # if in_rdm.ndim == 2:
        #     in_rdm = get_triu_off_diag_flat(in_rdm)
        # elif in_rdm.ndim > 2:
        #     raise ValueError("Shape of input rdm is %s" % (in_rdm.shape))
    np.save(fpath_dst, in_rdm)
    return fpath_dst


def get_input_rdm_flat_from_file(fpath, loader=None):
    if loader is None:
        loader = RDMLoaderNPY()
    loader.set_path(fpath)
    rdm = loader.load()

    if rdm.ndim == 1:
        return rdm
    elif rdm.ndim == 2 and 1 in rdm.shape:
        return rdm.flatten()
    elif rdm.ndim > 2:
        raise ValueError("File does not contain a 2D matrix (%s)" % fpath)
    else:
        return get_triu_off_diag_flat(rdm)
