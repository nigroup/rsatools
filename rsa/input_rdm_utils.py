import os
import numpy as np

from rsa.mat_utils import get_triu_off_diag_flat


def calc_input_rdm(fpath_src_activations, key=""):
    """
    Calculate Input RDM

    :param fpath_src_activations: path to .npy or HDF5 file with activations
    :param key: key or field (for HDF5)
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
    acts = acts.reshape(num_samples, -1)
    in_rdm = (1 - np.corrcoef(acts, acts, rowvar=True))[:num_samples, num_samples:]
    return in_rdm


def calc_and_save_input_rdm(fpath_src_activations, fpath_dst, key="", do_triu=True):
    in_rdm = calc_input_rdm(fpath_src_activations, key=key)
    if do_triu:
        in_rdm = get_triu_off_diag_flat(in_rdm)
    np.save(fpath_dst, in_rdm)
    return fpath_dst


def get_input_rdm_flat_from_file(fpath):
    rdm = np.load(fpath)
    if rdm.ndim == 1:
        return rdm
    elif rdm.ndim == 2 and 1 in rdm.shape:
        return rdm.flatten()
    elif rdm.ndim > 2:
        raise ValueError("File does not contain a 2D matrix (%s)" % fpath)
    else:
        return get_triu_off_diag_flat(rdm)
