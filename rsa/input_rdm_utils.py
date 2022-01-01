import os
import numpy as np

def calc_input_rdm(fpath_src_activations, key=""):
    """
    Calculate Input RDM

    :param fpath_src_activations: path to .npy file with activations
    :return: Input RDM
    """
    _, ext = os.path.splitext(fpath_src_activations)
    if ext.endswith('npy'):
        acts = np.load(fpath_src_activations)
    elif ext.endswith('h5'):
        import tables as tb
        with tb.File(fpath_src_activations, "r") as hf:
            acts = hf[key]
    num_samples = acts.shape[0]
    acts = acts.reshape(num_samples, -1)
    in_rdm = (1 - np.corrcoef(acts, acts, rowvar=True))[:num_samples, num_samples:]
    return in_rdm


def calc_and_save_input_rdm(fpath_src_activations, fpath_dst):
    in_rdm = calc_input_rdm(fpath_src_activations)
    np.save(fpath_dst, in_rdm)
    return fpath_dst
