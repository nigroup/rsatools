import numpy as np


def calc_input_rdm(fpath_src_activations):
    acts = np.load(fpath_src_activations)
    num_samples, _ = acts.shape
    in_rdm = (1 - np.corrcoef(acts, acts, rowvar=True))[:num_samples, num_samples:]
    return in_rdm


def calc_and_save_input_rdm(fpath_src_activations, fpath_dst):
    in_rdm = calc_input_rdm(fpath_src_activations)
    np.save(fpath_dst, in_rdm)
    return fpath_out
