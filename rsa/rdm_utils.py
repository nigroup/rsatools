from rsa.mat_utils import triu_off_diag_to_mat


def triu_off_diag_vec_to_rdm(triu1vec):
    in_rdm = triu_off_diag_to_mat(triu1vec)
    in_rdm += in_rdm.T
    return in_rdm
