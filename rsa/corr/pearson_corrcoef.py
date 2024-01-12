import numpy as np
from rsa.corr.base_pearson_linear import BasePearsonLinear


class PearsonCorrcoef(BasePearsonLinear):
    def __init__(self, shape):
        self.num_samples = shape[0]

    def calculate(self, x):
        rho = np.corrcoef(x, rowvar=True)
        assert(rho.shape[0] == x.shape[0])
        assert(rho.shape[1] == x.shape[0])
        return rho
