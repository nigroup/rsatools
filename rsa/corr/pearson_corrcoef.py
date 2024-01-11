import numpy as np
from rsa.corr.base_pearson_linear import BasePearsonLinear


class PearsonCorrcoef(BasePearsonLinear):
    def __init__(self, shape):
        self.num_samples = shape[0]

    def calculate(self, x):
        rho = np.corrcoef(x, x, rowvar=True)
        return rho[:self.num_samples, self.num_samples:]
