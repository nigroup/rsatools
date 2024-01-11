class BasePearsonLinear:
    def __init__(self, shape):
        self.num_samples = shape[0]

    def calculate(self, x):
        raise NotImplementedError("This is the base class.")