import errno
import os
from pathlib import Path
import numpy as np


class RDMLoaderNPY:

    def __init__(self):
        self.fpath = None

    def _check_path(self):

        if not Path(self.fpath).is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.fpath)

    def set_path(self, fpath):

        self.fpath = fpath
        self._check_path()

    def _load_rdm_from_file(self):
        rdm = np.load(self.fpath)
        return rdm

    def load(self):
        self._check_path()
        rdm = self._load_rdm_from_file()
        return rdm


class RDMLoaderNPZ(RDMLoaderNPY):

    def __init__(self):

        super().__init__()
        self.key = None

    def set_key(self, key):
        self.key = key

    def _load_rdm_from_file(self):

        h = np.load(self.fpath)
        return h[self.key]


class RDMLoaderInMemory(RDMLoaderNPY):

    def _check_path(self):
        # Nothing to do? Or check type for ndarray?
        pass

    def _load_rdm_from_file(self):
        return self.fpath
