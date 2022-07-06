''' solver '''
# pylint: disable=no-name-in-module
from common.solver import BaseSolver
from . import config
from .arch import arch


class Solver(BaseSolver):
    ''' Solver '''
    def __init__(self, train_data, val_data, resume_path=None):
        super().__init__(config, arch, train_data, val_data, resume_path)
