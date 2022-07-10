''' solver '''
# pylint: disable=no-name-in-module
from copy import deepcopy
from tensorflow.keras.callbacks import LearningRateScheduler
from common import logging
from common.solver import BaseSolver
from common.callbacks import TrainDataShuffleCallback, ValidationWithEMACallback
from . import config
from .arch import arch


class Solver(BaseSolver):
    ''' Solver '''
    def __init__(self, train_data, val_data, resume_path=None):
        super().__init__(config, arch, train_data, val_data, resume_path)

    def build_model(self):
        ''' build_model (no suport resume) '''
        self.model = self.arch(**self.config.model)
        self.model_ema = deepcopy(self.model)
        logging.info(f'Create model successfully! Params: [{self.model.count_params() / 1e3:.2f}]K')

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]
