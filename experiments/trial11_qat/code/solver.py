''' solver '''
# pylint: disable=no-name-in-module
from tensorflow.keras.callbacks import LearningRateScheduler
from common.solver import BaseSolver, BaseQuantSolver
from common.callbacks import TrainDataShuffleCallback, ValidationWithEMACallback
from . import config, qat_config
from .arch import arch


class Solver(BaseSolver):
    ''' Solver '''
    def __init__(self, train_data, val_data, resume_path=None):
        super().__init__(config, arch, train_data, val_data, resume_path)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]


class QuantSolver(BaseQuantSolver):
    ''' QuantSolver '''
    def __init__(self, train_data, val_data, resume_path=None, qat_path=None):
        super().__init__(qat_config, train_data, val_data, resume_path, qat_path)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]
