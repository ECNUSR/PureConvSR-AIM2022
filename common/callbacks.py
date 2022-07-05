''' callbacks '''
# pylint: disable=no-name-in-module
from os import path as osp
import pickle
import tqdm
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from common import logging
from common.metrics import calc_psnr


class TrainDataShuffleCallback(Callback):
    ''' TrainDataShuffleCallback '''
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs=None):
        self.train_data.shuffle()


class ValidationCallback(Callback):
    ''' ValidationCallback '''
    def __init__(self, trial_name, val_data, state, interval=1):
        super().__init__()
        self.trial_name = trial_name
        self.val_data = val_data
        self.best_epoch = state['best_epoch']
        self.best_psnr = state['best_psnr']
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval != 0:
            return

        # validate
        psnr = 0.0
        for lr, hr in tqdm.tqdm(self.val_data):
            sr = self.model(lr)
            sr_numpy = K.eval(sr)
            psnr += calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
        psnr = psnr / len(self.val_data)
        loss = logs['loss']

        # save best status
        if psnr >= self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch
            self.model.save(osp.join('experiments', self.trial_name, 'best_status'), overwrite=True, include_optimizer=True, save_format='tf')
            with open(osp.join('experiments', self.trial_name, 'state.pkl'), 'wb') as f:
                pickle.dump({
                    'current_epoch': epoch,
                    'best_epoch': self.best_epoch,
                    'best_psnr': self.best_psnr
                }, f)

        logging.info(f'Epoch: {epoch} | PSNR: {psnr:.4f} | Loss: {loss:.4f} | lr: {K.get_value(self.model.optimizer.lr):.2e} | Best_PSNR: {self.best_psnr:.4f} in Epoch [{self.best_epoch}]')

        # record tensorboard
        logging.tb_log(epoch, loss=loss, psnr=psnr)
        logging.report(f'epoch: {epoch}, loss: {loss:.4f}, psnr: {psnr:.4f}')
