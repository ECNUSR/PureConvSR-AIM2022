''' callbacks '''
# pylint: disable=no-name-in-module
from os import path as osp
import pickle
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
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

        logging.info(f'Epoch: {epoch + 1} | PSNR: {psnr:.4f} | Loss: {loss:.4f} | lr: {K.get_value(self.model.optimizer.lr):.2e} | Best_PSNR: {self.best_psnr:.4f} in Epoch [{self.best_epoch + 1}]')

        # record tensorboard
        logging.tb_log(epoch, loss=loss, psnr=psnr)
        logging.report(f'Validation [epoch: {epoch + 1}]\n\t# loss: {loss:.4f}\n\t# psnr: {psnr:.4f} Best: {self.best_psnr:.4f} @ {self.best_epoch + 1} epoch')


class SimulationResidual(Callback):
    ''' SimulationResidual '''
    def on_batch_end(self, batch, logs=None):
        ''' on_batch_end '''
        for layer in self.model.layers:
            if 'simulation_residual' in layer.name:
                weight = layer.weights[0].numpy()
                channel = weight.shape[2] - 3
                weight2 = np.zeros((3, 3, channel + 3, 27))
                weight2[:, :, :channel, :] = weight[:, :, :channel, :]
                weight2[:, :, channel:, :] = 0
                for j in range(27):
                    weight2[1, 1, channel + j % 3, j] = 1
                layer.weights[0].assign(weight2)


class ValidationWithEMACallback(Callback):
    """ EMACallback """
    def __init__(self, trial_name, val_data, state, interval=1, decay=0.999):
        super().__init__()
        self.trial_name = trial_name
        self.val_data = val_data
        self.best_epoch = state['best_epoch']
        self.best_psnr = state['best_psnr']
        self.interval = interval
        self.decay = decay

    def on_train_begin(self, logs=None):
        self.sym_trainable_weights = self.model.trainable_weights
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in self.sym_trainable_weights}

    def on_batch_end(self, batch, logs=None):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval != 0:
            return

        # validate
        psnr = 0.0
        model = self._make_mv_model()
        for lr, hr in tqdm.tqdm(self.val_data):
            sr = model(lr)
            sr_numpy = K.eval(sr)
            psnr += calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
        psnr = psnr / len(self.val_data)
        loss = logs['loss']

        # save best status
        if psnr >= self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch
            model.save(osp.join('experiments', self.trial_name, 'best_status'), overwrite=True, include_optimizer=True, save_format='tf')
            with open(osp.join('experiments', self.trial_name, 'state.pkl'), 'wb') as f:
                pickle.dump({
                    'current_epoch': epoch,
                    'best_epoch': self.best_epoch,
                    'best_psnr': self.best_psnr
                }, f)

        logging.info(f'Epoch: {epoch + 1} | PSNR: {psnr:.4f} | Loss: {loss:.4f} | lr: {K.get_value(model.optimizer.lr):.2e} | Best_PSNR: {self.best_psnr:.4f} in Epoch [{self.best_epoch + 1}]')

        # record tensorboard
        logging.tb_log(epoch, loss=loss, psnr=psnr)
        logging.report(f'Validation [epoch: {epoch + 1}]\n\t# loss: {loss:.4f}\n\t# psnr: {psnr:.4f} Best: {self.best_psnr:.4f} @ {self.best_epoch + 1} epoch')

    def on_train_end(self, logs=None):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self):
        self.model.save('/tmp/mobial_sr', overwrite=True, save_format='tf')
        model2 = load_model('/tmp/mobial_sr', custom_objects={'tf': tf})
        for sym_weight in model2.trainable_weights:
            K.set_value(sym_weight, self.mv_trainable_weights_vals[sym_weight.name])
        return model2
