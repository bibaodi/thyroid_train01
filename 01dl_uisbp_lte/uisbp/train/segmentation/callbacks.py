import numpy as np
from tensorflow.keras.callbacks import Callback, LambdaCallback
from tensorflow.keras import backend as K


class SGDR(Callback):

    def __init__(self, nb, cycle_mult=1, base_lr=0.001, max_lr=0.006):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.nb = nb
        self.cycle_mult = cycle_mult
        self.history = {}

    def calc_lr(self):
        cos_out = np.cos(np.pi*(self.cycle_iter) / self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
        return self.base_lr + (self.max_lr - self.base_lr) / 2 * cos_out

    def on_train_begin(self, logs={}):
        logs = logs or {}

        self.cycle_iter = 0
        self.iterations = 0

        if self.iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.calc_lr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.calc_lr())
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

