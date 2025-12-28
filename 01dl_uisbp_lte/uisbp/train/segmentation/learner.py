import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model

from .callbacks import SGDR


class Learner:
    """
    Learner class based on that of the fast.ai library.
    """
    def __init__(self,
                 model,
                 data,
                 opt_fn=Adam(lr=1e-4),
                 loss='binary_crossentropy',
                 outdir=f'results{os.sep}'):
        """
        Initializes Learner.

        Args:
            model (Keras `Model`): Input Keras model (does not need to be compiled)
            data (`DataSet` instance): Input `DataSet` instance
            opt_fn: Keras Optimzer function
            loss: Keras loss function (either string (if Keras builtin) or function)
            outdir (str): Root output directory for results
        """

        self.data = data
        self.model = model
        self.opt_fn = opt_fn
        self.loss = loss
        self.outdir = outdir
        self.callbacks = None
        self.sched = None
        self.nepoch = 0
        os.makedirs(f'{outdir}', exist_ok=True)

    def fit(self,
            lrs,
            n_cycle,
            cycle_len=None,
            cycle_mult=1,
            metrics=(),
            callbacks=None,
            monitor=('val_mean_dice_coef', 'max'),
            **kwargs):
        """
        Fit model over `n_cycle` cyles

        Args:
            lrs (float): Initial learning rate
            n_cycle (int): Number of cycles to fit (if cycle_len=None) this is just the number of epochs
            cycle_len (int): Number of SGDR cycles before learning rate is reset
            cycle_mult (int): Cycle multiplier before learning rate is reset
            metrics (tuple): Tuple of metrics to keep track of while training
            callbacks (list): List of Keras callbacks
            monitor (tuple): Tuple of (metric_monitor_name, mode) where metric_monitor is, for e.g. 'val_mean_dice_coef' and mode is 'max'
            kwargs (dict): Additional kwargs to pass to Keras fit_generator
        """

        self.callbacks = callbacks or []

        # add standard callbacks
        checkpointer = ModelCheckpoint(f'{self.outdir}{os.sep}weights.h5',
                                       verbose=0,
                                       save_best_only=True,
                                       monitor=monitor[0],
                                       mode=monitor[1])
        self.callbacks.append(checkpointer)

        csv_logger = CSVLogger(f'{self.outdir}{os.sep}training_log.csv', append=True)
        self.callbacks.append(csv_logger)

        if cycle_len:
            epochs = cycle_len * n_cycle if cycle_mult == 1 else math.ceil(
                cycle_len * (1 - cycle_mult**n_cycle) / (1 - cycle_mult))
            self.sched = SGDR(self.data.trn_dl.n /
                              self.data.trn_dl.batch_size * cycle_len,
                              cycle_mult=cycle_mult,
                              base_lr=1e-9,
                              max_lr=lrs)
            self.callbacks.append(self.sched)
        else:
            epochs = n_cycle
        self.model.compile(loss=self.loss,
                           optimizer=self.opt_fn,
                           metrics=list(metrics))
        self.model.fit_generator(self.data.trn_dl,
                                 steps_per_epoch=len(self.data.trn_dl),
                                 epochs=self.nepoch + epochs,
                                 callbacks=self.callbacks,
                                 validation_data=self.data.val_dl,
                                 validation_steps=len(self.data.val_dl),
                                 initial_epoch=self.nepoch,
                                 **kwargs)
        self.nepoch += epochs

    def predict(self, on_test_set=False, save=True):
        """Return predictions on validation or test set"""
        test_dl = self.data.test_dl if on_test_set else self.data.val_dl

        try:
            # turn off shuffle
            shuffle = test_dl.shuffle
            test_dl.shuffle = False

            y_pred = self.model.predict_generator(test_dl)
            if save:
                name = 'test' if on_test_set else 'valid'
                np.save(f'{self.outdir}{os.sep}y_pred_{name}.npy', y_pred)
        finally:
            # revert shuffle state
            test_dl.shuffle = shuffle

        return y_pred

    def unfreeze(self):
        """ Unfreeze all layers """
        for layer in self.model.layers:
            layer.trainable = True

    def save(self, fname):
        """Save model file """
        model_file = f'{self.outdir}{os.sep}{fname}.h5'
        self.model.save(model_file)
        def save_tflite(optimize=0):
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model) 
            tflite_file = f'{self.outdir}{os.sep}{fname}-raw.tflite'
            if not os.path.exists(tflite_file):
                tflite_model = converter.convert()
                open(tflite_file, "wb").write(tflite_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            suffix=''
            if 0==optimize:
                suffix='f16'
                converter.target_spec.supported_types = [tf.float16]
            elif 1==optimize:
                suffix='dyn' # dunamic range quantization
            elif 2==optimize:
                suffix='8bit' # full 8-bit interge
            tflite_model = converter.convert()
            tflite_file = f'{self.outdir}{os.sep}{fname}-{suffix}.tflite'
            open(tflite_file, "wb").write(tflite_model)
        save_tflite(0)
        save_tflite(1)
        def save_lr_history():
            print(f"type={type(self.sched.history)}; content={self.sched.history}")
            with open(f'{self.outdir}{os.sep}lr_history.json', 'w') as f:
                json.dump(str(self.sched.history), f)
        save_lr_history()

        return model_file

    def load(self, fname, weights_only=True, custom_objects=None):
        """
        Load model from file.

        Args:
            fname (str): Full path to hdf5 model file
            weights_only (bool): Option to only load weights and not full model
            custom_objects (dict): Dictionary of custom objects to load if loading full model

        Returns:
            Loaded model
        """

        if weights_only:
            self.model.load_weights(f'{fname}')
        else:
            self.model = load_model(f'{fname}', custom_objects=custom_objects)
