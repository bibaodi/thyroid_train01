import os
import time

import numpy as np
from tensorflow.python.platform import tf_logging as logging

from uisbp import augmentations
from uisbp.augmentations import (Augmentation, elastic_transform, identity, random_flip_lr, random_zoom,
                                 random_rotation, random_shear, random_shift, random_gain)
from uisbp.transform_utils import adjust_hist
from uisbp.data import DataLoader
from uisbp.train.mil.config import CommonConfig, DatasetConfig, DeviceConfig, OutputConfig, TrainingConfig
from uisbp.train.mil import input
from uisbp.train.mil.train_binary import train


def main():
    # Configurations
    # Which of GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = DeviceConfig.CUDA_VISIBLE_DEVICES

    logging.set_verbosity(logging.INFO)

    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)

    st = time.time()
    print('Creating training and validation Dataset classes for on-the-fly data loading')

    ## Load training and validation data
    dl = DataLoader(DatasetConfig.PROCESSED_DATA_PATH, dataset=DatasetConfig.DATASET_NAME)
    X_train, Y_train, ids_train, X_valid, Y_valid, ids_valid = dl.get_train_valid_split(size=DatasetConfig.IMAGE_HEIGHT, 
        seed=DatasetConfig.TRAIN_VAL_SPLIT_SEED, test_size=DatasetConfig.TRAIN_VAL_SPLIT, filter_nerve_images=False, return_test=False)
    labels = dl.labels

    print("X, Y train sizes are: ", X_train.shape, Y_train.shape)
    print("X, Y validate sizes are: ", X_valid.shape, Y_valid.shape)
    if labels is not None:
        print("Class labels: ", labels)

    ## Augmentations to perform on training dataset
    augs = tuple([
        Augmentation(fn=identity),
        Augmentation(fn=elastic_transform, kwargs={'sigma': 24, 'alpha': 720}),
        Augmentation(fn=random_flip_lr, kwargs={'p': 0.5}),
        Augmentation(fn=random_zoom, kwargs={'zoom_range':(0.9, 1.1)}),
        Augmentation(fn=random_rotation, kwargs={'rg': 20.0, 'fill_mode':'constant'}),
        Augmentation(fn=random_shear, kwargs={'intensity': 0.05}),
        Augmentation(fn=random_shift, kwargs={'wrg': 0.1, 'hrg': 0.1, 'fill_mode':'constant'}),
        Augmentation(fn=random_gain, kwargs={'gain_range': (0.2, 1.8)})
        ])

    ## Initialize training and validation data generators for MIL
    # CLAHE preprocessing for image, but not mask
    preprocess_func = lambda x, y: (adjust_hist(x), y)

    dataset_train = input.Dataset(X_train, Y_train, ids_train, preprocess_fn=preprocess_func, seed=DatasetConfig.BATCHING_SEED, 
        augmentation=augs, categorical=False, input_queue_size=CommonConfig.INPUT_QUEUE_SIZE,
        use_compose=False)
    print("Y train count: {}, positive label count: {}".
            format(len(dataset_train.y_labels), np.count_nonzero(dataset_train.y_labels)))

    dataset_val = input.Dataset(X_valid, Y_valid, ids_valid, preprocess_fn=preprocess_func, categorical=False, 
        input_queue_size=CommonConfig.INPUT_QUEUE_SIZE)
    print("Y validate count: {}, positive label count: {}".
            format(len(dataset_val.y_labels), np.count_nonzero(dataset_val.y_labels)))

    print('Done loading data, time = %.3f' % (time.time() - st))

    # Begin training
    train(dataset_train, dataset_val, CommonConfig.MODEL_CALLABLE_NAME, 
        batch_size=CommonConfig.BATCH_SIZE_TRAIN, num_checkpoints=CommonConfig.NUM_CHECKPOINTS, output_modelpath_ckpt=CommonConfig.MODEL_PATH, 
        output_modelpath_pb=CommonConfig.MODEL_PATH_PB, pb_as_text=CommonConfig.AS_TEXT, tb_path=CommonConfig.TB_PATH, 
        img_height=DatasetConfig.IMAGE_HEIGHT, img_width=DatasetConfig.IMAGE_WIDTH, input_scale=DatasetConfig.INPUT_SCALE, class_weights=DatasetConfig.CLASS_WEIGHTS, 
        log_device_placement=DeviceConfig.LOG_DEVICE_PLACEMENT, allow_soft_placement=DeviceConfig.ALLOW_SOFT_PLACEMENT, allow_growth=DeviceConfig.ALLOW_GROWTH,
        train_period=OutputConfig.TRAIN_PERIOD, val_period=OutputConfig.VAL_PERIOD, save_period=OutputConfig.SAVE_PERIOD, 
        summary_period=OutputConfig.SUMMARY_PERIOD, train_sample_size=OutputConfig.TRAIN_SAMPLE_SIZE, 
        val_sample_size=OutputConfig.VAL_SAMPLE_SIZE, train_info_period=OutputConfig.TRAIN_INFO_PERIOD, 
        num_epoch=TrainingConfig.NUM_EPOCH, learning_rate=TrainingConfig.LEARNING_RATE, momentum=TrainingConfig.MOMENTUM, opt_method=TrainingConfig.OPT_METHOD, 
        model_seed=TrainingConfig.MODEL_SEED, moving_average_decay=TrainingConfig.MOVING_AVERAGE_DECAY, num_epochs_per_decay=TrainingConfig.NUM_EPOCHS_PER_DECAY, 
        learning_rate_decay_factor=TrainingConfig.LEARNING_RATE_DECAY_FACTOR,
        input_ckptfile=CommonConfig.MODEL_FILEPATH_TRAIN)


if __name__ == '__main__':
    main()