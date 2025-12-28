import os
import time

import numpy as np
from tensorflow.python.platform import tf_logging as logging

from uisbp.transform_utils import adjust_hist
from uisbp.data import DataLoader
from uisbp.train.mil.config import CommonConfig, DatasetConfig, DeviceConfig, OutputConfig, TrainingConfig
from uisbp.train.mil import input
from uisbp.train.mil.validate_binary import validate


def main():    
    # Configurations
    # Which of GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = DeviceConfig.CUDA_VISIBLE_DEVICES

    logging.set_verbosity(logging.INFO)

    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)

    st = time.time()
    print('Creating validation Dataset classes for on-the-fly data loading')

    ## Load training and validation data
    dl = DataLoader(DatasetConfig.PROCESSED_DATA_PATH, dataset=DatasetConfig.DATASET_NAME)
    X_train, Y_train, ids_train, X_valid, Y_valid, ids_valid = dl.get_train_valid_split(size=DatasetConfig.IMAGE_HEIGHT, 
        seed=DatasetConfig.TRAIN_VAL_SPLIT_SEED, test_size=DatasetConfig.TRAIN_VAL_SPLIT, filter_nerve_images=False, return_test=False)
    labels = dl.labels

    print("X, Y validate sizes are: ", X_valid.shape, Y_valid.shape)
    if labels is not None:
        print("Class labels: ", labels)

    ## Initialize training and validation data generators for MIL
    # CLAHE preprocessing for image, but not mask
    preprocess_func = lambda x, y: (adjust_hist(x), y)

    dataset = input.Dataset(X_valid, Y_valid, ids_valid, preprocess_fn=preprocess_func, categorical=False, 
        input_queue_size=CommonConfig.INPUT_QUEUE_SIZE)

    print('Done loading data, time = %.3f' % (time.time() - st))

    # Begin validation
    validate(dataset, CommonConfig.MODEL_CALLABLE_NAME, 
        batch_size=CommonConfig.BATCH_SIZE_TEST, modelpath=CommonConfig.MODEL_PATH, 
        img_height=DatasetConfig.IMAGE_HEIGHT, img_width=DatasetConfig.IMAGE_WIDTH, input_scale=DatasetConfig.INPUT_SCALE, class_weights=DatasetConfig.CLASS_WEIGHTS, 
        log_device_placement=DeviceConfig.LOG_DEVICE_PLACEMENT, allow_soft_placement=DeviceConfig.ALLOW_SOFT_PLACEMENT, allow_growth=DeviceConfig.ALLOW_GROWTH,
        test_info_period=OutputConfig.TEST_INFO_PERIOD, use_average_variables=OutputConfig.USE_AVERAGE_VARIABLES, 
        moving_average_decay=TrainingConfig.MOVING_AVERAGE_DECAY, 
        print_file=CommonConfig.PRINT_FILE)


if __name__ == '__main__':
    main()