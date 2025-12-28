from os.path import join, isabs
import types

import numpy as np
from ruamel import yaml

# from ..dev.util import find_data_dir, find_repo_root
from ...dev.util import find_data_dir, find_repo_root

# CONFIG_FILEPATH = join(find_repo_root(), "uisbp", "train", "mil", "config_mil_binary.yml")
CONFIG_FILEPATH = join(find_repo_root(), "uisbp", "train", "mil", "config_mil_multi_label.yml")
   
def to_boolean(value):
    """
    Converts a value to boolean
    """
    if isinstance(value, str):
        value = value.lower() in ("yes", "true", "t", "y", "on", "1")
    else:
        value = bool(value)

    return value


def read_config(filepath):
    """
    Args:
        filepath: string
            The path to the .yml file

    Returns: 
        data: dict
            A dict containing the contents of the configuration .yml file

    Reads a .yml configuration file and returns the result in a dict
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    return data


config_data = read_config(CONFIG_FILEPATH).get("wingspan")


class Config():
    @classmethod
    def property_names_and_values(cls):
        """
        Get the list of tuple containing class attributes and their corresponding values
        """
        prop_names_and_values = [(p, getattr(cls, p)) for p in dir(cls) if not p.startswith('__') and not p.endswith('__')
            and not isinstance(getattr(cls, p), (dict, types.MethodType))]

        return prop_names_and_values


class DatasetConfig(Config):
    """
    Constants for the data set.
    """
    dataset_config_data = config_data.get("dataset")

    # Path to processed data containing files 'img_train.npy', 'imgs_mask_train.npy', 'imgs_fname_train.csv', 
    # 'img_test.npy', 'imgs_mask_test.npy', 'imgs_fname_test.csv'
    PROCESSED_DATA_PATH = dataset_config_data.get("processed-data-path")   
    if PROCESSED_DATA_PATH is not None:
        if not isabs(PROCESSED_DATA_PATH):
            PROCESSED_DATA_PATH = join(find_repo_root(), PROCESSED_DATA_PATH)

    # Dataset name. Valid options are [feb, mar, apr]
    DATASET_NAME = dataset_config_data.get("dataset-name").lower()

    # The proportion of the training set that should be assigned to the validation dataset
    TRAIN_VAL_SPLIT = float(dataset_config_data.get("train-val-split", 0.2))

    # The seed for ensuring the same data split when splitting training data into train/validate
    TRAIN_VAL_SPLIT_SEED = int(dataset_config_data.get("train-val-split-seed", 0))

    # Dataset batching randomization seed
    BATCHING_SEED =  int(dataset_config_data.get("batching-seed", 0))

    # Size of input images to network
    IMAGE_HEIGHT = int(dataset_config_data.get("image-height", 454))
    IMAGE_WIDTH = int(dataset_config_data.get("image-width", 454))
    IMAGE_CHANNELS = int(dataset_config_data.get("image-channels", 1))

    # Number to divide the input data by to prevent floating-point overflow
    INPUT_SCALE = float(dataset_config_data.get("input-scale", 1.0))

    # Which scenario to use for splitting the dataset into a training and testing set
    # Options are: {'random', 'subject'}
    SCENARIO = dataset_config_data.get("scenario", "random")

    # Class weights for binary classification
    CLASS_WEIGHTS = dataset_config_data.get("class-weights",  (1.0, 1.0))

       
class TrainingConfig(Config):
    """
    Constants for training the model, saving and loading model parameters
    """   
    training_config_data = config_data.get("training")

    # Number of training epochs to train the model
    NUM_EPOCH = int(training_config_data.get("num-epoch", 50))

    # Initial learning rate
    LEARNING_RATE = float(training_config_data.get("learning-rate", 1e-3))

    # The momentum factor. Lies in [0, 1]. As BATCH_SIZE_TRAIN decreases, MOMENTUM should increase to encourage the use of more previous batches update information to guide the gradient
    MOMENTUM = float(training_config_data.get("momentum", 0.9))

    if MOMENTUM < 0 or MOMENTUM > 1:
        raise ValueError("MOMENTUM should be within [0, 1] but is {}".format(MOMENTUM))

    # "sgd_momentum" or "adam"
    OPT_METHOD = training_config_data.get("opt-method", "adam")

    # Random seed for tensorflow model
    MODEL_SEED = int(training_config_data.get("model-seed", 0))

    # The decay to use for the moving average for creating shadow variable
    MOVING_AVERAGE_DECAY = float(training_config_data.get("moving-average-decay", 0.9999))

    # Epochs after which learning rate decays.
    NUM_EPOCHS_PER_DECAY = int(training_config_data.get("num-epochs-per-decay", NUM_EPOCH))
    
    # Learning rate decay factor
    LEARNING_RATE_DECAY_FACTOR = float(training_config_data.get("learning-rate-decay-factor", 0.1))

  
class CommonConfig(Config):
    """
    Constants for both training and testing
    """
    common_config_data = config_data.get("common")

    ## Number of samples in batch during training
    BATCH_SIZE_TRAIN = int(common_config_data.get("batch-size-train", 64))

    # Number of samples in batch during testing
    BATCH_SIZE_TEST = int(common_config_data.get("batch-size-test", BATCH_SIZE_TRAIN))

    # This must be more than twice the maximum of BATCH_SIZE_TRAIN and BATCH_SIZE_TEST
    INPUT_QUEUE_SIZE = int(common_config_data.get("input-queue-size", 4 * max(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)))

    """
    Constants for saving and loading model parameters
    """
    # Output directory for saving learned model and tensorboard events
    MODEL_NAME = common_config_data.get("model-name", "model-default")
    MODEL_DIR = join(find_data_dir("models"), MODEL_NAME)
    MODEL_PATH = join(MODEL_DIR, "model") 

    # Path to tensorboard events
    TB_PATH = join(find_data_dir("logs"), MODEL_NAME)

    # Output directory for saving results
    RESULT_DIR = find_data_dir(join("results", MODEL_NAME))

    # Where to store computed network values on test dataset
    TEST_RESULT_PATH = common_config_data.get("test-result-path", join(RESULT_DIR, "test_results.npz"))
    if not isabs(TEST_RESULT_PATH):
        TEST_RESULT_PATH = join(find_repo_root(), TEST_RESULT_PATH)  

    # Get the checkpoint number to load during training for finetuning
    MODEL_CHECKPOINT_NUM_TRAIN = common_config_data.get("model-checkpoint-num-train")
    
    # Get the checkpoint number to load during test
    MODEL_CHECKPOINT_NUM_TEST = int(common_config_data.get("model-checkpoint-num-test", TrainingConfig.NUM_EPOCH - 1))
    
    # Path to pre-trained model for fine-tuning
    if MODEL_CHECKPOINT_NUM_TRAIN is not None:
        MODEL_FILEPATH_TRAIN = MODEL_PATH + "-" + str(MODEL_CHECKPOINT_NUM_TRAIN)
        if not isabs(MODEL_FILEPATH_TRAIN):
            MODEL_FILEPATH_TRAIN = join(find_repo_root(), MODEL_FILEPATH_TRAIN)
    else:
        MODEL_FILEPATH_TRAIN = None

    # Location of trained model for testing purposes
    MODEL_FILEPATH_TEST = common_config_data.get("model-filepath-test", MODEL_PATH + "-" + str(MODEL_CHECKPOINT_NUM_TEST))
    if not isabs(MODEL_FILEPATH_TEST):
        MODEL_FILEPATH_TEST = join(find_repo_root(), MODEL_FILEPATH_TEST)

    # Number of checkpoints to store
    NUM_CHECKPOINTS = int(common_config_data.get("num-checkpoints", 1))

    # Name of callable that defines tensorflow model in model.py
    MODEL_CALLABLE_NAME = common_config_data.get("model-callable-name", "mil_multi_label_sz_96")

    # Protobuf related
    AS_TEXT = to_boolean(common_config_data.get("as-text", False))

    if AS_TEXT:
        MODEL_PATH_PB = join(MODEL_DIR, "graph.pbtxt")
    else:
        MODEL_PATH_PB = join(MODEL_DIR, "graph.pb")

    MODEL_PATH_FROZEN_PB = join(MODEL_DIR, "frozen_graph.pb")

    MODEL_PATH_OPTIMIZED_PB = join(MODEL_DIR, "optimized_frozen_graph.pb")

    # This is used for validation purposes and tracking performance across epochs
    PRINT_FILE = common_config_data.get("print-file", "validation_model_selection.txt")

    # The threshold used for converting scores into binary decisions
    OPT_THRESHOLD = float(common_config_data.get("opt-threshold", 0.5))


class OutputConfig(Config):
    """
    Constants for display and summaries
    """
    output_config_data = config_data.get("output")

    # Whether to use trained variables or their shadow copies (average trained variables) for inference
    USE_AVERAGE_VARIABLES = to_boolean(output_config_data.get("use-average-variables", False))

    # Sample how many volumes(CT) for training set evaluation
    TRAIN_SAMPLE_SIZE = int(output_config_data.get("train-sample-size", CommonConfig.BATCH_SIZE_TRAIN))

    # Sample how many volumes(CT) for validation
    # This affects the validation time
    VAL_SAMPLE_SIZE = int(output_config_data.get("val-sample-size", CommonConfig.BATCH_SIZE_TRAIN))

    # Do an evaluation on subset of training set every TRAIN_PERIOD iterations
    TRAIN_PERIOD = int(output_config_data.get("train-period", 1))

    # Do a validation every VAL_PERIOD iterations
    VAL_PERIOD = int(output_config_data.get("val-period", 1))

    # Save the summary every SUMMARY_PERIOD iterations
    SUMMARY_PERIOD = int(output_config_data.get("summary-period", 100))

    # Save the model parameters to checkpoint file every SAVE_PERIOD epochs
    SAVE_PERIOD = int(output_config_data.get("save-period", 1))

    # Print out training information every TRAIN_INFO_PERIOD iterations
    TRAIN_INFO_PERIOD = int(output_config_data.get("train-info-period", 1))

    # Print out test information every TEST_INFO_PERIOD iterations
    TEST_INFO_PERIOD = int(output_config_data.get("test-info-period", 1))


class DeviceConfig(Config):
    """
    Device configuration related
    """
    device_config_data = config_data.get("device")

    # Which GPU to make visible
    CUDA_VISIBLE_DEVICES = device_config_data.get("cuda-visible-devices", "0")

    # Allow device soft device placement
    ALLOW_SOFT_PLACEMENT = to_boolean(device_config_data.get("allow-soft-placement", True))

    # Log placement of ops on devices
    LOG_DEVICE_PLACEMENT = to_boolean(device_config_data.get("log-device-placement", False))

    # Whether to allow GPU allocation to grow
    ALLOW_GROWTH = to_boolean(device_config_data.get("allow-growth", True))

    # A bound on the amount of GPU memory available to the TensorFlow process
    PER_PROCESS_GPU_MEMORY_FRACTION = float(device_config_data.get("per-process-gpu-memory-fraction", 1.0))


# print config when this module is imported
if to_boolean(config_data.get("print-config", False)):
    config_classes = (DatasetConfig, TrainingConfig, CommonConfig, OutputConfig, DeviceConfig)

    for config_class in config_classes:
        for param, value in config_class.property_names_and_values():
            print(config_class.__name__ + "." + param, "=", value)