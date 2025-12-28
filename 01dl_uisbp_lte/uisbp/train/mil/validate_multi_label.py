from datetime import datetime
import os
from os.path import basename, dirname, isfile, join
import time

from tensorflow.python.keras import backend as K
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
from tqdm import tqdm

from . import losses
from . import model 
from . import scoring


def epoch_number(s, prefix, suffix):
    """
    Get the epoch number from a string which looks like: prefix + number + suffix
    """
    return int(s.split(prefix)[1].split(suffix)[0])


def best_accuracy_score_and_threshold(y_true, y_score, n_thr=101):
    """
    Args:
        y_true: An np.ndarray
            The ground truth class for each sample
        y_score: An np.ndarray
            The model probability for each sample
        n_thr: int, optional
            The number of threshold samples

    Returns:
            The model accuracy

            The corresponding threshold.

    Raises:
        ValueError
            When the values of y_score do not lie in [0, 1], the values of y_true are not binary, n_thr is not a positive integer.
    """
    if len(np.unique(y_true)) > 2:
        raise ValueError("The values of `y_true` must be binary")

    y_score_np = np.array(y_score)
    if np.any(np.logical_or(y_score_np < 0, y_score_np > 1)):
        raise ValueError("The values of `y_score` must lie in the interval [0, 1]")

    if not isinstance(n_thr, int) or n_thr <= 0:
        raise ValueError("`n_thr` must be a positive integer")

    # Get the list of thresholds
    thr_vec = np.linspace(0, 1, n_thr)

    # Estimate the accuracy using each threshold
    acc_vec = np.zeros((n_thr, ))

    for ix, thr in enumerate(thr_vec):
        acc_vec[ix] = metrics.accuracy_score(y_true, (y_score > thr).astype(np.int))

    # Compute original accuracy and threshold
    best_idx = np.argmax(acc_vec)
    acc = acc_vec[best_idx]
    thr = thr_vec[best_idx]

    return acc, thr


def validate(dataset, model_callable_name, num_classes,
    batch_size=1, modelpath="models/trained_model/model",
    img_height=96, img_width=96, input_scale=255, class_weights=None,
    log_device_placement=False, allow_soft_placement=True, allow_growth=True,
    test_info_period=20, use_average_variables=False, moving_average_decay=0.9,
    print_file='validation_model_selection.txt'):
    """
    Args:
        dataset: Dataset  
            A Dataset object containing the validation data
        model_callable_name: str
            Name of function defined within the model.py file for creating tensorflow model
        num_classes: int
            The number of classes
        batch_size: int, optional
            The validation batch size
        modelpath: str, optional
            Path and prefix for reading checkpoint files
        img_height: int, optional
            Image height to resize input images to
        img_width: int, optional
            Image width to resize input images to
        input_scale: float, optional
            Value to normalize input image intensity by before feading to model
        class_weights: tuple, optional
            num_classes-element tuple to specify weight to be applied to loss function to deal with data imbalance
        log_device_placement: bool, optional
            Whether to log device placement in tensorflow model
        allow_soft_placement: bool, optional
            Whether to allow operations to be placed on other devices when the requested resources are not available
        allow_growth: bool, optional
            Whether tensorflow should incrementally use the GPU memory or to map the entire memory
        test_info_period: int, optional
            The number of evaluation steps after which to display test information. This information is also displayed from the first step until test_info_period
        use_average_variables: bool, optional
            Whether to use average (shadow) variables instead of current variables
        moving_average_decay: float, optional
            The moving average decay constant. Lies in [0, 1]  
        print_file: str
            Path to text file which contains a subset of the print statements

    Returns:
        Doesn't return anything. It only has side-effects of writing optimal epoch number and corresponding detection threshold to input txt file: print_file
    """
    # Open file to store printout of validation performance
    print_file_obj = open(print_file, 'a')

    data_size = dataset.size()
    print('Validation dataset size: % d' % (data_size))

    real_eps = 1e-10 # For display purposes

    print("Start validation")
    print("Number of samples: %d" % (data_size))
    print("It'll take %d iterations for the full validation set." % (data_size//batch_size))
    

    with tf.Graph().as_default():
        startstep = 0

        # Placeholder(s) for graph input and loss computation
        img_ = tf.placeholder('float32', shape=(None, img_height, img_width, 1), name='input')
        y_ = tf.placeholder('float32', shape=(None, num_classes), name='y')
              
        # Get the MIL model
        model_func = getattr(model, model_callable_name)

        if model_func is None:
            raise ValueError("`%s` is not a function defining a model in `%s`" % (model_callable_name, model))

        # Get the output of the MIL model
        mdl = model_func(input_placeholder=img_, num_classes=num_classes, input_scale=input_scale, is_training=False, give_summary=True)  

        _, pool_out, fc_out = mdl.outputs
        
        # Loss
        loss = losses.mil_loss_multi_label(y_, pool_out, fc_out, pool_coef=1.0, fc_coef=1.0, class_weights=class_weights)
        
        # img score
        img_score = scoring.create_score_multi_label(fc_out) 
        
        # Get instance of Saver to restore model variables
        restorer = tf.train.Saver()

        if use_average_variables: 
            print("Using average variables for inference")
            # Create an ExponentialMovingAverage object
            ema = tf.train.ExponentialMovingAverage(moving_average_decay)

            # Create a Saver that loads trainable variables from their saved shadow values and non-trainable variables from their regular values 
            var_dict = {ema.average_name(v): v for v in tf.trainable_variables()}
            restorer_ema = tf.train.Saver(var_dict)  
        else:
            print("Using normal variables for inference")      

        config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        config.gpu_options.allow_growth = allow_growth

        sess = tf.Session(config=config)

        print("*******************************************************************************************************************", file=print_file_obj)
        print("Assessing performance:", file=print_file_obj)
        print("*******************************************************************************************************************", file=print_file_obj)                    
        
        # Get the list of checkpoint files in the model directory
        model_dir = dirname(modelpath)
        prefix = basename(modelpath) + "-"
        suffix = ".index"
        epoch_numbers = [epoch_number(f, prefix, suffix) for f in os.listdir(model_dir) if isfile(join(model_dir, f)) and f.startswith(prefix) and f.endswith(suffix)]

        opt_metric_performance_vec = np.zeros((len(epoch_numbers), ))
        opt_metric_thr_vec = np.zeros((len(epoch_numbers), ))
        for ix, epoch in enumerate(epoch_numbers):
            print("Validating model %d out of %d" % (ix + 1, len(epoch_numbers))) 

            # Path to the current checkpoint file
            ckpt_file = modelpath+ "-" + str(epoch)

            # Restore original variables
            restorer.restore(sess, ckpt_file)

            if use_average_variables:
                # Restore shadow (moving average) variables
                restorer_ema.restore(sess, ckpt_file)               

            img_label = []
            img_scores = []
            
            step = startstep
            try:
                for batch_x, batch_y, _ in tqdm(dataset.batches(batch_size), ncols=80):
            
                    step += 1

                    start_time = time.time()

                    feed_dict = {img_: batch_x, y_: batch_y, K.learning_phase(): 0}

                    loss_value, score = sess.run([loss, img_score], feed_dict=feed_dict)        

                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if (step + 1) % test_info_period == 0 or step < test_info_period:
                        sec_per_batch = float(duration)
                        print('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f sec/batch)' % (datetime.now(), step, loss_value,
                            batch_size/(duration + real_eps), sec_per_batch))

                    img_label.extend(batch_y.tolist())
                    img_scores.extend(score.tolist())
                    
            except KeyboardInterrupt:
                print("Testing interrupted")

                # Close the print file
                print_file_obj.close()

            opt_perf, opt_thr = best_accuracy_score_and_threshold(np.array(img_label), np.array(img_scores))
            opt_metric_performance_vec[ix] = opt_perf
            opt_metric_thr_vec[ix] = opt_thr

            print("Model number = %d, Best accuracy = %.2f %%, Best thr = %.4f" % (epoch, opt_perf*100., opt_thr), file=print_file_obj)

        # Select the model with the best performance
        best_idx = np.argmax(opt_metric_performance_vec)
        opt_perf_best = opt_metric_performance_vec[best_idx]
        opt_thr_best = opt_metric_thr_vec[best_idx]
        opt_best_epoch_number = epoch_numbers[best_idx]
        
        print("", file=print_file_obj)
        print("Best Performance:", file=print_file_obj)
        print("Best Opt. Epoch: %d" % opt_best_epoch_number, file=print_file_obj)
        print("Opt. metric = %.2f %%, Opt. thr = %.4f" % (opt_perf_best*100., opt_thr_best), file=print_file_obj)
        print("*******************************************************************************************************************", file=print_file_obj)                 

    # Close the print file
    print_file_obj.close() 