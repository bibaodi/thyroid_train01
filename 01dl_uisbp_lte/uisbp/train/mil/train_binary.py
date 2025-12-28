from datetime import datetime
from os.path import basename, dirname
import time

from tensorflow.python.keras import backend as K
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf

from . import losses
from . import model 
from . import scoring


def train(dataset_train, dataset_val, model_callable_name, 
    batch_size=1, num_checkpoints=1, output_modelpath_ckpt="models/trained_model/model", output_modelpath_pb="models/trained_model/graph.pb", pb_as_text=False, tb_path="logs/trained_model/", 
    img_height=96, img_width=96, input_scale=255, class_weights=None,
    log_device_placement=False, allow_soft_placement=True, allow_growth=True,
    train_period=10, val_period=10, save_period=10, summary_period=10, train_sample_size=1000, val_sample_size=1000, train_info_period=20, 
    num_epoch=100, learning_rate=1e-3, momentum=0.99, opt_method='adam', model_seed=0, moving_average_decay=0.9, num_epochs_per_decay=50, learning_rate_decay_factor=0.1,
    input_ckptfile=None):
    """
    Args:
        dataset_train: Dataset                  
            A Dataset object containing the training data
        dataset_val: Dataset  
            A Dataset object containing the validation data
        model_callable_name: str
            Name of function defined within the model.py file for creating tensorflow model
        batch_size: int, optional
            The training batch size
        num_checkpoints: int, optional
            The number of checkpoints to keep stored during training
        output_modelpath_ckpt: str, optional
            Path and prefix for saving checkpoint files
        output_modelpath_pb: str, optional
            Path for saving model files as protobuf
        pb_as_text: bool, optional
            Whether to save protobuf file as text or binary
        tb_path: str, optional
            Path to tensorboard log file
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
        train_period: int, optional
            The number of epochs after which to evaluate performance on a subset of the training data
        val_period: int, optional
            The number of epochs after which to evaluate performance on a subset of the validation data
        save_period: int, optional
            The number of epochs after which to save the current trained model
        summary_period: int, optional
            The number of epochs after which to write a summary of the specified nodes and metrics to the tensorboard log
        train_sample_size: int, optional
            The number of training samples to evaluate performance on during train_period
        val_sample_size: int, optional
            The number of validation samples to evaluate performance on during val_period
        train_info_period: int, optional
            The number of optimization steps after which to display training information. This information is also displayed from the first step until train_info_period
        num_epoch: int, optional
            The number of training epochs (an epoch is a single pass through the training data)
        learning_rate: float, optional
            The initial learning rate
        momentum: float, optional
            The fraction of the update vector of the past time step to be added to the current update vector. Only used when opt_method is "sgd_momentum"
        opt_method: str, optional
            The optimization algorithm to use. Options are "adam" and "sgd_momentum"
        model_seed: int, optional
            The random seed used for initializing tensorflow model weights and biases
        moving_average_decay: float, optional
            The moving average decay constant. Lies in [0, 1]        
        num_epochs_per_decay: int, optional
            The number of epochs after which to scale the learning rate by learning_rate_decay_factor
        learning_rate_decay_factor: float, optional
            The factor to multiply learning by after a given number of iterations
        input_ckptfile: str, optional
            Path to a checkpoint file for the model parameters to kickoff training

    Returns:
        Doesn't return anything. It only has side-effects of saving trained models and training logs to file
    """
    if input_ckptfile:
        print('train(...) called for fine-tuning training.')
    else:
        print('train(...) called for from-scratch training.')

    data_size_train = dataset_train.size()
    print('Training dataset size: %d' % (data_size_train))

    data_size_valid = dataset_val.size()
    print('Validation dataset size: %d' % (data_size_valid))

    # Assume internal checkpoint files are saved with the global step as follows:
    # "filename_prefix-global_step"
    global_step_reload = 0
    if input_ckptfile:
        global_step_reload = int(basename(input_ckptfile).split('-')[-1])

    real_eps = 1e-10 # For display purposes
    with tf.Graph().as_default():
        # Set random seed for tensorflow
        np.random.seed(model_seed) # keras initialization
        tf.set_random_seed(model_seed) # base tensorflow initialization

        startstep = 0 if not input_ckptfile else global_step_reload
        print("Step = %d" % (startstep))
        global_step = tf.Variable(startstep, trainable=False)

        # Placeholder(s) for graph input and loss computation
        img_ = tf.placeholder('float32', shape=(None, img_height, img_width, 1), name='input')
        y_ = tf.placeholder('float32', shape=(None), name='y')
              
        # Get the MIL model
        model_func = getattr(model, model_callable_name)

        if model_func is None:
            raise ValueError("`%s` is not a function defining a model in `%s`" % (model_callable_name, model))

        # Get the output of the MIL model
        mdl = model_func(input_placeholder=img_, input_scale=input_scale, is_training=True, give_summary=True)   
            
        pool_out = mdl.outputs[0]
        
        # Gather the update ops for BatchNormalization from keras model
        update_ops = []
        for layer in mdl.layers: 
            if len(layer.updates) > 0:
                update_ops.extend(layer.updates)      
                    
        loss = losses.mil_loss_binary(y_, pool_out, class_weights=class_weights)
        prediction = scoring.create_prediction_binary(pool_out)
        score = scoring.create_score_binary(pool_out)      

        # Create training op
        train_op = model.train(loss, global_step, update_ops, data_size_train, batch_size, learning_rate, 
            num_epochs_per_decay, learning_rate_decay_factor, opt_method=opt_method, momentum=momentum, 
            moving_average_decay=moving_average_decay, var_list=mdl.trainable_weights)

        # Build the summary operation based on the collection of Summaries
        summary_op = tf.summary.merge_all()

        # Create additional summaries to be updated on a different schedule from summary_op
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_loss_summary = tf.summary.scalar('validation_loss', validation_loss)

        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        train_loss = tf.placeholder('float32', shape=(), name='train_loss')
        train_loss_summary = tf.summary.scalar('train_loss', train_loss)

        train_acc = tf.placeholder('float32', shape=(), name='train_accuracy')
        train_acc_summary = tf.summary.scalar('train_accuracy', train_acc)

        if input_ckptfile:
            restorer = tf.train.Saver(model.get_variables_to_restore())

        # Get instance of Saver to store ALL model variables
        # This ensures that the exponential moving average shadow variables are also stored
        saver = tf.train.Saver(max_to_keep=num_checkpoints)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        config.gpu_options.allow_growth = allow_growth

        sess = tf.Session(config=config)

        # Write the graph to protobuf
        logdir = dirname(output_modelpath_pb)
        pb_name = basename(output_modelpath_pb)
        tf.train.write_graph(sess.graph, logdir, pb_name, as_text=pb_as_text) 

        summary_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)

        if input_ckptfile:
            # Load checkpoint file
            restorer.restore(sess, input_ckptfile)
            print('Variables restored from checkpoint file: %s' % (input_ckptfile))
        else:
            # From scratch
            sess.run(init_op)
            print('Done initializing from scratch')

        print('Now training all the model parameters')

        step = startstep
        try:
            for epoch in range(num_epoch):
                print('epoch: %d' % (epoch))
                
                # Shuffle training and validation data
                dataset_train.shuffle()
                dataset_val.shuffle()

                for batch_x, batch_y, _ in dataset_train.batches(batch_size, dynamic_seed=epoch):
                    step += 1

                    start_time = time.time()                    

                    feed_dict = {img_: batch_x, y_: batch_y, K.learning_phase(): 1}

                    _, loss_value, = sess.run([train_op, loss], feed_dict=feed_dict)

                    duration = time.time() - start_time

                    # Print training information
                    if step % train_info_period == 0 or step - startstep < train_info_period:
                        sec_per_batch = float(duration)
                        print('%s: step %d, loss = %.4f, (%.1f examples/sec; %.3f sec/batch)' % (datetime.now(),
                            step, loss_value, batch_size/(duration + real_eps), sec_per_batch))

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'     
                
                # Analyze training subset
                if (epoch + 1) % train_period == 0:
                    train_losses = []
                    train_predictions = []
                    train_scores = []
                    train_y = []
                    for train_batch_x, train_batch_y, _ in dataset_train.sample_batches(batch_size, train_sample_size):
                        train_feed_dict = {img_: train_batch_x, y_: train_batch_y, K.learning_phase(): 0}
                        tr_loss, tr_pred, tr_score = sess.run([loss, prediction, score], feed_dict=train_feed_dict)

                        train_losses.append(tr_loss)
                        train_predictions.extend(tr_pred.tolist())
                        train_scores.extend(tr_score.tolist())
                        train_y.extend(train_batch_y.tolist())

                    tr_loss = np.mean(train_losses)
                    tr_acc = metrics.accuracy_score(np.array(train_y), np.array(train_predictions))

                    print('%s: epoch %d, train loss = %.4f, acc = %.2f %%' % (datetime.now(), epoch,
                        tr_loss, tr_acc*100.))

                    # Training summary
                    train_loss_summ = sess.run(train_loss_summary, feed_dict={train_loss: tr_loss})
                    train_acc_summ = sess.run(train_acc_summary, feed_dict={train_acc: tr_acc})
                    summary_writer.add_summary(train_loss_summ, epoch)
                    summary_writer.add_summary(train_acc_summ, epoch)
                    summary_writer.flush()

                # Analyze validation subset
                if (epoch + 1) % val_period == 0:
                    print("Validating model")
                    val_losses = []
                    val_predictions = []
                    val_scores = []
                    val_y = []
                    for val_batch_x, val_batch_y, _ in dataset_val.sample_batches(batch_size, val_sample_size):
                        val_feed_dict = {img_: val_batch_x, y_: val_batch_y, K.learning_phase(): 0}
                        val_loss, val_pred, val_score = sess.run([loss, prediction, score], feed_dict=val_feed_dict)

                        val_losses.append(val_loss)
                        val_predictions.extend(val_pred.tolist())
                        val_scores.extend(val_score.tolist())
                        val_y.extend(val_batch_y.tolist())

                    val_loss = np.mean(val_losses)
                    val_acc = metrics.accuracy_score(np.array(val_y), np.array(val_predictions))

                    print('%s: epoch %d, validation loss = %.4f, acc = %.2f %%' % (datetime.now(), epoch,
                        val_loss, val_acc*100.))

                    # Validation summary
                    val_loss_summ = sess.run(validation_loss_summary, feed_dict={validation_loss: val_loss})
                    val_acc_summ = sess.run(validation_acc_summary, feed_dict={validation_acc: val_acc})
                    summary_writer.add_summary(val_loss_summ, epoch)
                    summary_writer.add_summary(val_acc_summ, epoch)
                    summary_writer.flush()

                if (epoch + 1) % save_period == 0:
                    print("Saving model parameters to checkpoint file")
                    saver.save(sess, output_modelpath_ckpt, global_step=epoch)

                if (epoch + 1) % summary_period == 0:
                    print('Running summary')
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, epoch)
                    summary_writer.flush()

            # Save final model
            print("Saving final model parameters to checkpoint file")
            saver.save(sess, output_modelpath_ckpt, global_step=epoch)
                
        except KeyboardInterrupt:
            print("Training interrupted")