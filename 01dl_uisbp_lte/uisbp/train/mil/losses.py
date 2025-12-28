import numpy as np
import tensorflow as tf


"""
Loss related
"""
def binary_cross_entropy(y_true=None, y_pred=None, epsilon=1e-10, pos_weight=1.0):
    """
    Args:
        y_true: tf.Tensor
            The ground truth labels in binarized format
        y_pred: tf.Tensor
            The estimated probability of belonging to each class
        pos_weight: float
            Weight to be applied when y_true is 1
        epsilon: float
            An offset to avoid computing log(0)

    Returns:
        bce: tf.Tensor
            The binary cross entropy

    y_true and y_pred must have the same shape
    """
    if y_true is None or y_pred is None:
        ValueError("`y_true` and `y_pred` must be specified")

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = tf.identity(- tf.add(tf.multiply(pos_weight, tf.multiply(y_true, tf.log(y_pred + epsilon))), tf.multiply(1 - y_true, tf.log(1 - y_pred + epsilon))), name="bce")

    return bce


def weighted_binary_cross_entropy(y_true=None, y_pred=None, class_weights=None, epsilon=1e-10):
    """
    Args:
        y_true: tf.Tensor
            The ground truth labels in binarized format
        y_pred: tf.Tensor
            The estimated probability of belonging to each class
        class_weights: num_classes element vector
            The weight to be assigned to the classes
        epsilon: float
            An offset to avoid computing log(0)

    Returns:
        wbce: tf.Tensor
            The weighted binary cross entropy

    Raises:
        ValueError:
            If y_true or y_pred is not specified
            If the size of class_weights does not match the number of classes specified in y_true or y_pred

    y_true and y_pred must have the same shape
    """
    if y_true is None or y_pred is None:
        ValueError("`y_true` and `y_pred` must be specified")

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Get the number of classes
    num_classes = y_pred.get_shape()[-1].value

    if class_weights is None:
        class_weights = ones((num_classes, ), dtype=np.float32)

    if num_classes is not None:
        if num_classes != len(class_weights) and (num_classes > 2):
                raise ValueError("The size of `class_weights` must match the number of classes")

    if num_classes is None or num_classes <= 2:
        # Binary task
        # Compute the positive weight given that the negative (0) class has a weight of 1
        pos_weight = class_weights[1]/class_weights[0]

        # # Transform y_pred to logits for robust calculations and gradients
        # output = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        # output = tf.log(output / (1 - output))

        # # Compute weighted binary cross entropy
        # wbce = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=output, pos_weight=pos_weight)
        wbce = tf.identity(binary_cross_entropy(y_true=y_true, y_pred=y_pred, epsilon=epsilon, pos_weight=pos_weight), name="wbce")
    else:
        # Multiple classes
        # Convert class_weights to tf.Tensor
        class_weights_tensor = tf.constant(np.array(class_weights).reshape(-1), dtype=tf.float32)

        # Weigh binary cross entropy using class weights
        wbce = tf.multiply(binary_cross_entropy(y_true=y_true, y_pred=y_pred, epsilon=epsilon), class_weights_tensor, name="wbce")
    
    return wbce


def mil_loss_binary(y_true, pool_out, class_weights=None):
    """
    Args:
        y_true: tf.Tensor
            A tensor containing the class label indices starting from 0
        pool_out: tf.Tensor
            The output of the MIL pooling layer
        class_weights: num_classes element vector
            The weight to be assigned to the classes
    
    Returns:
        total_loss: scalar tf.Tensor
            The weighted binary cross entropy loss

    This loss is designed for use in binary classification tasks. In such cases, only the MIL pooling layer is needed. The following fully connected layer is unnecessary
    """
    # Weighted binary cross entropy loss for MIL pooling layer output
    class_loss = tf.reduce_sum(weighted_binary_cross_entropy(y_true=tf.reshape(y_true, (-1, )), y_pred=tf.reshape(pool_out, (-1, )), 
        class_weights=class_weights), name="class_loss")
  
    # Overall loss
    total_loss = tf.identity(class_loss, name="total_loss")

    # Add losses to a new collection for later tracking
    tf.add_to_collection('named_losses', class_loss)

    return total_loss


def mil_loss_multi_label(y_true, pool_out, fc_out, pool_coef=1.0, fc_coef=1.0, class_weights=None):
    """
    Args:
        y_true: tf.Tensor
            A tensor containing the class label in one-hot encoding form
        pool_out: tf.Tensor
            The output of the MIL pooling layer
        fc_out: tf.Tensor
            The output of the fully connected (FC) layer following the MIL pooling layer  
        pool_coef: float
            The coefficient of the loss on the output of the MIL pooling layer
        fc_coef: float
            The coefficient of the loss on the output of FC with sigmoid activation
        class_weights: num_classes element vector
            The weight to be assigned to the classes
    
    Returns:
        total_loss: scalar tf.Tensor
            The MIL loss function based on the paper "Classifying and segmenting microscopy images with deep multiple instance learning"
            modified to include possible weights `pool_coef` and `fc_coef` on the loss components and class weights for imbalanced datasets.

    This loss function is designed for Multi-label classification tasks.
    """
    # Set parameters with None values
    if pool_coef is None:
        pool_coef = 1.0
    if fc_coef is None:
        fc_coef = 1.0

    # Get the number of classes
    num_classes = y_true.get_shape()[-1].value

    if class_weights is None:
        class_weights = ones((num_classes, ))

    if num_classes != len(class_weights):
        raise ValueError("The size of `class_weights` must match the number of classes")

    y_true = tf.cast(y_true, tf.float32)
    pool_out = tf.cast(pool_out, tf.float32)
    fc_out = tf.cast(fc_out, tf.float32)

    # Weighted binary cross entropy loss for MIL pooling layer output
    mil_pool_loss = tf.identity(pool_coef * tf.reduce_sum(weighted_binary_cross_entropy(y_true=y_true, y_pred=pool_out, class_weights=class_weights)), name="mil_pool_loss")
   
    # Weighted binary cross entropy loss for post-activation FC layer output
    class_loss = tf.identity(fc_coef * tf.reduce_sum(weighted_binary_cross_entropy(y_true=y_true, y_pred=fc_out, class_weights=class_weights)), name="class_loss")
    
    # Overall loss
    total_loss = tf.identity(mil_pool_loss + class_loss, name="total_loss")

    # Add losses to a new collection for later tracking
    tf.add_to_collection('named_losses', mil_pool_loss)
    tf.add_to_collection('named_losses', class_loss)

    return total_loss
