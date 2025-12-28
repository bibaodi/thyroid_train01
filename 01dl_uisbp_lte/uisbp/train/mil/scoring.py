import tensorflow as tf


"""
Classification related
"""
def create_prediction_binary(pool_out, thr=0.5):
    """
    Args:
        pool_out: tf.Tensor 
            The output of the MIL pooling layer. 
            Size: (Number of samples, 1)

    Returns:
        pred: tf.Tensor
            The class predictions
    """        
    pred = tf.reshape(tf.cast(pool_out > thr, tf.int32), (-1, ))
    
    return pred


def create_score_binary(pool_out):
    """
    Args:
        pool_out: tf.Tensor 
            The output of the MIL pooling layer. 
            Size: (Number of samples, 1)
    
    Returns:
        score: tf.Tensor
            The score for the predicted class
    """        
    score = tf.reshape(pool_out, (-1, ))
    
    return score


def create_prediction_multi_label(fc_out, thr=0.5):
    """
    Args:
        fc_out: tf.Tensor 
            The output of the fully connected (FC) layer following the MIL pooling layer with sigmoid activation. 
            Size: (Number of samples, Number of classes)

    Returns:
        pred: tf.Tensor
            The class predictions
    """       
    pred = tf.cast(fc_out > thr, tf.int32)
    
    return pred


def create_score_multi_label(fc_out):
    """
    Args:
        fc_out: tf.Tensor 
            The output of the fully connected (FC) layer following the MIL pooling layer with sigmoid activation.
            Size: (Number of samples, Number of classes)
    
    Returns:
        score: tf.Tensor
            The score for the predicted classes
    """        
    score = fc_out

    return score