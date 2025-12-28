# tensorflow 1.6
from tensorflow.python.keras._impl.keras.engine import Layer
from tensorflow.python.keras._impl.keras.initializers import Constant
# tensorflow >= 1.9
# from tensorflow.python.keras.engine import Layer
# from tensorflow.python.keras.initializers import Constant

import numpy as np
from scipy.special import logit
import tensorflow as tf


# Define MIL global pooling layer
class NoisyAndLayer(Layer):
    def __init__(self, trainable, **kwargs):
        self.trainable = trainable
        super(NoisyAndLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Define trainable layer weights and set `built` to true
        """
        # Get the number of samples, spatial super-pixels, and classes
        self.num_samples = input_shape[0]
        self.spatial_size = input_shape[1] * input_shape[2]
        self.num_classes = input_shape[-1]

        # Create trainable weight variable(s) for this layer.
        self.a = tf.exp(self.add_weight(name='ln_a', shape=(1, ), initializer=Constant(value=np.log(10)), trainable=self.trainable), name='a')
        self.b = tf.sigmoid(self.add_weight(name='logit_b', shape=(1, self.num_classes), initializer=Constant(value=logit(0.2)), trainable=self.trainable), name='b')

        super(NoisyAndLayer, self).build(input_shape)  # Be sure to call this, to set self.built = True

    def call(self, z):
        """
        Manipulate input tensor z, to obtain and return output tensor
        """
        z = tf.reshape(z, [-1, self.spatial_size, self.num_classes]) # (num bags, num instances, num classes)

        # Get the probability that each patch/instance belongs to each class 
        p = tf.sigmoid(z)

        p_bar = tf.reduce_mean(p, axis=1) # num_bags x num_classes
    
        offset = tf.sigmoid(tf.multiply(self.b, -self.a))
        numer = tf.subtract(tf.sigmoid(tf.multiply(self.a, tf.subtract(p_bar, self.b))), offset)
        denom = tf.subtract(tf.sigmoid(tf.multiply(self.a, tf.subtract(1.0, self.b))), offset)
        output = tf.divide(numer, denom, name="noisy_and")

        return output

    def compute_output_shape(self, input_shape):
        """
        Let keras know the size of the output tensor from call()
        """
        return (input_shape[0], input_shape[-1]) # (number of samples, number of classes)
