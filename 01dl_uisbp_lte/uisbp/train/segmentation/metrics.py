from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


def mean_dice_coef(y_true, y_pred, smooth=1e-6):
    top = 2 * K.sum(y_true * y_pred, axis=[1, 2, 3]) + smooth
    bot = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3
                                                              ]) + smooth
    return K.mean(top / bot)


def mean_jaccard_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(
        y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection) / (union + smooth))


def weighted_binary_cross_entropy(y_true, y_pred, weights):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
    c1 = y_true * tf.math.log(y_pred) * weights[0] + (
        1 - y_true) * tf.math.log(1 - y_pred) * weights[1]
    return -tf.reduce_mean(tf.reduce_sum(c1, axis=-1))


def weighted_cross_entropy(y_true, y_pred, weights):
    y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
    return -tf.reduce_mean(
        tf.reduce_sum(y_true * tf.math.log(y_pred) * weights, axis=-1))


class CrossEntropyJaccardLoss:
    def __init__(self,
                 jaccard_weight=0,
                 class_weights=None,
                 num_classes=1,
                 smooth=1,
                 special_indexs=[],
                 special_weights=[]):
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.smooth = smooth
        self.special_indexs = special_indexs
        self.special_weights = special_weights

    def __call__(self, y_true, y_pred):
        if self.class_weights is not None:
            weights = tf.convert_to_tensor(self.class_weights,
                                           dtype=tf.float32)
        else:
            weights = tf.convert_to_tensor(
                np.ones(shape=K.int_shape(y_true)[-1]), dtype=tf.float32)

        if self.num_classes == 1:
            wce_loss = weighted_binary_cross_entropy(y_true, y_pred, weights)
        else:
            wce_loss = weighted_cross_entropy(y_true, y_pred, weights)

        jaccard_loss = K.epsilon()
        if self.jaccard_weight:
            cls_weight = 1 / (self.num_classes - len(self.special_weights) + sum(self.special_weights))
            for cs in range(self.num_classes):
                for spec_index in self.special_indexs:
                    if cs == spec_index:
                        wgt = cls_weight * self.special_weights[self.special_indexs.index(spec_index)]
                    else:
                        wgt = cls_weight
                    intersection = K.sum(y_true[..., cs] * y_pred[..., cs])
                    union = K.sum(y_true[..., cs]) + K.sum(
                        y_pred[..., cs]) + K.epsilon()
                    dice_origin = ((intersection + self.smooth) /
                                 (union - intersection + self.smooth))
                    dice_reward_tn = (union - intersection + self.smooth) / (union + self.smooth)
                    dice_fixed = dice_origin 
                    jaccard_loss += (dice_fixed * wgt)

        return wce_loss - self.jaccard_weight * K.log(jaccard_loss)

    @property
    def __name__(self):
        return CrossEntropyJaccardLoss.__name__


class MeanDice:
    def __init__(self, labels, cls):
        self.labels = labels
        self.cls = cls
        self.cls_index = list(labels).index(cls)

    def __call__(self, y_true, y_pred):
        idx = self.cls_index
        top = 2 * K.sum(y_true[..., idx] * y_pred[..., idx], axis=[1, 2])
        bot = K.sum(y_true[..., idx], axis=[1, 2]) + K.sum(
            y_pred[..., idx], axis=[1, 2]) + K.epsilon()
        return K.mean(top / bot)

    @property
    def __name__(self):
        return f'mean_dice_{self.cls.lower()}'
