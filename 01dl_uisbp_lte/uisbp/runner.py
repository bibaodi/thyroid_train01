import numpy as np
import cv2
import tensorflow as tf

import uisbp
from .post_processing import post_process_imgs, resize_mask
from .preprocess import crop_img


def preprocess_img(img, size=416, do_clahe=False):
    """
    Bottom crop image to square multiple of 32 and then resize. Will optionally apply
    CLAHE histogram equalization.

    Args:
        img (np.ndarray): Input (nrow x ncol x nchan) image array
        size (int):
            Target square size of image (if size is None, will find
            the nearest multiple of 32 of the smallest side of image)
        do_clahe (boolean): Option to do CLAHE histogram equalization

    Returns:
        np.ndarray: Square cropped and resized image
    """
    cimg = crop_img(img)
    r, c, *_= cimg.shape
    nch = 1
    if r != size:
        rimg = cv2.resize(cimg, (size, size), interpolation=cv2.INTER_AREA).reshape(size, size, nch)
    else:
        rimg = cimg.reshape(r, c, nch)

    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(7, 7))
        return clahe.apply(rimg)[...,None]
    else:
        return rimg

def load_graph(frozen_graph_filename):
    """ Load tensorflow graph from protobuffer file.

    Args:
        frozen_graph_filename (str): Full path to frozen pb file

    Returns:
        tensorflow graph
    """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def get_classification_thresh(img_size):
    """ Get MIL classifier threshold based on image size."""
    raise Exception('DO not know how to decide this threshold')
    size_vec = [96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416]
    thresh_vec = [0.661, 0.405, 0.614, 0.976, 0.930, 0.888, 0.949, 0.897, 0.711, 0.862, 0.723]
    thresh = dict(zip(size_vec, thresh_vec))
    return thresh[img_size]

class ModelRunner:

    """
    Load model for predicting.

    `load_model` will load network model from proto buffer file.
    `predict_on_img` will predict and (optionally) do post processing on a single image
    `predict` method will predict and do post processing on all input images.
    """
    def __init__(self, do_clahe=False, output_type='mask'):
        """
        Initialize

        Args:
            do_clahe (boolean): Option to do CLAHE histogram equalization
            output_type (str): 'classifier' for image-level probability, 'mask' for pixel-level probability
        """
        self.img_size = None
        self.do_clahe = do_clahe
        self.sess = None
        if output_type not in ['mask', 'classifier']:
            raise ValueError(f'Unknown output_type {output_type}. Must be either mask or classifier')
        self.output_type = output_type

    def load_model(self, model='unet', dset='feb', img_size=416):
        """
        Load model into memory. Depending on the image size, model, and dataset a different
        model will be loaded. For speed, use a small input size (i.e. 128).

        Warning: All input images should have size that is divisible by 32.

        Args:
            model (str): Model type. options: [unet, linknet, mil]
            dset (str): Use model trained on this dataset options: [feb, mar]
            img_size (int): Square size of image
        """
        self.img_size = img_size
        model_path = f'{uisbp.__path__[0]}/graphs/'
        if img_size % 32 != 0:
            raise ValueError('Input image shape must be divisible by 32')

        # segmentation only has a subset of models
        if model in ['unet', 'linknet'] and img_size not in [96, 128, 160, 224, 416, 448]:
            raise ValueError('Input size must be in [96, 128, 160, 224, 416] for segmentation')

        # load graph
        import os
        graph_name = f'{model_path}/{model}_{dset}_{img_size}/output_graph.pb'
        print(f"model file:", graph_name)
        if not os.path.exists(graph_name):
            raise PermissionError('graph file not exist!')
        self.graph = load_graph(graph_name)

        # get input and output tensors
        ops = self.graph.get_operations()
        self.x, self.y = ops[0].outputs[0], ops[-1].outputs[0]

    def predict_on_img(self, sess, img, post_process=True, channel_first_output=False):
        """
        Predict on a single image

        Args:
            sess: Tensorflow session
            img (np.ndarray): Input nrow x ncol x nch image
            post_process (bool): Boolean to perform post processing

        Returns:
            np.ndarray: prediction
        """
        raise NotImplementedError

    def predict(self, X_test, post_process=True, channel_first_model=False):
        """
        Predict on a array of images

        Args:
            X_test (np.ndarray): Input n_image x nrow x ncol x nch image
            post_process (bool): Boolean to perform post processing
            channel_first_model : model ouput is nch x nrow x ncol ot nrow x ncol x nch

        Returns:
            np.ndarray: n_image x nrow x ncol x nch mask prediction probability
        """
        Y_pred = []
        with tf.Session(graph=self.graph) as sess:
            for ii, img in enumerate(X_test):
                Y_pred.append(self.predict_on_img(sess, img, post_process=post_process, channel_first_model=channel_first_model))
        if self.output_type == 'classifier':
            return np.array(Y_pred).flatten()
        return np.array(Y_pred)

class SegmentationRunner(ModelRunner):

    """
    Load Unet or Linknet model for prediction

    `load_model` will load network model from proto buffer file.
    `predict_on_img` will predict and (optionally) do post processing on a single image
    `predict` method will predict and do post processing on all input images.
    """
    def __init__(self, thresh=0.26, pc=0.005, do_clahe=False):
        """
        Initialize

        Args:
            thresh (float): Probability threshold for output maps (i.e. value > thresh == 1)
            pc (float): Minimum percentage of total pixels that mask can occupy.
            do_clahe (boolean): Option to do CLAHE histogram equalization
        """
        super().__init__(do_clahe, output_type='mask')
        self.thresh = thresh
        self.pc = pc

    def predict_on_img(self, sess, img, post_process=True, channel_first_model=False):
        """
        Predict on a single image

        Args:
            sess: Tensorflow session
            img (np.ndarray): Input nrow x ncol x nch image
            post_process (bool): Boolean to perform post processing
            channel_first_model : model output format

        Returns:
            np.ndarray: nrow x ncol x nch mask prediction
        """
        orig_size = img.shape[0]
        new_img = preprocess_img(img, size=self.img_size, do_clahe=self.do_clahe)
        ypred = sess.run(self.y, feed_dict={self.x: new_img[None]})[0]

        # convert channel X row X col to row x col x channel
        if channel_first_model:
            ypred = np.moveaxis(ypred, 0, 2)
        
        if post_process:
            ypred = post_process_imgs(ypred[None], self.thresh, self.pc, cc=True)[0][0]
        return resize_mask(ypred, orig_size)


class ClassificationRunner(ModelRunner):

    """
    Load MIL model for prediction.

    `load_model` will load network model from protobuffer file.
    `predict_on_img` will predict and (optionally) do post processing on a single image
    `predict` method will predict and do post processing on all input images.
    """
    def __init__(self, thresh=None, do_clahe=False):
        """
        Initialize

        Args:
            thresh (float): Probability threshold for classification (i.e. value > thresh == 1)
            do_clahe (boolean): Option to do CLAHE histogram equalization
        """
        super().__init__(do_clahe, output_type='classifier')
        self.thresh = thresh

    def predict_on_img(self, sess, img, post_process=True, **kwargs):
        """
        Predict on a single image

        Args:
            sess: Tensorflow session
            img (np.ndarray): Input nrow x ncol x nch image
            post_process (bool): Boolean to perform post processing

        Returns:
            np.ndarray: classifier prediction
        """
        new_img = preprocess_img(img, size=self.img_size, do_clahe=self.do_clahe)
        ypred = sess.run(self.y, feed_dict={self.x: new_img[None]})[0]
        if post_process:
            self.thresh = self.thresh or get_classification_thresh(self.img_size)
            ypred = np.array(ypred > self.thresh, dtype=np.int32)

        return ypred
