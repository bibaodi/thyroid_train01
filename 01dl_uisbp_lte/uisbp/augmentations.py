import numpy as np

from keras.preprocessing.image import flip_axis, apply_transform, transform_matrix_offset_center

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# Augmentation class
class Augmentation:
    """
    Augmentation container class.

    Args:
        fn (callable): Augmentation function that inputs (X,y) and outputs augmented (X,y)
        kwargs (dict): Any keyword arguments to augmentation function
        weight (int): Weight to use in real-time data augmentation (default 1)
    """
    def __init__(self, fn, kwargs=None, weight=1):
        self.fn = fn
        self.kwargs = kwargs or {}
        self.weight = weight


def identity(image, mask):
    return image, mask

    
def elastic_transform(image, mask, alpha, sigma, rng=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

     Args:
        x (np.ndarray): Input image tensor. Must be 3D.
        y (np.ndarray): Input mask tensor. Must be 3D.
        alpha (int): scale multiplier on gaussian deformations.
        sigma (int): standard deviation of gaussian filter.
        rng (RandomState): An instance of NumPy RandomState.

    Returns:
        transformed image and mask tensors.
 
    """
    if rng is None:
        rng = np.random.RandomState(None)
 
    shape = image.shape[:-1]

    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image_out = map_coordinates(image[:, :, 0], indices, order=1, mode='reflect').reshape(shape + (1,))

    # Handle multiple channels in mask
    num_channels = mask.shape[-1]
    mask_out = np.zeros_like(mask)
    for ix in range(num_channels):
        mask_out[:, :, ix] = map_coordinates(mask[:, :, ix], indices, order=1, mode='reflect').reshape(shape)

    return image_out, mask_out


def flip_lr(x, y):
    """
    Flip image along y-axis.
    Args:
        x (np.ndarray): Input image tensor. Must be 3D.
        y (np.ndarray): Input mask tensor. Must be 3D.

    Returns:
        flipped image and mask tensors.
    """
    return flip_axis(x, 1), flip_axis(y, 1)


def random_flip_lr(x, y, p=0.5, rng=None):
    """
    Flip image along y-axis.
    Args:
        x: Input image tensor. Must be 3D.
        y: Input mask tensor. Must be 3D.
        rng (RandomState): An instance of NumPy RandomState.

    Returns:
        flipped image and mask tensors.
    """
    if rng is None:
        rng = np.random.RandomState(None)

    flip_prob = rng.uniform()

    if flip_prob < p:
        return flip_lr(x, y)
    else:
        return x, y


def random_zoom(x, y, zoom_range, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0., rng=None):
    """
    Performs a random spatial zoom of a Numpy image tensor.
    Args:
        x (np.ndarray): Input image tensor. Must be 3D.
        y (np.ndarray): Input mask tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        rng (RandomState): An instance of NumPy RandomState.
    Returns:
        Zoomed Numpy image and mask tensor.
    Raises:
        ValueError: if `zoom_range` isn't a tuple.
    """
    if rng is None:
        rng = np.random.RandomState(None)

    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)

    return x, y


def random_rotation(x, y, rg, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0., rng=None):
    """Performs a random rotation of a Numpy image tensor.
    Args:
        x (np.ndarray): Input image tensor. Must be 3D.
        y (np.ndarray): Input mask tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        rng (RandomState): An instance of NumPy RandomState.
    Returns:
        Rotated Numpy image tensor.
    """
    if rng is None:
        rng = np.random.RandomState(None)

    theta = np.pi / 180 * rng.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)

    return x, y


def random_shear(x, y, intensity, row_index=0, col_index=1, channel_index=2, fill_mode='constant', cval=0., rng=None):
    """
    Performs a random spatial shear of a Numpy image tensor.
    Args:
        x (np.ndarray): Input image tensor. Must be 3D.
        y (np.ndarray): Input mask tensor. Must be 3D.
        intensity (np.ndarray): Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        rng (RandomState): An instance of NumPy RandomState.
    Returns:
        Sheared Numpy image tensor.
    """
    if rng is None:
        rng = np.random.RandomState(None)

    shear = rng.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)

    return x, y


def random_shift(x, y, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0., rng=None):
    """
    Performs a random spatial shift of a Numpy image tensor.
    Args:
        x (np.ndarray): Input image tensor. Must be 3D.
        y (np.ndarray): Input mask tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        rng (RandomState): An instance of NumPy RandomState.
    Returns:
        Shifted Numpy image tensor.
    """
    if rng is None:
        rng = np.random.RandomState(None)
 
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = rng.uniform(-hrg, hrg) * h
    ty = rng.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)

    return x, y


def adjust_gain(img: np.ndarray, gain: float=1.0) -> np.ndarray:
    """
    Adjust the gain of the image via an exponential transform.
    Args:
        img (np.ndarray): Single image array.
        gain (float): Gain factor (gain=1 means no change).
    Returns:
        ndarray (np.ndarray): New image with gain adjustment
    """
    new_img = (img / 255) ** gain * 255

    return new_img.astype(np.uint8)


def random_gain(x, y, gain_range, rng=None):
    """
    Randomly adjust gain of input image, keeping the mask image unaltered.
    Args:
        x (np.ndarray): Input image tensor.
        y (np.ndarray): Input mask tensor.
        gain_range: Tuple of floats; gain range to select from.
        rng (RandomState): An instance of NumPy RandomState.        
    Returns:
        Numpy image with gain adjusted and mask tensor retained
    Raises:
        ValueError
            When `gain_range` isn't a 2-element tuple. 
            When any element of gain_range is not positive.
            When gain_range[0] > gain_range[1].
    """
    if rng is None:
        rng = np.random.RandomState(None)

    if len(gain_range) != 2:
        raise ValueError('gain_range should be a tuple or list of two floats. Received arg: %s' % gain_range)

    if any(np.array(gain_range) <= 0):
        raise ValueError('gain_range should lie in the positive interval. Received arg: %s' % gain_range)

    if gain_range[0] > gain_range[1]: 
        raise ValueError('gain_range[0] must be less than or equal to gain_range[1]. Received arg: %s' % gain_range)

    if gain_range[0] == 1 and gain_range[1] == 1:
        gain = 1
    else:
        gain = rng.uniform(gain_range[0], gain_range[1], 1)    

    x = adjust_gain(x, gain=gain)

    return x, y


# Default augmentations and keyword arguments
default_augs = tuple([
    Augmentation(fn=identity),
    Augmentation(fn=elastic_transform, kwargs={'sigma':10, 'alpha':100}),
    Augmentation(fn=random_flip_lr),
    Augmentation(fn=random_zoom, kwargs={'zoom_range':(0.8, 1.2)}),
    Augmentation(fn=random_rotation, kwargs={'rg': 10.0, 'fill_mode':'reflect'}),
    Augmentation(fn=random_shear, kwargs={'intensity': 0.5}),
    Augmentation(fn=random_shift, kwargs={'wrg': 0.2, 'hrg': 0.2, 'fill_mode':'reflect'})
    ])
