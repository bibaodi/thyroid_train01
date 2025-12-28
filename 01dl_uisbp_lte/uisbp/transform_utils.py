import math

import numpy as np
import cv2

def crop(img: np.ndarray, row_start: int, col_start: int, size: int) -> np.ndarray:
    """
    Crop image to size `size` starting from row `row_start` and column `col_start`.

    Args:
        img (ndarray): Single image array.
        row_start (int): Starting row for crop (new rows row_start:row_start+size)
        col_start (int): Starting column for crop (new cols col_start:col_start+size)
        size (int): Cropped image size (assumed square)

    Returns:
        ndarray: `size` x `size` cropped image
    """
    return img[row_start:row_start+size, col_start:col_start+size]

def resize(img: np.ndarray, size: int=None, interpolation: int=cv2.INTER_AREA) -> np.ndarray:
    """
    Returns a square resized image using `cv2.resize`

    Args:
        img (ndarray): Single image array.
        min_size (int): Optional output size (otherwise uses minimum or rows and cols)
        interpolation (int): cv2 interpolation method (default cv2.INTER_AREA)

    Returns:
        ndarray: resized image array
    """
    r, c, nch = img.shape
    if size is None:
        size = min(r, c)
    return cv2.resize(img, (size, size), interpolation=interpolation).reshape(size, size, nch)

def center_crop(img: np.ndarray, min_size: int=None) -> np.ndarray:
    """
    Returns a center crop of an image

    Args:
        img (ndarray): Single image array.
        min_size (int): Optional output size (otherwise uses minimum or rows and cols)

    Returns:
        ndarray: Center cropped image
    """
    r, c, *_ = img.shape
    if min_size is None:
        min_size = min(r, c)
    start_r = math.ceil((r - min_size) / 2)
    start_c = math.ceil((c - min_size) / 2)
    return crop(img, start_r, start_c, min_size)

def scale_min(img: np.ndarray, size: int, interpolation: int=cv2.INTER_AREA) -> np.ndarray:
    """
    Scales the image so that the smallest axis is of size targ.

    Args:
        img (ndarray): Single image array.
        size (int): target size
        interpolation (int): cv2 interpolation method (default cv2.INTER_AREA)

    Returns:
        Minimum axis scaled image
    """
    r, c, nch = img.shape
    ratio = size / min(r, c)
    sz = (max(math.floor(c * ratio), size), max(math.floor(r * ratio), size))
    return cv2.resize(img, sz, interpolation=interpolation).reshape(sz[1], sz[0], nch)

def zoom_cv(img: np.ndarray, zoom: float, interpolation: int=cv2.INTER_AREA,
            mode: int=cv2.BORDER_CONSTANT) -> np.ndarray:
    """
    Zoom image.

    Args:
        img (ndarray): Single image array.
        zoom (float): Zoom percentage (0.1 = 10 percent zoom in, -0.1 = 10 percent zoom out)
        interpolation (int): cv2 interpolation method (default cv2.INTER_AREA)
        mode (int): cv2 border mode method (default cv2.BORDER_CONSTANT)

    Return:
        Zoomed image
    """
    if zoom == 0:
        return img

    r, c, nch = img.shape
    M = cv2.getRotationMatrix2D((c / 2, r / 2), 0, zoom + 1)
    return cv2.warpAffine(img, M, (c, r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation).reshape(r, c, nch)

def stretch_cv(img: np.ndarray, stretch_row: float, stretch_col: float,
               interpolation: int=cv2.INTER_AREA) -> np.ndarray:
    """
    Stretch Image in vertical and/or horizontal direction

    Args:
        img (ndarray): Single image array.
        stretch_row (float): Stretch range along rows [should be in 0 to 1 range]
        stretch_col (float): Stretch range along cols [should be in 0 to 1 range]
        interpolation (int): cv2 interpolation method (default cv2.INTER_AREA)

    Returns:
        stretched image
    """
    if stretch_row == 0 and stretch_row == 0:
        return img
    r, c, nch = img.shape
    x = cv2.resize(img, None, fx=stretch_row+1, fy=stretch_col+1,
                   interpolation=interpolation)
    nr, nc, *_ = x.shape
    cr = (nr - r) // 2
    cc = (nc - c) // 2
    return x[cr:r+cr, cc:c+cc].reshape(r, c, nch)

def rotate_cv(img: np.ndarray, deg: float, mode: int=cv2.BORDER_CONSTANT,
              interpolation: int=cv2.INTER_AREA) -> np.ndarray:
    """
    Rotates an image by deg degrees

    Args:
        img (ndarray): Single image array.
        deg (float): degree to rotate.
        interpolation (int): cv2 interpolation method (default cv2.INTER_AREA)
        mode (int): cv2 border mode method (default cv2.BORDER_CONSTANT)

    Returns:
        ndarray: Rotated image
    """
    r, c, nch = img.shape
    M = cv2.getRotationMatrix2D((c / 2, r / 2), deg, 1)
    return cv2.warpAffine(img, M, (c,r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation).reshape(r, c, nch)


def elastic_cv(img: np.ndarray, alpha: float, sigma: float,
               mode: int=cv2.BORDER_CONSTANT, interpolation: int=cv2.INTER_AREA,
               random_state: np.random.RandomState=np.random) -> np.ndarray:
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

     Args:
        img (ndarray): Single image array.
        alpha (int): scale multiplier on gaussian deformations
        sigma (int): standard deviation of gaussian filter
        random_state (RandomState): Random state generator

    Returns:
        ndarray: Elastically transformed image
    """

    shape = img.shape[:2]
    r, c, nch = img.shape

    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur(random_state.rand(*shape) * 2 - 1, (blur_size, blur_size), 0) * alpha
    dy = cv2.GaussianBlur(random_state.rand(*shape) * 2 - 1, (blur_size, blur_size), 0) * alpha

    y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    mapy, mapx = (y + dy).astype(np.float32), (x + dx).astype(np.float32)

    res_x = cv2.remap(img, mapx, mapy, interpolation=interpolation, borderMode=mode)

    return res_x.reshape(r, c, nch)

def adjust_gain(img: np.ndarray, gain: float=1.0) -> np.ndarray:
    """
    Adjust the gain of the image via an exponential transform.

    Args:
        img (np.ndarray): Single image array
        gain (float): Gain factor (gain=1 means no change)

    Returns:
        ndarray: New image with gain adjustment
    """
    new_img = (img / 255) ** gain * 255
    return new_img.astype(np.uint8)

def adjust_hist(img, do_clahe=True):
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(7, 7))
        return clahe.apply(img)[:, :, None]
    else:
        minV = float(np.min(img.flatten()))
        maxV = float(np.max(img.flatten()))
    return np.array(255 * (img - minV) / (maxV - minV), dtype=np.uint8)
