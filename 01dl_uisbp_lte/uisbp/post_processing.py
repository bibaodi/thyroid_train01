import numpy as np
import cv2
import io

from .transform_utils import resize


def grays_to_RGB(img):
    """
    Convert a 1-channel grayscale image into 3 channel RGB image

    Args:
        img (ndarray): Array of images

    Returns:
        ndarrray: Stacked array with 3 channels
    """
    return np.dstack((img, img, img))

def get_contours(mask):
    """ Return contours from mask.

    Args:
        img (ndarray): Array for single mask

    Returns:
        list: List of cv2 contours
    """
    ret, threshed_img = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def get_bbox_from_mask(mask):
    """
    Return bounding box mask, given contour mask.

    Args:
        mask (ndarray): Input mask array

    Returns:
        ndarray: Bounding box mask 0 outside box, 255 inside box
    """
    if np.sum(mask) == 0:
        return mask

    ret, threshed_img = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    for c in contours:
        # get the bounding rect
        bbox.append(cv2.boundingRect(c))

    bbox = np.array(bbox)
    # get bounds
    minx, miny = np.min(bbox[:,0]), np.min(bbox[:,1])
    maxx = np.max(bbox[:,0] + bbox[:, 2])
    maxy = np.max(bbox[:,1] + bbox[:, 3])

    bb = np.zeros_like(mask)
    cv2.rectangle(bb, (minx, miny), (maxx, maxy), 255, -1)

    return bb

def image_plus_mask(img, mask, mask2=None, inc_bbox=False):
    """
    Returns a copy of the grayscale image, converted to RGB,
    and with the edges of the mask added in red.

    Args:
        img (ndarray): Array for single image in grayscale or RGB
        mask (ndarray): Array for single mask (plotted in red)
        mask2 (ndarray): Additional array for single mask (plotted in cyan)
        inc_bbox (bool): If True, will plot bounding box around contours

    Returns:
        ndarray: Combined image + mask array
    """
    if np.ndim(img) == 2:
        img_color = grays_to_RGB(img)
    else:
        img_color = img
    contours = get_contours(mask)
    cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)
    if inc_bbox:
        bbox = get_bbox_from_mask(mask)
        contours = get_contours(bbox)
        cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)

    if mask2 is not None:
        contours = get_contours(mask2)
        cv2.drawContours(img_color, contours, -1, (0, 255, 255), 2)
        if inc_bbox:
            bbox = get_bbox_from_mask(mask2)
            contours = get_contours(bbox)
            cv2.drawContours(img_color, contours, -1, (0, 255, 255), 2)

    return img_color

def get_labels_from_mask(masks):
    """
    Get binary labels from input segmentation masks.
    Masks must be 0's and 1's only
    """
    return np.array(np.sum(masks, axis=(1,2,3)) > 0, dtype=np.uint8)

def threshold(imgs, thresh):
    """
    Digitizes image based on probability threshold.

    Args:
        imgs (ndarray): Full (n_image, n_row, n_col, n_channel) image array
        thresh (float): Threshold value between 0 and 1

    Returns:
        ndarray: Digitized mask array
    """
    ret = np.zeros(imgs.shape)
    nch = imgs.shape[-1]
    for ct, img in enumerate(imgs):
        if nch == 1:
            ret[ct,:,:,0] = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        else:     
            ret[ct,:,:,:] = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    return ret


def threshold_on_multi_channel(imgs, threshs):
    """
    Digitizes image based on probability threshold, each label has invidual threshold.

    Args:
        imgs (ndarray): Full (n_image, n_row, n_col, n_channel) image array
        threshs (list of float): Threshold value between 0 and 1 for multi-mask

    Returns:
        ndarray: Digitized mask array
    """
    ret = np.zeros(imgs.shape)
    nch = imgs.shape[-1]
    label_channel_thresh = np.array(threshs)
    for ct, img in enumerate(imgs):
            if nch == 1:
                ret[ct,:,:,0] = cv2.threshold(img, threshs[0], 1, cv2.THRESH_BINARY)[1]
            else:
                ret[ct,:,:,:] = np.greater(img, label_channel_thresh)
    return ret.astype(np.uint8)


def threshold_on_pixel_count(imgs, pc=0.005):
    """
    Threshold images on percentage of non-zero pixels. Zero out masks with
    pixel_percentage < pc.

    Args:
        imgs (ndarray): Full (n_image, n_row, n_col, n_channel) image array
        pc (float): Percentage of non-zero pixels

    Returns:
        ndarray: Filtered masks based on pixel count.
    """
    if not 0 < pc < 1:
        raise ValueError('pc must be in [0,1]')

    pixel_per = np.array([np.sum(y)/y.size for y in imgs])
    imgs[pixel_per<pc] = 0
    return imgs

def prune_components(img, min_area_per=0.002, deep_prune=True):
    """
    Post processing step to prune away small patches and ouput a convex
    hull of (at most) the two largest patches

    Args:
        img (np.ndarray): nrow x ncol x nchan thresholded input mask
        min_area_per (float): Minimum area percentage (i.e. patch area / image area) to be kept
        deep_prune : False - only reomve area smaller than min_area

    Returns:
        np.ndarray: Output mask
    """

    nch = img.shape[-1]
    masks = np.zeros_like(img)
    for ct in range(nch):
        # get contours from image
        image, contours, hier = cv2.findContours(img[:,:,ct].astype(np.uint8),
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for each contour
        mask = np.zeros(image.shape, dtype=np.uint8)

        min_area = mask.size * min_area_per

        # sort contours by area
        areas = np.array([cv2.contourArea(c) for c in contours])
        area_idx = np.argsort(areas)[::-1]
        areas = areas[area_idx]
        sorted_contours = np.array([c for c in contours])[area_idx]

        # get rid of small contours
        sorted_contours = np.array(sorted_contours)[areas>=min_area]
        
        if deep_prune:
            # keep no more than two boxes
            if len(sorted_contours) > 1:
                keep_areas = areas[areas>min_area]
                ratio = keep_areas[1] / keep_areas[0]
                sorted_contours = sorted_contours[:2] if ratio > 0.3 else [sorted_contours[0]]

            for cnt in sorted_contours:
                # get convex hull
                hull = cv2.convexHull(cnt)
                cv2.drawContours(mask, [hull], -1, 1, -1)

        else:        
            #for cnt in sorted_contours:
            cv2.drawContours(mask, sorted_contours, -1, 1, -1)
        masks[..., ct] = mask     

    return masks


def post_process_imgs_multi_label(imgs, thresh={}, pc=0.01, cc=True):
    """
    Apply post processing to images, each label have its own threshold.
    1. Threshold based on probability value (thresh)
    2. Threshold on pixel percentage (pc)
    3. Remove small connected components (cc)
    param:
        imgs: the list of numpy which is the prediction for a image.
        thresh: dict for label's threshold. e.g({'plaque': 1e-3, 'CA': 0.26})
        pc: the percentage for a label's pixel count in the full image
        cc: Remove small connected components
    """
    if not isinstance(thresh, dict):
        raise ValueError("thresh must be dict")
    if len(thresh) != imgs[0].shape[-1]:
        raise ValueError("thresh must be algn to prediction's last axis")
        
    Y_pred_thresh = threshold_on_multi_channel(imgs, list(thresh.values()))
    Y_pred_thresh = threshold_on_pixel_count(Y_pred_thresh, pc)
    if cc:
        Y_pred_thresh = np.array(list(map(lambda x: prune_components(x, deep_prune=False).reshape(x.shape), Y_pred_thresh)))
    y_thresh_labels = get_labels_from_mask(Y_pred_thresh)

    return Y_pred_thresh, y_thresh_labels


def post_process_imgs(imgs, thresh=0.26, pc=0.01, cc=True):
    """
    Apply post processing to images.

    1. Threshold based on probability value (thresh)
    2. Threshold on pixel percentage (pc)
    3. Remove small connected components (cc)
    """
    Y_pred_thresh = threshold(imgs, thresh)
    Y_pred_thresh = threshold_on_pixel_count(Y_pred_thresh, pc)
    if cc:
        Y_pred_thresh = np.array(list(map(lambda x: prune_components(x, deep_prune=False).reshape(x.shape), Y_pred_thresh)))
    y_thresh_labels = get_labels_from_mask(Y_pred_thresh)

    return Y_pred_thresh, y_thresh_labels

##### Post processing for Multi-label data #####
def resize_mask(mask, size, interpolation=cv2.INTER_NEAREST):
    """ Resize mask

    Args:
        mask (ndarray): Single mask array
        size (int): Targe output size
        interpolation (int): Interpolation method

    Returns:
        ndarray: Resized mask array
    """
    return np.dstack([resize(mask[..., ii][..., None], size=size, interpolation=interpolation)
                      for ii in range(mask.shape[-1])])


def convert_mask_to_lbl(mask, label_dict, thresh=0.5):
    """ Converts mask to multi-label color array

    Args:
        mask (ndarray): Single mask array
        label_dict (dict): Dictionary of label name, code number key, value pairs
        thresh (float): Threshold value to keep value. Below value -> background class

    Return:
        np.ndarray: Colormap array
    """
    lbl = np.zeros(mask.shape[:2], dtype=np.int32)
    ct = 0

    # threshold mask
    thresh_mask = np.max(mask, axis=-1) < thresh

    # find max
    amask = np.argmax(mask, axis=-1)
    for key in sorted(label_dict):
        if key != 'UncertainZone':
            idx = amask == ct
            lbl[idx] = label_dict[key]
            ct += 1

    # if below threshold, assign to background class
    lbl[thresh_mask] = 0
    return lbl

# copy from labelme.utils.draw-->
# similar function as skimage.color.label2rgb
def label2rgb(
    lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0, colormap=None,
):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    colormap = _validate_colormap(colormap, n_labels)
    colormap = (colormap * 255).astype(np.uint8)

    lbl_viz = colormap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz

def _validate_colormap(colormap, n_labels):
    if colormap is None:
        colormap = label_colormap(n_labels)
    else:
        assert colormap.shape == (colormap.shape[0], 3), \
            'colormap must be sequence of RGB values'
        assert 0 <= colormap.min() and colormap.max() <= 1, \
            'colormap must ranges 0 to 1'
    return colormap

def draw_label(label, img=None, label_names=None, colormap=None, **kwargs):
    """Draw pixel-wise label with colorization and label names.

    label: ndarray, (H, W)
        Pixel-wise labels to colorize.
    img: ndarray, (H, W, 3), optional
        Image on which the colorized label will be drawn.
    label_names: iterable
        List of label names.
    """
    import matplotlib.pyplot as plt

    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    colormap = _validate_colormap(colormap, len(label_names))

    label_viz = label2rgb(
        label, img, n_labels=len(label_names), colormap=colormap, **kwargs
    )
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'
                          .format(value=label_value, name=label_name))
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)
    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out
# copy from labelme.utils.draw <--

def make_labeled_images(imgs, masks, labels, size, thresh=0.5):
    """
    Make new images with predicted mask labels overlaid

    Args:
        imgs (ndarray): Full array of images
        masks (ndarray): Full array of multilabel masks
        labels (list): List of label names
        size (int): Target output size
        thresh (float): Threshold value to keep value. Below value -> background class

    Returns:
        List of labeled images
    """
    label_dict = {lab: ct for ct, lab in enumerate(labels[::-1])}
    rmasks = np.array([convert_mask_to_lbl(resize_mask(mask, size), label_dict, thresh=thresh) for mask in masks])

    limgs = []
    label_names = sorted(label_dict, key=label_dict.get)
    print(label_dict, label_names)
    for rmask, img in zip(rmasks, imgs):
        out = draw_label(rmask, img=img[..., 0], label_names=label_names)
        limgs.append(out)

    return limgs
