from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import Iterator
#from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
from collections import defaultdict
import cv2

from .augmentations import Augmentation
from .transform_utils import resize, crop

class ImageMaskDataGenerator(Iterator):
    """
    Data generator for images and masks.

    Args:
        X: numpy array of images with size (n_images, n_rows, n_cols, n_channels)
        y: numpy array of images with size (n_images, n_rows, n_cols, n_channels)
        batch_size: Number of images per batch
        preprocess_fn: Preprocessing function with arguments X, y returning processed X and y arrays
        shuffle: Option to shuffle the data between epochs (default True)
        seed: Random seeding for data shuffling
        augmentation: List of ``Augmentation`` instances (defaults to identity function)
        output_labels: Option to output binary class labels instead of masks (for MIL)
        output_mask_and_labels: Option to output both the mask and labels (for multi-output networks)
        output_labels_only: Option to output binary class labels instead of masks (for classification)
    """
    def __init__(self, X, y, batch_size=32, preprocess_fn=None,
                 shuffle=True, seed=None, augmentation=None,
                 output_labels=False, output_mask_and_labels=False,
                 output_labels_only=False):

        if output_labels and output_mask_and_labels:
            raise ValueError('Cannot set both output_labels and output_mask_and_labels')

        # Set preprocess function to identity if none specified
        if preprocess_fn is None:
            self._preprocess = lambda x, y: (x, y)
        else:
            self._preprocess = preprocess_fn

        self.X, self.y = X, y

        # Get one-hot labels from image maps
        self.y_labels = np.array(np.sum(self.y, axis=(1,2,3)) > 0, dtype=np.uint8)
        self.y_labels_cat = np.array(to_categorical(self.y_labels, num_classes=2), dtype=K.floatx())

        # set augmentation to identity if none
        if augmentation is None:
            augmentation = [Augmentation(fn=lambda x,y: (x,y), kwargs={})]

        # dump augmentation functions
        print('augmentation fn count in ImageMaskDataGenerator.__init__', len(augmentation))
        for aug in augmentation:
            print("augmeatation fn name", aug.fn.__name__)

        # get augmentation functions and set probability
        self.augs = augmentation

        # flag to output labels instead
        self.output_labels = output_labels
        self.output_labels_only = output_labels_only

        # flag to ouput mask and labels
        self.output_mask_and_labels = output_mask_and_labels

        super().__init__(X.shape[0], batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, index_array):
        """
        Given index array of samples, returns a minibatch of images and masks.

        Args:
            index_array (list): indices of sample images and masks

        Returns:
            batch_x: minibatch of images
            batch_y: minibatch of masks
        """
        batch_x = np.zeros(tuple([len(index_array)] + list(self.X.shape)[1:]), dtype=K.floatx())
        batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            batch_x[i], batch_y[i] = self.apply_transform(self.X[j], self.y[j])

        if self.output_labels:
            return batch_x, [self.y_labels_cat[index_array], self.y_labels_cat[index_array]]
        elif self.output_mask_and_labels:
            return batch_x, [batch_y, self.y_labels[index_array]]
        elif self.output_labels_only:
            return batch_x, self.y_labels_cat[index_array]
        else:
            return batch_x, batch_y


    def next(self):
        """
        For python 2.x.

        Returns:
            iterator: The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            try:
                index_array = next(self.index_generator)
            except ValueError:
                index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


    def apply_transform(self, image, mask):
        """
        Applies all random transforms in a composition.

        Args:
            image: Image array with shape (n_rows, n_cols, n_channels)
            mask: Mask array with shape (n_rows, n_cols, n_channels)

        Returns:
            (transformed image, transformed mask)
        """

        # preprocess
        image, mask = self._preprocess(image, mask)

        # compose augmentations
        for aug in self.augs:
            image, mask = aug.fn(image, mask, **aug.kwargs)

        # digitize mask
        mask[np.nonzero(mask)] = 1.0

        return image, mask



class DataSet:
    """ Data set container for storing generators.

    Attributes:
        trn_dl: Data loader (i.e. `ImageMaskDataGenerator`) for training set
        val_dl: Data loader (i.e. `ImageMaskDataGenerator`) for validation set
        test_dl: Data loader (i.e. `ImageMaskDataGenerator`) for test set

    """
    def __init__(self, trn, val, batch_size=32, augs=None, test=None, **kwargs):
        """
        Args:
            trn (tuple): Tuple of (imgs, masks) ndarrays for training set
            val (tuple): Tuple of (imgs, masks) ndarrays for validation set
            batch_size (int): Batch size for training (default 32)
            augs (tuple): Tuple of `Augmentation`s. (default None)
            test (tuple): Tuple of (imgs, masks) ndarrays for test set (default None)
            kwargs: Any additional keyword arguments to pass to `ImageMaskDataGenerator`
        """

        # set up generators
        self.trn_dl = ImageMaskDataGenerator(trn[0], trn[1], batch_size=batch_size,
                                             augmentation=augs, **kwargs)

        self.val_dl = ImageMaskDataGenerator(val[0], val[1], batch_size=batch_size, **kwargs)

        if test is not None:
            self.test_dl = ImageMaskDataGenerator(test[0], test[1], batch_size=batch_size,
                                                  shuffle=False, **kwargs)
        else:
            self.test_dl = None


def get_video_dict(ids):
    """
    Returns dictionary with unique video name as key and frame indices as values.

    Args:
        ids: Filename ids

    Returns:
        dict: video index dictionary keyed on video name
    """
    vdict = defaultdict(lambda: [])
    for ct, idx in enumerate(ids):
        if 'frame' in idx:
            label = idx.split('_frame')[0]
        else:
            labels = idx.split('_')
            label = '_'.join(labels[:-1])
        vdict[label].append(ct)

    # sort by frame number
    for key, val in vdict.items():
        frame_no = np.array([float(idx.split('_')[-1]) for idx in ids[val]])
        vdict[key] = np.array(val)[np.argsort(frame_no)]
    return vdict

def get_split_ids(ids, test_size=0.2, seed=123):
    # set random state
    random_state = np.random.RandomState(seed)

    # get video ids
    vdict = get_video_dict(ids)

    # randomly permuted ids
    rids = random_state.permutation(list(vdict.keys()))

    nsplit = int((1 - test_size) * len(rids))

    train_ids = np.concatenate([vdict[key] for key in rids[:nsplit]])
    test_ids = np.concatenate([vdict[key] for key in rids[nsplit:]])

    return train_ids, test_ids


def split_data(X, Y, ids, test_size=0.2, seed=123, **kwargs):
    train_ids, test_ids = get_split_ids(ids, test_size=test_size, seed=seed)

    return X[train_ids], X[test_ids], Y[train_ids], Y[test_ids], ids[train_ids], ids[test_ids]


class DataLoader:

    def __init__(self, datapath, dataset='apr'):

        self.X_train = np.load(f'{datapath}/imgs_train.npy')
        self.Y_train = np.load(f'{datapath}/imgs_mask_train.npy')
        self.ids_train = np.genfromtxt(f'{datapath}/imgs_fname_train.txt', dtype=np.unicode)

        self.X_test = np.load(f'{datapath}/imgs_test.npy')
        self.Y_test = np.load(f'{datapath}/imgs_mask_test.npy')
        self.ids_test = np.genfromtxt(f'{datapath}/imgs_fname_test.txt', dtype=np.unicode)

        if dataset in ['binary', 'multi', 'apr', 'may']:
            self.split_data = split_data
        else:
            raise ValueError(f'Unknown dataset {dataset}')

        if dataset in ['may', 'multi']:
            self.labels = np.loadtxt(f'{datapath}/labels.txt', dtype=np.unicode)
        else:
            self.labels = None

    def resize(self, imgs, size, interpolation=cv2.INTER_AREA):

        if imgs.shape[-1] in [1, 3]:
            return np.array([resize(crop(img, 0, 0, min(img.shape[0], img.shape[1])),
                                    size, interpolation=interpolation) for img in imgs])
        else:
            ret_array = np.zeros((imgs.shape[0], size, size, imgs.shape[-1]), dtype=np.uint8)
            for ct, img in enumerate(imgs):
                ret_array[ct, ...] = np.dstack([resize(crop(img[..., ii][..., None], 0, 0, min(img.shape[0], img.shape[1])),
                                                       size, interpolation=interpolation) for ii in range(img.shape[-1])])
            return ret_array

    def _filter_imgs(self, imgs, masks, ids):
        y_labels = np.array(np.sum(masks, axis=(1, 2, 3)) > 0, dtype=np.uint8)
        idx = y_labels == 1
        return imgs[idx], masks[idx], ids[idx]

    def get_train_valid_split(self, seed=123, test_size=0.2, return_test=True, size=416, filter_nerve_images=False):
        X = self.resize(self.X_train, size)
        Y = self.resize(self.Y_train, size, interpolation=cv2.INTER_NEAREST)

        X_train, X_valid, Y_train, Y_valid, ids_train, ids_valid = self.split_data(
            X, Y, self.ids_train, seed=seed, test_size=test_size, scenario='subject')

        if filter_nerve_images:
            X_train, Y_train, ids_train = self._filter_imgs(X_train, Y_train, ids_train)
            X_valid, Y_valid, ids_valid = self._filter_imgs(X_valid, Y_valid, ids_valid)

        if return_test:
            X_test, Y_test, ids_test = self.get_test_data(size=size)
            if filter_nerve_images:
                X_test, Y_test, ids_test = self._filter_imgs(X_test, Y_test, ids_test)
            return X_train, Y_train, ids_train, X_valid, Y_valid, ids_valid, X_test, Y_test, ids_test
        else:
            return X_train, Y_train, ids_train, X_valid, Y_valid, ids_valid

    def get_test_data(self, size=416):
        return self.resize(self.X_test, size), self.resize(self.Y_test, size, interpolation=cv2.INTER_NEAREST), self.ids_test
