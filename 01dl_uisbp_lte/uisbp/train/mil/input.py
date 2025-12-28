from concurrent.futures import ThreadPoolExecutor
import inspect
import queue
import threading

import numpy as np

from ...augmentations import Augmentation


# Helper function(s)
def to_categorical(y, num_classes=None):
    """
    Converts a class vector or N-D array (integers) to binary 2-D array (matrix) or (N+1)-D array, respectively.
    E.g. for use with categorical_crossentropy.
    Args:
      y: vector or np.ndarray
        class vector or N-D array to be converted into a matrix or (N+1)-D array, respectively.
        (integers from 0 to num_classes). If y is negative, a zero vector is returned for that class
      num_classes: int
        total number of classes
    Returns:
      An (N+1)-D array representation of the input.
    
    A modification of tf.keras.utils.to_categorical that maps negative class values to a zero vector as is done by tf.one_hot()
    """
    y = np.array(y, dtype='int')

    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    # Turn y into a single vector
    y = y.ravel()

    if num_classes is None:
        num_classes = np.max(y) + 1

    n = y.shape[0]
    categorical = np.zeros((n, num_classes))

    # Get the location of non-negative y's
    non_negative_bool = y >= 0

    # Set the values
    categorical[np.arange(n)[non_negative_bool], y[non_negative_bool]] = 1

    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


class Dataset:
    """
    Data generator for images and whole image labels
    """
    def __init__(self, X, y, ids, preprocess_fn=None, seed=None, augmentation=None, categorical=True, 
        input_queue_size=8, use_compose=False):  
        """
        Args:
            X (np.ndarray): numpy array of images with size (n_images, n_rows, n_cols, n_channels)
            y (np.ndarray): numpy array of images with size (n_images, n_rows, n_cols, n_channels). The image mask array
                If n_channels is 1, we assume whole-image binary classification.
                If n_channels > 1, we assume whole-image multi-label classification. 
                Note: For segmentation, which occurs per pixel, the classification is Multi-clas
            ids (np.ndarray): numpy array of image IDs with size (n_images, )
            preprocess_fn (Callable): Preprocessing function with arguments X, y returning processed X and y arrays
            seed (int): Random seeding for data shuffling
            augmentation (list): List of ``Augmentation`` instances (defaults to identity function)
            categorical (bool): Whether to return one-hot encoded vectors for each image classification label or the label
            input_queue_size (int): The size of the input data queue
            use_compose (bool): Whether to compose the list of augmentations or to sample one of them at random using the augmentation weight 
        """
        self.X, self.y, self.ids = X, y, ids

        self.index_array = np.arange(self.X.shape[0], dtype=np.int32)

        # Set preprocess function to identity if none specified
        if preprocess_fn is None:
            self._preprocess = lambda x, y: (x, y)
        else:
            self._preprocess = preprocess_fn

        self.categorical = categorical

        self.shuffled = False

        self.input_queue_size = input_queue_size

        self.use_compose = use_compose

        # Get image size
        self.img_size = self.X.shape[1: 3]

        # Get number of mask channels
        self.n_channels = self.y.shape[3]

        if self.n_channels == 1:
            self.binary_task = True
        else:
            self.binary_task = False

        # Get the number of classes
        self.num_classes = 2
        if not self.binary_task:
            self.num_classes = self.n_channels

        # Get labels from segmentation maps
        if self.binary_task:
            self.y_labels = np.array(np.sum(self.y, axis=(1, 2, 3)) > 0, dtype=np.uint8)

            if self.categorical:
                # Get one-hot labels
                self.y_labels_cat = np.array(to_categorical(self.y_labels, num_classes=self.num_classes), dtype=np.float32)
        else:
            # Label indicator format / inclusive-class format
            self.y_labels = np.array(np.sum(self.y, axis=(1, 2)) > 0, dtype=np.uint8) # (num_samples, num_classes)

            if self.categorical:
                # Exclusive-class format
                # Convert binary multi-label into binary multi-class labels
                # For example, in a 3-class problem, [0, 1, 1] represents the presence of class 1 and 2 and the absence of class 0. This should map to 
                # [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
                # Note: tf.keras's to_categorical maps negatives similar to python negative indexing, so -1 does not map to a zero vector. Use locally defined to_categorical instead
                class_labels = np.arange(1, self.num_classes + 1) # [1, 2, ..., num_classes]
                self.y_labels_cat = np.array(to_categorical((self.y_labels * class_labels) - 1, num_classes=self.num_classes), dtype=np.float32)

        # Convert labels to float
        self.y_labels = self.y_labels.astype(np.float32)

        # Set augmentation to identity if none
        if augmentation is None:
            augmentation = [Augmentation(fn=lambda x,y: (x,y), kwargs={}, weight=1)]

        # Get augmentation functions and set probability
        self.augs = augmentation
        self.probs = np.array([float(a.weight) for a in self.augs])
        self.probs /= np.sum(self.probs)

        # Set random seed
        self.batch_rng = np.random.RandomState(seed) # Controls the batching of data

        # Get an offset for seeding augmentation
        self.aug_seed_offset = self.batch_rng.randint(int(1e6))

        self.data_size = self.size()


    def shuffle(self):
        """
        Shuffle the sample indexes
        """
        self.index_array = np.arange(len(self.index_array))
        self.batch_rng.shuffle(self.index_array)
        self.shuffled = True


    def batches(self, batch_size, dynamic_seed=0):
        """
        Get batches of data until the entire dataset is consumed

        Args:
            batch_size (int): The number of samples in each batch of data
            dynamic_seed (int): An integer that is set from a variable that is monotonic with iterations (optimization steps)
                For example, epoch number
        Yields:
            (tuple): (sample image, sample label, sample id)
        """      
        for x, y, i in self._batches_fast(self.index_array, batch_size, input_queue_size=self.input_queue_size, dynamic_seed=dynamic_seed):
            yield x, y, i

    
    def sample_batches(self, batch_size, max_num_samples, dynamic_seed=0):
        """
        Get batches of data until `max_num_samples` samples are consumed

        Args:
            batch_size (int): The number of samples in each batch of data
            max_num_samples (int): The maximum number of samples to obtain from the dataset
            dynamic_seed (int): An integer that is set from a variable that is monotonic with iterations (optimization steps)
                For example, epoch number
        Yields:
            (tuple): (sample image, sample label, sample id)
        """
        L = min(max_num_samples, len(self.index_array)) 

        for x, y, i in self._batches_fast(self.index_array[:L], batch_size, input_queue_size=self.input_queue_size, dynamic_seed=dynamic_seed):
            yield x, y, i


    def _load_image(self, idx, seed=None): 
        """
        Args:
            idx (int): The index of the sample image to be loaded and transformed
            seed (int): The seed for the random number generator used to perform data augmentation

        Returns:
            (tuple): (Transformed sample image, sample label, image index in dataset)
        """
        # Create a random number generator
        rng = np.random.RandomState(seed)

        # Apply preprocessing and augmentation transform
        b_x, b_y = self.apply_transform(self.X[idx], self.y[idx], rng=rng)
        
        # Image classification label
        y = self.y_labels[idx]
        if self.categorical:
            y = self.y_labels_cat[idx]
        
        s = (b_x, y, idx)   

        return s


    def _batches_fast(self, index_array, batch_size, input_queue_size=8, dynamic_seed=None):
        """        
        Get batches of data

        Args:
            index_array (np.ndarray): A set of sample indexes to be extracted from the dataset
            batch_size (int): The number of samples in each batch of data
            input_queue_size (int): The size of the input data queue
            dynamic_seed (int): An integer that is set from a variable that is monotonically increasing
                with iterations (optimization steps). For example, epoch number

        Yields:
            (tuple): (sample image, sample label, sample id)
        """
        n = len(index_array)

        def load(index_array, q, batch_size):
            n = len(index_array)
            for i in range(0, n, batch_size):
                stop_idx = min(i + batch_size, self.data_size)
                sub = index_array[i: stop_idx]

                # Control randomness in augmentation
                # Create seeds for reproducibility using local seeds, external dynamic seeds, and a random seed offset
                if isinstance(dynamic_seed, int):
                    seeds = np.array(sub) + dynamic_seed * self.data_size + self.aug_seed_offset
                else:
                    seeds = [None] * len(sub)
                
                img_out = [self._load_image(sub[ix], seeds[ix]) for ix in range(len(sub))]
                
                img = np.concatenate(tuple([b[0][np.newaxis] for b in img_out]), axis=0).astype(np.float32)
                labels = np.array([b[1] for b in img_out], dtype=np.float32) 
                ids = np.array([b[2] for b in img_out]) # image IDs

                collect = (img, labels, ids)
                 
                # Add to queue
                q.put(collect)

            # Indicator for end of transmission
            q.put(None)

        # This must be larger than twice the batch_size
        q = queue.Queue(maxsize=input_queue_size)

        # Background loading CT process
        p = threading.Thread(target=load, args=(index_array, q, batch_size))
        # Daemon child is killed when parent exits
        p.daemon = True
        p.start()

        for i in range(0, n, batch_size):            
            item = q.get()

            if item is None:
                break

            yield item

        
    def apply_transform(self, image, mask, rng=None):
        """
        Either applies a composition of transfroms from an augmentation list or 
        randomly applies transfrom from augmentation list based on its weight.

        Args:
            image (np.ndarray): Image array with shape (n_rows, n_cols, n_channels)
            mask (np.ndarray): Mask array with shape (n_rows, n_cols, n_channels)
            rng (RandomState): An instance of NumPy RandomState. 
                Controls the choice of augmentation and augmentation parameters

        Returns:
            (tuple): (transformed image, transformed mask)
        """        
        # preprocess
        image, mask = self._preprocess(image, mask)

        if self.use_compose:
            # Compose augmentation functions
            for idx in range(len(self.augs)):
                # Get augmentation function to apply
                aug = self.augs[idx]

                # Include random number generator to kwargs
                if len(inspect.getfullargspec(aug.fn)[0]) > 2:
                    aug.kwargs["rng"] = rng

                # Apply augmentation
                image, mask = aug.fn(image.astype(np.float32), mask.astype(np.float32), **aug.kwargs)
        else:
            # Get augmentation function to apply
            idx = rng.choice(np.arange(len(self.augs)), p=self.probs)
            aug = self.augs[idx]

            # Include random number generator to kwargs
            if len(inspect.getfullargspec(aug.fn)[0]) > 2:
                aug.kwargs["rng"] = rng

            # Apply augmentation
            image, mask = aug.fn(image.astype(np.float32), mask.astype(np.float32), **aug.kwargs)

        # digitize mask
        mask[np.nonzero(mask)] = 1.0

        return image, mask


    def size(self):
        """ Size of index_array """
        return len(self.index_array)