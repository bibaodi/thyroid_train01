'''
all functions are from process_apr_2018_data.ipynb (2018/05/07)

Move funtions to estimate_apr_helper.py in order to make code more readable
'''

import os
import shutil
from collections import defaultdict
from glob import glob
import re
import cv2
import numpy as np

from uisbp.transform_utils import crop, center_crop, resize

# The followings are only used for apr BP data
# linux pattern
IMG_FILENAME_PATTERN = re.compile(r"/(?P<hospital_id>\d+)_(?P<patient_number>\d+)_(?P<patient_name>[a-zA-Z]+)_(?P<scan>BP(?:(?:_L)|(?:_R))?)_(?P<sequence>\d+)/(?P<img_type>[a-zA-Z_0-9]+)/(?P<filename>[a-zA-Z_0-9]+)\..*")
# windows pattern
# IMG_FILENAME_PATTERN = re.compile(r"\\(?P<hospital_id>\d+)_(?P<patient_number>\d+)_(?P<patient_name>[a-zA-Z]+)_(?P<scan>BP(?:(?:_L)|(?:_R))?)_(?P<sequence>\d+)\\(?P<img_type>[a-zA-Z_0-9]+)\\(?P<filename>[a-zA-Z_0-9]+)\..*")

def get_raw_image_filenames(img_root_dir, img_type='positive', extension='jpg'):
    """
    Returns full path to all image files of type `img_type`

    Args:
        img_root_dir (str): Full path to root directory of stored images
        img_type (str): Image type options are ['positive', 'negative]
        extension (str): File-type extension [default 'jpg']

    Returns:
        (list): List of image filenames
    """
    pattern = os.path.join(img_root_dir, f"**/{img_type}/*{extension}")
    img_filenames = sorted(glob(pattern, recursive=True))

    return img_filenames

def get_imgs_output_filepath(img_filepath, root_output_dir, mask=False):
    """
    Makes output filename for new cropped image.

    Args:
        img_filepath (str): Full path to raw image file
        root_output_directory (str): The root of the output directory to store new cropped images

    Returns:
        str: Output filepath
    """

    img_info = IMG_FILENAME_PATTERN.search(img_filepath)
    if img_info is None:
        raise ValueError(f'{img_filepath} not valid')
    output_filename = f"{img_info['filename']}_mask.bmp" if mask else f"{img_info['filename']}.bmp"
    return os.path.join(root_output_dir, img_info['img_type'], output_filename)

def make_mask(maskfile, mask_size=(824, 540)):
    
    bboxes = np.genfromtxt(maskfile, dtype=np.int32, delimiter='\t')
    bboxes = bboxes[None] if np.ndim(bboxes) == 1 else bboxes
    bb = np.zeros(mask_size, dtype=np.uint8)
    for bbox in bboxes:
        cv2.rectangle(bb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
        
    return bb

def process_raw_images_to_file(img_root_dir, img_type='positive', output_dir='processed_images', clahe=False):
    """
    Process raw images to cropped images with just ultrasound.

    Args:
        img_root_dir (str): Full path to root directory of stored images
        img_type (str): Image type options are ['1', '0', 'mark']
        output_dir (str): Root output directory for cropped images
    """

    os.makedirs(f'{output_dir}/{img_type}', exist_ok=True)
    
    size_dict = defaultdict(lambda: 0)
    
    # get both images and masks
    if img_type == 'positive':
        img_files = get_raw_image_filenames(img_root_dir, img_type=img_type, extension='jpg')
        mask_files = get_raw_image_filenames(img_root_dir, img_type=img_type, extension='bboxes.tsv')
        
        for img_file, mask_file in zip(img_files, mask_files):
            
            if img_file.split('/')[-1].split('.')[0] != mask_file.split('/')[-1].split('.')[0]:
                raise ValueError(f'Image file {img_file} and mask file {mask_file} do not match!')
                
            try:
                img_outdir = get_imgs_output_filepath(img_file, root_output_dir=output_dir, mask=False)
                mask_outdir = get_imgs_output_filepath(mask_file, root_output_dir=output_dir, mask=True)
            except ValueError:
                print(f'{img_file} is not valid filename')
                continue
                
            img = cv2.imread(img_file, 0)
            mask = make_mask(mask_file, mask_size=img.shape)
            if clahe:
                img = adjust_hist(img)
            cv2.imwrite(img_outdir, img)
            cv2.imwrite(mask_outdir, mask)
            
            size_dict[img.shape] += 1
        print(size_dict)
            
    elif img_type == 'negative':
        img_files = get_raw_image_filenames(img_root_dir, img_type=img_type, extension='jpg')
        for img_file in img_files:
            try:
                img_outdir = get_imgs_output_filepath(img_file, root_output_dir=output_dir, mask=False)
            except ValueError:
                print(f'{img_file} is not valid filename')
                continue
                
            img = cv2.imread(img_file, 0)
            if clahe:
                img = adjust_hist(img)
            cv2.imwrite(img_outdir, img)
            size_dict[img.shape] += 1
        print(size_dict)
    
    else:
        raise ValueError(f'Unknown image type {img_type}')
                
def get_ids(filenames, img_type='positive'):
    img_ids = []
    for img in filenames:
        if img_type == 'negative':
            xs = img.split('/')[-1].split('.bmp')[0].split('_')
            img_ids.append('_'.join(xs))
        else:
            xs = img.split('/')[-1].split('.bmp')[0].split('_mask')[0]
            img_ids.append(xs)
    return img_ids


def get_np_files_from_images(datapath, crop_fn, size=None):
    """
    Reads in processed images and masks and stores in a numpy array.

    Args:
        datapath (str): Base path to images
        size (int): Output (square) size of images

    Returns:
        np.ndarray: n_image x size x size x 1 array of images
        np.ndarray: n_image x size x size x 1 array of masks
        np.ndarray: n_image array of unique file names
    """
    
    if crop_fn is None:
        def crop_fn(img, size=(824, 540), interpolation=cv2.INTER_AREA):
            return cv2.resize(img, size, interpolation=interpolation)

    imgs, masks, ids = [], [], []

    # nerve images and masks
    imgfiles = glob(f'{datapath}/positive/*bmp')
    x = sorted(list(filter(lambda x: 'mask' not in x.lower(), imgfiles)))
    y = sorted(list(filter(lambda x: 'mask' in x.lower(), imgfiles)))
    xids = get_ids(x, img_type='positive')
    yids = get_ids(y, img_type='positive')

    xdict = {key: val for key, val in zip(xids, x)}
    ydict = {key: val for key, val in zip(yids, y)}
    diff = list(set(xids).symmetric_difference(set(yids)))
    intersection = sorted(list(set(xids).intersection(set(yids))))
    if diff:
        print(f'Warning: {diff} does not contain both an image and a mask')
    for idx in intersection:
        imgfile, maskfile = xdict[idx], ydict[idx]
        # print("imgfile", imgfile)
        imgs.append(resize(crop_fn(cv2.imread(imgfile, 0))[..., None], size=size))
        masks.append(resize(crop_fn(cv2.imread(maskfile, 0))[..., None], size=size, interpolation=cv2.INTER_NEAREST))
        # print("imgfile", imgfile, imgfile.split('/')[-1].split('.')[0])
        ids.append('positive/' + imgfile.split('/')[-1].split('.')[0])
        
    # non-nerve images and masks
    imgfiles = glob(f'{datapath}/negative/*bmp')
    x = sorted(list(filter(lambda x: 'mask' not in x.lower(), imgfiles)))
    for imgfile in x:
        imgs.append(resize(crop_fn(cv2.imread(imgfile, 0))[..., None], size=size))
        ids.append('negative/' + imgfile.split('/')[-1].split('.')[0])
        masks.append(np.zeros((size, size, 1), dtype=np.uint8))

    img_array = np.array(imgs)
    mask_array = np.array(masks)
    print(img_array.shape, mask_array.shape, len(ids))
    return img_array, mask_array, np.array(ids)

def crop_fn(img):
    r, c, *_ = img.shape
    
    if r >= c:
        return crop(img, 0, 0, c)
    else:
        return center_crop(img)
    

def imagesToNpy(image_dir, out_dir):

    shutil.rmtree
    tempdir = f'{out_dir}/processed/'
    shutil.rmtree(tempdir, ignore_errors=True)

    # process postitive, negative, and masks
    process_raw_images_to_file(image_dir, output_dir=tempdir)
    process_raw_images_to_file(image_dir, img_type='negative', 
                                output_dir=tempdir)

    imgs, masks, ids = get_np_files_from_images(tempdir, crop_fn=crop_fn, size=480)

    print("shapes: ", imgs.shape, masks.shape, ids.shape)

    np_outdir = f'{tempdir}/np_data/'
    os.makedirs(np_outdir, exist_ok=True)

    np.save(f'{np_outdir}/imgs_test.npy', imgs)
    np.save(f'{np_outdir}/imgs_mask_test.npy', masks)
    np.savetxt(f'{np_outdir}/imgs_fname_test.txt', ids, fmt='%s')

def loadImageData(out_dir):
    datapath = f'{out_dir}/processed/np_data/' ## np_outdir
    imgs = np.load(f'{datapath}/imgs_test.npy')
    masks = np.load(f'{datapath}/imgs_mask_test.npy')
    ids = np.loadtxt(f'{datapath}/imgs_fname_test.txt', dtype=np.unicode)
    return imgs, masks, ids
    