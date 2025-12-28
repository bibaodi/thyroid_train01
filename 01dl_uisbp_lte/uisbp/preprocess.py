from glob import glob
import re
import os
import math
from typing import Any
from collections import defaultdict
import json
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm

import cv2
import numpy as np

from uisbp.transform_utils import crop, resize, adjust_hist
"""
Functions to process ultrasound images and convert to numpy arrays. These functions assume that the directory
structure is that described below in the regex.
"""

# regex pattern for matching raw image filenames
# this is only used for binary data format until end of 2018. There is only BP data set in this format.
IMG_FILENAME_PATTERN = re.compile(
    r"/(?P<hospital_id>\d+)_(?P<patient_number>\d+)_(?P<patient_name>[a-zA-Z]+)_(?P<scan>BP(?:(?:_L)|(?:_R))?)_(?P<sequence>\d+)/(?P<img_type>[a-zA-Z_0-9]+)/(?P<filename>[a-zA-Z_0-9]+)\..*"
)

CROP_SIZE = 448


def crop_img(img):
    r, c = img.shape[:2]
    row_start = 0
    if r >= c:
        col_start = 0
        size = c
    else:
        size = r
        col_start = int(math.ceil((c - size) / 2))

    return crop(img, row_start, col_start, size)


def crop_img_and_remove_black_edge(img, true_width_left: int,
                                   true_width_right: int):
    r, c = img.shape[:2]
    true_width = true_width_right - true_width_left + 1
    if true_width < 1:
        raise ValueError("true width wrong!")
    row_start = 0
    if true_width > r:
        padding_size = (true_width - r, *img.shape[1:])
        padding_data = np.zeros(padding_size, dtype=np.uint8)
        img = np.concatenate((img, padding_data), axis=0)
        print(f"padding to image: {padding_size}")

    col_start = true_width_left
    size = true_width
    xshift = col_start
    yshift = row_start
    return img[row_start:row_start + size, col_start:col_start +
               size], xshift, yshift


def get_ids(filenames, img_type='positive'):
    img_ids = []
    for img in filenames:
        if img_type == 'negative':
            xs = img.split(os.sep)[-1].split('.bmp')[0].split('_')
            img_ids.append('_'.join(xs))
        else:
            xs = img.split(os.sep)[-1].split('.bmp')[0].split('_mask')[0]
            img_ids.append(xs)
    return img_ids


def merge_region(img, ksize, iters, morph_shape, dofill=True):
    kernel = cv2.getStructuringElement(morph_shape, (ksize, ksize))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)
    if dofill:
        img = closing.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, 1, cv2.FILLED)
        closing = img.astype(np.float)

    return closing


class BinaryLabelData:
    def __init__(self,
                 img_root_dir: str,
                 outdir: str,
                 img_filename_pattern: Any = IMG_FILENAME_PATTERN):
        self.img_root_dir = img_root_dir
        self.outdir = outdir
        self.img_filename_pattern = img_filename_pattern

    def get_raw_image_filenames(self, img_type='positive', extension='jpg'):
        """
        Returns full path to all image files of type `img_type`

        Args:
            img_type (str): Image type options are ['positive', 'negative]
            extension (str): File-type extension [default 'jpg']

        Returns:
            (list): List of image filenames
        """
        pattern = os.path.join(self.img_root_dir,
                               f"**{os.sep}{img_type}{os.sep}*{extension}")
        img_filenames = sorted(glob(pattern, recursive=True))

        return img_filenames

    def get_imgs_output_filepath(self,
                                 img_filepath,
                                 root_output_dir,
                                 mask=False):
        """
        Makes output filename for new cropped image.

        Args:
            img_filepath (str): Full path to raw image file
            root_output_directory (str): The root of the output directory to store new cropped images
            mask (bool): Boolean indicating whether or not image is mask

        Returns:
            str: Output filepath
        """
        img_info = self.img_filename_pattern.search(img_filepath)
        if img_info is None:
            raise ValueError(f'{img_filepath} not valid')
        output_filename = f"{img_info['filename']}_mask.bmp" if mask else f"{img_info['filename']}.bmp"
        return os.path.join(root_output_dir, img_info['img_type'],
                            output_filename)

    def make_mask(self, maskfile, mask_size=(824, 540)):
        """
        Make bounding box binary mask from mask file

        Args:
            maskfile (str): Full path to bounding box file
            mask_size (tuple): Tuple of height x width

        Returns:
            numpy array of binary mask file
        """
        bboxes = np.genfromtxt(maskfile, dtype=np.int32, delimiter='\t')
        bboxes = bboxes[None] if np.ndim(bboxes) == 1 else bboxes
        bb = np.zeros(mask_size, dtype=np.uint8)
        for bbox in bboxes:
            cv2.rectangle(bb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)

        return bb

    def process_raw_images_to_file(self,
                                   img_type='positive',
                                   output_dir='processed_images',
                                   clahe=False):
        """
        Process raw images to cropped images with just ultrasound.

        Args:
            img_root_dir (str): Full path to root directory of stored images
            img_type (str): Image type options are ['1', '0', 'mark']
            output_dir (str): Root output directory for cropped images
        """
        output_dir = f'{self.outdir}{os.sep}{output_dir}'
        os.makedirs(f'{output_dir}{os.sep}{img_type}', exist_ok=True)

        size_dict = defaultdict(lambda: 0)

        # get both images and masks
        if img_type == 'positive':
            img_files = self.get_raw_image_filenames(img_type=img_type,
                                                     extension='jpg')
            mask_files = self.get_raw_image_filenames(img_type=img_type,
                                                      extension='bboxes.tsv')

            for img_file, mask_file in zip(img_files, mask_files):

                if img_file.split(os.sep)[-1].split('.')[0] != mask_file.split(
                        os.sep)[-1].split('.')[0]:
                    raise ValueError(
                        f'Image file {img_file} and mask file {mask_file} do not match!'
                    )

                try:
                    img_outdir = self.get_imgs_output_filepath(
                        img_file, root_output_dir=output_dir, mask=False)
                    mask_outdir = self.get_imgs_output_filepath(
                        mask_file, root_output_dir=output_dir, mask=True)
                except ValueError:
                    print(f'{img_file} is not valid filename')
                    continue

                img = cv2.imread(img_file, 0)
                mask = self.make_mask(mask_file, mask_size=img.shape)
                if clahe:
                    img = adjust_hist(img)
                cv2.imwrite(img_outdir, img)
                cv2.imwrite(mask_outdir, mask)

                size_dict[img.shape] += 1
            print(size_dict)

        elif img_type == 'negative':
            img_files = self.get_raw_image_filenames(img_type=img_type,
                                                     extension='jpg')
            for img_file in img_files:
                try:
                    img_outdir = self.get_imgs_output_filepath(
                        img_file, root_output_dir=output_dir, mask=False)
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

    @staticmethod
    def get_np_files_from_images(datapath, size=None):
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
        imgs, masks, ids = [], [], []

        # nerve images and masks
        imgfiles = glob(f'{datapath}{os.sep}positive{os.sep}*bmp')
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
            imgs.append(
                resize(crop_img(cv2.imread(imgfile, 0))[..., None], size=size))
            masks.append(
                resize(crop_img(cv2.imread(maskfile, 0))[..., None],
                       size=size,
                       interpolation=cv2.INTER_NEAREST))
            ids.append(imgfile.split(os.sep)[-1].split('.')[0])

        # non-nerve images and masks
        imgfiles = glob(f'{datapath}{os.sep}negative{os.sep}*bmp')
        x = sorted(list(filter(lambda x: 'mask' not in x.lower(), imgfiles)))
        for imgfile in x:
            imgs.append(
                resize(crop_img(cv2.imread(imgfile, 0))[..., None], size=size))
            ids.append(imgfile.split(os.sep)[-1].split('.')[0])
            masks.append(np.zeros((size, size, 1), dtype=np.uint8))

        img_array = np.array(imgs)
        mask_array = np.array(masks)
        print(img_array.shape, mask_array.shape, len(ids))
        return img_array, mask_array, np.array(ids)


# copy from labelme, delete support to ’instance‘ type
def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        group_id = shape.get('group_id')
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get('shape_type', None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins

# handle with overlapped labels
def shapes_to_label_withkey(img_shape,
                            shapes,
                            label_name_to_value,
                            keylabels:list=[],
                            para_type='class'):
    assert para_type in ['class', 'instance']

    cls_mask = np.zeros(img_shape[:2], dtype=np.int32)

    total_shapes = len(shapes)
    drawed_shapes = 0
    # draw other labels first
    for shape in shapes:
        shape_label = shape['label']
        if shape_label not in keylabels:
            polygons = shape['points']
            cls_name = shape_label
            cls_id = label_name_to_value[cls_name]
            mask = shape_to_mask(img_shape[:2], polygons)
            cls_mask[mask] = cls_id
            drawed_shapes += 1
    if drawed_shapes == total_shapes:
        return cls_mask
    # draw key label in order to make sure key label mask overlap others
    for keylabel in keylabels:
        for shape in shapes:
            shape_label = shape['label']
            if shape_label == keylabel:
                polygons = shape['points']
                cls_name = shape_label
                cls_id = label_name_to_value[cls_name]
                mask = shape_to_mask(img_shape[:2], polygons)
                cls_mask[mask] = cls_id
                drawed_shapes += 1
                if drawed_shapes == total_shapes:
                    return cls_mask
    return cls_mask


def correct_json_amend_one_shape(shape, xshift, ratio, finalsize):
    labelname = shape['label']
    for point in shape['points']:
        x = round((point[0] - xshift) * ratio)
        y = round(point[1] * ratio)
        errormax = 10
        if 0 - x > errormax or 0 - y > errormax or \
                x - finalsize > errormax or y - finalsize > errormax:
            print(
                f"@correct_json_amend_one_shape:Out of range shape position {labelname} : ({point[0], point[1]}) to ({x, y})")
        point[0] = min(max(x, 0), finalsize - 1)
        point[1] = min(max(y, 0), finalsize - 1)


def get_json_from_file(json_file,
                       need_correct=False,
                       xshift=0,
                       ratio=1.0,
                       finalsize=448):
    """
    将json文件获取到内存中, 如果json文件不存在, 返回空json
    returns (tuple):
        True, json_data 文件存在时
        False, json_data 文件不存在时, 返回空json
    """
    empty_json = """
        {
            "shapes": [],
            "lineColor": [
                0,255,0,128
            ],
            "fillColor": [
                255,0,0,128
            ],
            "imagePath": "not_exist.file",
            "imageData": null
        }
        """
    file_exist = True
    if os.path.isfile(json_file):
        with open(json_file) as fin:
            data = json.load(fin)
            if need_correct:
                for shape in data["shapes"]:
                    correct_json_amend_one_shape(shape, xshift, ratio,
                                                 finalsize)
            return file_exist, data
    else:
        file_exist = False
        data = json.loads(empty_json)
        return file_exist, data


def get_image_actual_width_index_range_mem(imagebuff=None):
    """
        根据已经读入内存的图片的矩阵获取图像有效宽度, 输入参数类型为numpy.ndarray
        返回tuple(left, right) 代表有效列的索引
    """
    if not imagebuff.any() or not isinstance(imagebuff, np.ndarray):
        raise ValueError("param is not np.ndarray")
    if len(imagebuff.shape) != 2 and len(imagebuff.shape) != 3:
        raise ValueError(
            "image must in IMREAD_GRAYSCALE or IMREAD_COLOR(height, row, gray)!"
        )

    gray_img = imagebuff.sum(axis=2) if len(imagebuff) == 3 else imagebuff

    gray_img_t = gray_img.transpose()

    left = 0
    right = gray_img_t.shape[0] - 1
    for col in range(gray_img_t.shape[0]):
        column = gray_img_t[col]
        column.sort()
        top20 = column[-20:]
        if top20.mean() > 10:
            left = col
            break
    for col in range(gray_img_t.shape[0] - 1, -1, -1):
        column = gray_img_t[col]
        column.sort()
        top20 = column[-20:]
        if top20.mean() > 10:
            right = col
            break
    return (left, right)


def get_image_actual_width_index_range(imagefile=None):
    """
        获取图片的实际有效宽度的索引范围
    """
    if not imagefile:
        raise ValueError("imagefile is None")
    if not os.path.isfile:
        msg = f"imagefile {imagefile} not exist"
        raise ValueError(msg)
    img = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    if not img.size:
        raise ValueError("imagefile read None")
        return None
    else:
        return get_image_actual_width_index_range_mem(img)


def correct_image_size(img_array, zoom=False, zoomed_size=0):
    """
        crop+resize. 返回处理好的图片并返回X轴偏移以及缩放比例. 
    """
    #print(f"@@@debug:correct_image_size: array shape is {img_array.shape}, type({type(img_array)})")
    if not img_array.any() or not isinstance(img_array, np.ndarray):
        raise ValueError("param is not np.ndarray")
    if len(img_array.shape) != 2 and len(img_array.shape) != 3:
        raise ValueError(
            "image must in IMREAD_GRAYSCALE or IMREAD_COLOR(height, row, gray)!"
        )
    if zoomed_size < 1:
        raise ValueError(f"zoomed size must visualized. current is {zoomed_size}")
    true_width = get_image_actual_width_index_range_mem(img_array.copy())
    #print(f"@@@debug:correct_image_size: true_width is {true_width})")
    img_croped, xshift, _ = crop_img_and_remove_black_edge(
        img_array, true_width[0], true_width[1])
    #print(f"@@@debug:correct_image_size: img_croped shape is {img_croped.shape})")
    if zoom:
        img_array_new = cv2.resize(img_croped,
                                   dsize=(zoomed_size, zoomed_size),
                                   interpolation=cv2.INTER_AREA)
    else:
        img_array_new = img_croped
    ratio = img_array_new.shape[0] / img_croped.shape[0]
    #print(f"debug: ratio={img_array_new.shape[0] ,img_croped.shape[0]}")
    return img_array_new, xshift, ratio


def read_image_and_correct_it(imagefile=None, zoom=False, zoomed_size=448, clahe=True):
    """
    读取图片, 如果图片不是448*448的那么进行crop, resize, final_size=zoomed_size
    注意: 此函数调用后需要配套修改json的坐标
    """
    #print(f"read_image_and_correct_it: file is {imagefile}")
    if not imagefile:
        raise ValueError("imagefile is None")
    if not os.path.isfile:
        msg = f"imagefile {imagefile} not exist"
        raise ValueError(msg)
    img = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(7, 7))
        img = clahe.apply(img)
    if not img.size:
        raise ValueError("imagefile read None")
    #print(f"@@@debug:read_image_and_correct_it: array shape is {img.shape}, type({type(img)})")
    #xshift = 0
    #ratio = 1
    #if img.shape[0:2] == (CROP_SIZE, CROP_SIZE):
    #    new_img = img
    #else:
    new_img, xshift, ratio = correct_image_size(img, zoom=zoom, zoomed_size=zoomed_size)

    return (new_img, xshift, ratio)


class MultiLabelData:
    def __init__(self, img_root_dir: str, outdir: str):
        self.img_root_dir = img_root_dir
        self.outdir = outdir

    def get_raw_image_filenames(self, extension='png', abandon_list=[]):
        """
        Returns full path to all image files of type `img_type`

        Args:
            extension (str): File-type extension [default 'jpg']

        Returns:
            (list): List of image filenames
        """
        pattern = os.path.join(self.img_root_dir, f"**{os.sep}*{extension}")
        all_img_filenames = sorted(glob(pattern, recursive=True))
        
        img_filenames = []
        for fname in all_img_filenames:
            video_name = fname.split(os.sep)[-2].split('_frms')[0]
            if video_name in abandon_list:
                continue
            else:
                img_filenames.append(fname)
        
        ids = []
        for fname in img_filenames:
            base = fname.split(os.sep)[-2].split('_frms')[0]
            frame = fname.split(os.sep)[-1].split('.')[0].split('-')[-1]
            ids.append('_'.join([base, frame]))

        return img_filenames, ids

    def get_imgs_masks_labels(self,
                              extension='png',
                              use_uncertain=False,
                              merge_bp=False,
                              size=448,
                              keylabels:list=[], abandon_list=[]):
        assert len(keylabels) >= 0  # keylabels must be something
        # get filenames
        #filenames, ids = self.get_raw_image_filenames(extension=extension)
        filenames, ids = self.get_raw_image_filenames(extension=extension, abandon_list=abandon_list)
        # get label names
        label_dict = {'_background_': 0}
        ct = 0
        for fname in filenames:
            json_file = fname.split('.png')[0] + '.json'
            _, data = get_json_from_file(json_file, finalsize=size)
            for shape in data['shapes']:
                label = shape['label']
                if label not in label_dict:
                    ct += 1
                    label_dict[label] = ct

        label_names = list(sorted(label_dict.keys()))
        uncertain = 0
        if not use_uncertain and "UncertainZone" in label_names:
            label_names.pop(label_names.index('UncertainZone'))
            uncertain = 1

        # make masks
        imgs, masks = [], []
        for fname in tqdm(filenames, ncols=80, desc="Processing Files"):
            json_file = fname.split('.png')[0] + '.json'

            img_datas = read_image_and_correct_it(fname, zoom=True, zoomed_size=size)
            img = img_datas[0]
            xshift = img_datas[1]
            ratio = img_datas[2]
            # open jsonfile
            _, data = get_json_from_file(json_file,
                                         need_correct=True,
                                         xshift=xshift,
                                         ratio=ratio,
                                         finalsize=img.shape[0])
            #img = cv2.imread(fname, 0)
            #print("####", cv2.IMREAD_COLOR, "#",  cv2.IMREAD_GRAYSCALE, "img shape", img.shape)
            imgs.append(img)

            # [448, 448] all polygons cover each other in one axis
            if keylabels:  # keep keylabels mask when it overlapps with others
                lbl = shapes_to_label_withkey(img.shape, data['shapes'],
                                              label_dict, keylabels)
            else:  # not consider overlap between labels
                lbl = shapes_to_label(img.shape, data['shapes'],
                                            label_dict)

            # get mask
            ct = 0
            mask = np.zeros(img.shape + (len(label_dict) - uncertain, ),
                            dtype=np.uint8)  # [448, 448, 5~10]
            for (key, val) in sorted(label_dict.items()):
                if key != 'UncertainZone' or use_uncertain:
                    mask[lbl == val, ct] = 1
                    ct += 1

            if merge_bp:
                # merge regions on BP
                indexBP = label_names.index('BP')
                # only BP need to be merged until end of 2018
                assert indexBP >= 0

                newBPimg = merge_region(mask[..., indexBP],
                                        ksize=100,
                                        iters=1,
                                        morph_shape=cv2.MORPH_ELLIPSE,
                                        dofill=True)
                newBPmask = newBPimg > 0

                for i in range(len(label_names)):
                    if i == indexBP:
                        mask[..., i] = newBPimg
                    else:
                        mask[...,
                             i][newBPmask] = 0  # remove overlap with BP region

            masks.append(mask)
        # get arrays
        img_list = [img[..., None] for img in imgs]
        img_array = np.stack(img_list, axis=0)
        mask_array = np.zeros((len(masks), size, size, len(label_names)),
                              dtype=np.uint8)
        for ct, mask in enumerate(masks):
            mask_array[ct, ...] = np.dstack([
                resize(crop_img(mask[..., ii][..., None]), size)
                for ii in range(mask.shape[-1])
            ])

        return img_array, mask_array, label_names, label_dict, np.array(ids)

    def generate_black_bottom_imgs(self, black_it_ratio, img):
        """
        black_it_ratio: list of float or null. 
        img: ndarray
        """
        imgs = []
       
        for black_ratio in black_it_ratio:
            img_black = img.copy()
            non_black_height = math.ceil(img_black.shape[0] * (1 - black_ratio))
            padding_size = (img_black.shape[0] - non_black_height, *img_black.shape[1:])
            img_black[non_black_height:, ...] = np.zeros(padding_size, dtype=np.uint8)
            imgs.append(img_black)
        return imgs

    def generate_black_bottom_img_names(self, black_it_ratio, img_filename):
        """
        black_it_ratio: list of float or null. 
        img_filename: str: the image's filename without suffix. (eg.05_63_XX_Vessel_R_01_0010)
        """
        fnames = []
        for black_ratio in black_it_ratio:
            serial_number = img_filename.split('_')[-1]
            new_name_base = img_filename.split('_' + serial_number)[0]
            new_name_base = f"{new_name_base}b{black_ratio:.2}".replace('.', '')
            new_name = new_name_base + '_' + serial_number
            fnames.append(new_name)
        return fnames

    def get_imgs_masks_labels_2(self, extension='png', size=448, keylabels=[], abandon_list=[], black_it_ratio=[]):
        """
        20190903
        返回一组图片的矩阵, 图片叠加标注后的mask的矩阵. 因为需要判断FP,FN那么需要将每张图携带的标签带着. 返回ids为[(imagename, labels_set)]
        注意: 因为以label作为维度的mask是以label的名字的字母序进行排序, 所以从函数结果的label_dict获取labelnames的时候需要调用sorted
        todo:
            需要将keylabel机制修改为order机制, 确保小的label一定在上面可以看见.
        """
        assert len(keylabels) >= 0  # keylabels must be something
        # get filenames
        filenames, imgs_index = self.get_raw_image_filenames(extension=extension, abandon_list=abandon_list)

        # get label names and get labels for one image
        label_dict = {'_background_': 0}
        ct = 0
        uncertain_imgs = []
        for _index, fname in enumerate(filenames):
            json_file = fname.split('.png')[0] + '.json'
            _, data = get_json_from_file(json_file)
            labels = set()
            for shape in data['shapes']:
                label = shape['label']
                if "UncertainZone" in label:  # 'UncertainZone' 标签不进行数据训练
                    uncertain_imgs.append(fname)
                    continue
                labels.add(label)
                if label not in label_dict:
                    ct += 1
                    label_dict[label] = ct
            imgs_index[_index] = (imgs_index[_index], labels)

        for uncertain_img in uncertain_imgs:
            _index = filenames.index(uncertain_img)
            filenames.pop(_index)
            imgs_index.pop(_index)

        # make masks
        imgs, masks = [], []
        count_i = 0
        for fname in tqdm(filenames, ncols=80, desc="Processing Files:"):
            json_file = fname.split('.png')[0] + '.json'

            img_datas = read_image_and_correct_it(fname, zoom=True, zoomed_size=size)
            img = img_datas[0]
            xshift = img_datas[1]
            ratio = img_datas[2]
            # open jsonfile
            _, data = get_json_from_file(json_file,
                                         need_correct=True,
                                         xshift=xshift,
                                         ratio=ratio,
                                         finalsize=img.shape[0])
            if black_it_ratio:
                transformed_imgs = self.generate_black_bottom_imgs(black_it_ratio, img)
                imgs.extend(transformed_imgs)
                _index = count_i
                transformed_fnames = self.generate_black_bottom_img_names(black_it_ratio, imgs_index[_index][0])
                for _tname in transformed_fnames:
                    _new_img_label_tuple = (_tname, imgs_index[_index][1])
                    imgs_index.insert(_index + 1, _new_img_label_tuple)
            count_i += len(black_it_ratio) + 1
            imgs.append(img)

            # [448, 448] all polygons cover each other in one axis
            if keylabels:  # keep keylabels mask when it overlapps with others
                lbl = shapes_to_label_withkey(img.shape, data['shapes'],
                                              label_dict, keylabels)
            else:  # not consider overlap between labels
                lbl = shapes_to_label(img.shape, data['shapes'],
                                            label_dict)

            # get mask
            ct = 0
            mask = np.zeros(img.shape + (len(label_dict), ),
                            dtype=np.uint8)  # [448, 448, 5~10]
            for (key, val) in sorted(label_dict.items()):  # 将一维mask转化为多维mask(维度与label的个数一致)
                if key != 'UncertainZone':
                    mask[lbl == val, ct] = 1
                    ct += 1
            transformed_masks = self.generate_black_bottom_imgs(black_it_ratio, mask)
            masks.extend(transformed_masks)
            masks.append(mask)

        # get arrays
        img_list = [img[..., None] for img in imgs]
        img_array = np.stack(img_list, axis=0)
        mask_array = np.zeros((len(masks), size, size, len(label_dict)),
                              dtype=np.uint8)
        for ct, mask in enumerate(masks):
            mask_array[ct, ...] = np.dstack([
                resize(crop_img(mask[..., ii][..., None]), size)
                for ii in range(mask.shape[-1])
            ])

        return img_array, mask_array, label_dict, np.array(imgs_index)
