'''
calculate IMT base on the segmentation of IM

Usage:
    similary to the evaluate scripts

Argument:
    model-file target-image

OutPut:
    evaluted images, csv file which include dice and other evaluation metrics

History：
    20200603-first-edition

Algorithm:
    1. 判定IM的有无
    2. 生产轮廓
    3. 根据生产的轮廓数量判定, 如果大于1个选择面积最大的那个
    4. 用x坐标计算长度
    5. 长度(25%, ~) (38mm计算)那么划分5段; 若长度(10%, 25%], 那么划分3段; 若(0, 10%)那么不分段, 只获取一个数值
    6. 逐段获取矩形, 以得到IMT的Thickness
    7. 将几个数值进行排序, 并做基本判定, 比如不能相差大于2倍, 输出三个数值(最大, 最小, 平均)
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
import tensorflow as tf
import glob
import time
import math
import json
import csv, statistics
import datetime
import seaborn as sns
import logging
import argparse
import hashlib
import tqdm
import PIL.Image
import PIL.ImageDraw

_logpath_ = r'cal_imt.log'
_logger_ = None

DICE_CONST = 2.0
DICE_TN = 1.0
MINIMUM = 1e-6
SEG_THRESHOLD = 0.25
KEY_LABEL = 'IM'  #'CA'  # 'Plaque'  # 'BP'
KEY_LABELS = ['Plaque', 'BP', 'CA', 'IM']
PREFIX_PREDICT_LABEL = 'pred'

OUTPUT_FULL_EXPECTION = True


LabelList_bp = ['ASM', 'BP', 'MSM', 'SCM', 'Vessel', '_background_']
LabelList_plaque = ['CA', 'JV', 'Plaque', '_background_']
LabelList_imt = ['CA', 'IM', 'JV', 'Plaque', '_background_']
LabelList_ca = LabelList_plaque
LabelList_bp_ca = ['ASM', 'BP', 'CA', 'JV', 'MSM', 'SCM', '_background_']
LabelList_interscalene = ['ASM', 'Interscalene', 'MSM', 'SCM', 'Vessel', '_background_']
LabelList = {
    "Plaque": LabelList_plaque,
    "IM": LabelList_imt,
    "CA": LabelList_ca,
    "BP": LabelList_bp,
    "Interscalene": LabelList_interscalene,
    "BP_CA": LabelList_bp_ca,
}

Thresholds_plaque = [0.1, 0.26, 1e-3, 0.26]
Thresholds_imt = [0.1, 0.1, 0.26, 1e-3, 0.26]
Thresholds_ca = Thresholds_plaque
Thresholds_bp = [0.26, 0.26, 0.26, 0.26, 0.26, 0.26]
Thresholds = {
    "Plaque": Thresholds_plaque,
    "IM": Thresholds_imt,
    "CA": Thresholds_ca,
    "BP": Thresholds_bp,
}

def prepareLogging(level=logging.INFO):
    logger = logging.getLogger("evaluate_linket_models")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    file_handler = logging.FileHandler(_logpath_)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(level)
    return logger

# use getLogger instead logger. In the future logger may be move to a module

def getLogger(level='info'):
    global _logger_, _logpath_
    if not _logger_:
        if level == 'info':
            log_level = logging.INFO
        elif level == 'debug':
            log_level = logging.DEBUG
        elif level == 'warn':
            log_level = logging.WARN
        else:
            log_level = logging.ERROR
        _logger_ = prepareLogging(log_level)
    return _logger_


def label_colormap(N=128):
    colors = sns.color_palette("bright", n_colors=N)
    col = np.array(colors)
    return col


def get_one_image_dice_coef(ypred, ytrue, smooth=MINIMUM):
    top = 2 * np.sum(ypred * ytrue) + smooth * DICE_CONST
    ypred_sum, ytrue_sum = np.sum(ypred), np.sum(ytrue)
    bottom = ypred_sum + ytrue_sum + smooth
    dice = top / bottom
    dice = dice if dice > 0.0001 else 0.0
    return dice, top, bottom, ypred_sum, ytrue_sum


def get_dice_coef(ypred, ytrue, smooth=MINIMUM):
    top = 2 * np.sum(ypred * ytrue, axis=(1, 2, 3)) + smooth * DICE_CONST
    bottom = np.sum(ypred, axis=(1, 2, 3)) + np.sum(ytrue,
                                                    axis=(1, 2, 3)) + smooth
    return top / bottom


def calcDice(y,
             model_label_names,
             ground_truth,
             masks_label_names,
             thresh=0.26):

    # get pred result region
    ch = model_label_names.index(KEY_LABEL)
    ypred = y[..., ch] > thresh
    ch = masks_label_names.index(
        KEY_LABEL) if KEY_LABEL in masks_label_names else -1
    if ch < 0:
        ytrue = np.zeros(ypred.shape)
    else:
        ytrue = ground_truth[..., ch]
    # calc dice
    key_dice = get_one_image_dice_coef(ypred, ytrue)
    return key_dice


def calculate_4_matrics_by_dice(dice, labels:list, inference, key_label):
    """
    @parameter: labels 当前图片标注的所有标签, 如果同一种类型标签有多个, 那么代表有多个目标, 此时需要判断TP+FP等组合情况. 
    @parameter: dice 当前图片的dice数值
    @parameter: inference touple of (ypred_sum, contours count) 推理结果中包含的像素数量和轮廓数的元组
    @parameter: key_label 当前计算的的标签类型
    针对单张图片将dice数值转化为TP TN FP FN
    当出现推理结果中有目标并且标注中也有目标但是两个目标不相交的时候FP&FN=1
    """
    if not isinstance(dice, float):
        print(f"dice should be float type: {dice}")
        return None
    if not isinstance(labels, list):
        print(f"labels should be list type:{labels}")
        return None

    getLogger().debug(
        f"calculate_4_matrics_bydice: dice={dice}, labels={labels}, keylabel={key_label}, inference={inference}"
    )
    result = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    if dice > 1.1 and key_label not in labels:
        result["TN"] = 1
    elif dice < 1.0 and dice >= 0.01:
        result["TP"] = 1
        #TODO 当数量大于1 的时候, 因为没办法逐个比对, 难以区分具体的情况
        if inference[1] > 1 and 1 == labels.count(key_label):
            result["FP"] = 1
        elif 1 == inference[1] and labels.count(key_label) > 1:
            result["FN"] = 1
    elif dice < 0.01:
        if key_label not in labels:
            result["FP"] = 1
        elif key_label in labels:
            result["FN"] = 1
            if inference[1]:
                result["FP"] = 1

    return (result, [result["TP"], result["TN"], result["FP"], result["FN"]])


def calculate_4_matrics_by_pixels(inference:np.ndarray, ground_truth:np.ndarray, label_dict:dict, keylabel:str):
    """
    param: inference (r,w, count(target)) dtype=uint8
    param: groud_truth (r, w, count(target)) dtype=uint8
        >根据每个像素的结果判断是属于TP TN FP FN中的哪一种,然后统计数量
        >>但是没有办法判断当前这一张图片的单一目标属于上述四种的哪一种, 但是不影响precison/recall的计算
    """
    #todo 一次计算所有想要的目标的结果
    result = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    ch = label_dict[keylabel] if keylabel in label_dict else 0
    result["TP"] = np.sum(np.logical_and(inference[..., ch], ground_truth[..., ch]))
    result["TN"] = np.sum(np.logical_and(np.logical_not(inference[..., ch]), np.logical_not(ground_truth[..., ch])))
    result["FP"] = np.sum(np.logical_and(inference[..., ch], np.logical_not(ground_truth[..., ch])))
    result["FN"] = np.sum(np.logical_and(np.logical_not(inference[..., ch]), ground_truth[..., ch]))

    return (result, [result["TP"], result["TN"], result["FP"], result["FN"]])


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


def shapes_to_label_withkey(img_shape,
                            shapes,
                            label_name_to_value,
                            keyLabel="",
                            bg_value:int=3):
    """
        bg_value: background default value
    """
    cls_mask = np.zeros(img_shape[:2], dtype=np.int32)
    if 0 != bg_value:
        cls_mask += bg_value
    # draw other labels first
    for shape in shapes:
        label = shape['label']
        if label != keyLabel:
            polygons = shape['points']
            cls_name = label
            cls_id = label_name_to_value[cls_name]
            mask = shape_to_mask(img_shape[:2], polygons)
            cls_mask[mask] = cls_id
    # draw key label in order to make sure key label mask overlap others
    for shape in shapes:
        label = shape['label']
        if label == keyLabel:
            polygons = shape['points']
            cls_name = label
            cls_id = label_name_to_value[cls_name]
            mask = shape_to_mask(img_shape[:2], polygons)
            cls_mask[mask] = cls_id
    return cls_mask


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


def correct_image_size(img_array, zoom=False, zoomed_size=0):
    """
        crop+resize. 返回处理好的图片并返回X轴偏移以及缩放比例. 
    """
    if not img_array.any() or not isinstance(img_array, np.ndarray):
        raise ValueError("param is not np.ndarray")
    if len(img_array.shape) != 2 and len(img_array.shape) != 3:
        raise ValueError(
            "image must in IMREAD_GRAYSCALE or IMREAD_COLOR(height, row, gray)!"
        )
    if zoomed_size < 1:
        raise ValueError(f"zoomed size must visualized. current is {zoomed_size}")
    true_width = get_image_actual_width_index_range_mem(img_array.copy())
    img_croped, xshift, _ = crop_img_and_remove_black_edge(
        img_array, true_width[0], true_width[1])
    getLogger().debug(f"img cropped size={img_croped.shape}")
    cropped_size = math.ceil(img_croped.shape[0] / 10) * 10
    img_croped = cv2.resize(img_croped,
                                dsize=(cropped_size, cropped_size),
                                interpolation=cv2.INTER_AREA)
    if not zoom:
        img_array_new = img_croped
    else:
        img_array_new = cv2.resize(img_croped,
                                    dsize=(zoomed_size, zoomed_size),
                                    interpolation=cv2.INTER_AREA)
    ratio = img_array_new.shape[0] / img_croped.shape[0]
    return img_array_new, xshift, ratio, img_croped


def read_image_and_correct_it(imagefile=None, zoom=False, zoomed_size=448, clahe=True):
    """
    读取图片, 如果图片不是448*448的那么进行crop, resize, final_size=zoomed_size
    注意: 此函数调用后需要配套修改json的坐标
    """
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
    new_img, xshift, ratio, cropped_img = correct_image_size(img, zoom=zoom, zoomed_size=zoomed_size)

    return (new_img, xshift, ratio, cropped_img)


def get_raw_image_filenames(img_root_dir="", extension='png', abandon_list=[]):
    """
    Returns full path to all image files of type `img_type`

    Args:
        extension (str): File-type extension [default 'jpg']

    Returns:
        (list): List of image filenames
    """
    pattern = os.path.join(img_root_dir, f"**{os.sep}*{extension}")
    all_img_filenames = sorted(glob.glob(pattern, recursive=True))
    
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


def transfer_mask_from_1d_to_nd(mask_1d, label_dict):
    """
    make mask from 1d(96,96) to nd (96, 96, n)
    """
    getLogger().debug(f"transfer_mask_from_1d_to_nd: mask_1d.shape={mask_1d.shape}; labeldict={label_dict}")
    mask_nd = np.zeros(mask_1d.shape[:2] + (len(label_dict), ),
                    dtype=np.uint8)  
    for (key, val) in label_dict.items():  
        if key != 'UncertainZone':
            mask_nd[mask_1d == val, val] = 1
    return mask_nd


def get_imgs_masks_labels(img_root_dir="", label_names: list=[], extension='png', size=448, keyLabel='', abandon_list=[], black_it_ratio=[]):

    assert len(keyLabel) >= 0  # keyLabel must be something
    assert len(label_names) > 0 #labels must not empty 
    # get filenames
    filenames, imgs_index = get_raw_image_filenames(img_root_dir=img_root_dir, extension=extension, abandon_list=abandon_list)

    # get label names and get labels for one image
    label_dict = {'_background_': 0}
    img_info_touple_s = imgs_index
    for i, label in enumerate(label_names):
        label_dict[label] = i
    for _index, fname in enumerate(filenames):
        json_file = fname.split('.png')[0] + '.json'
        _, data = get_json_from_file(json_file)
        labels = []
        for shape in data['shapes']:
            label = shape['label']
            labels.append(label)
        img_info_touple_s[_index] = (imgs_index[_index], labels)

    print(f"get_imgs_masks_labels: label dict is {label_dict}")

    # make masks
    imgs, masks_nd, masks_1d, cropped_imgs = [], [], [], []
    count_i = 0
    for fname in tqdm.tqdm(filenames, ncols=80, desc="Processing Files:"):
        json_file = fname.split('.png')[0] + '.json'

        img_datas = read_image_and_correct_it(fname, zoom=True, zoomed_size=size)
        img = img_datas[0]
        xshift = img_datas[1]
        ratio = img_datas[2]
        cropped_img = img_datas[3]
        # open jsonfile
        _, data = get_json_from_file(json_file,
                                        need_correct=True,
                                        xshift=xshift,
                                        ratio=ratio,
                                        finalsize=img.shape[0])
        count_i += len(black_it_ratio) + 1
        imgs.append(img)
        cropped_imgs.append(cropped_img)
    
        if keyLabel:  # keep keylabel mask when it overlapps with others
            mask_1d = shapes_to_label_withkey(img.shape, data['shapes'],
                                            label_dict, keyLabel, bg_value=label_dict['_background_'])
        masks_1d.append(mask_1d)
        mask_nd = transfer_mask_from_1d_to_nd(mask_1d, label_dict)
        masks_nd.append(mask_nd)
    # get arrays
    img_list = [img[..., None] for img in imgs]
    img_array = np.stack(img_list, axis=0)
    masks_nd_array = np.stack(masks_nd, axis=0)
    return img_array, masks_nd_array, np.array(img_info_touple_s, dtype=object), np.stack(masks_1d, axis=0), np.stack(cropped_imgs, axis=0)


def drawLabelLegend(outimg, imgwidth, imgheight, label_names, labelcolors, ratio, prediction=True):
    labelwidth = 45*ratio
    linelength = 20*ratio
    labelheight = 14*ratio
    font = cv2.FONT_HERSHEY_SIMPLEX

    leglabels = label_names.copy()
    if '_background_' in leglabels:
        leglabels.remove('_background_')

    for ct, lbl in enumerate(leglabels):
        if prediction:
            lbl_c = PREFIX_PREDICT_LABEL + lbl
        else:
            lbl_c = lbl
        color = labelcolors[lbl_c].astype(np.float, copy=False) * 255

        left = int(imgwidth - labelwidth - linelength)
        top = int(imgheight - labelheight * (ct + 1.5*ratio))
        right = int(imgwidth - labelwidth - 5*ratio)
        bottom = int(imgheight - labelheight * (ct + 1*ratio))

        cv2.rectangle(outimg, (left, top), (right, bottom), color, -1)
        cv2.putText(outimg, lbl, (int(imgwidth - labelwidth),
                                  int(imgheight - labelheight * (ct + 1*ratio))),
                    font, 0.4*ratio, color, 1, cv2.LINE_AA)


def drawContourRes(mask, img, frmsize, label_names, labelcolors, thresh=0.26, ratio=1.0, prediction=True):
    getLogger().debug(f"@@@debug: labelnames={label_names}, shape={mask.shape}")
    ypred = mask
    nch = ypred.shape[-1] if len(ypred.shape) > 2 else 1
    outimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if '_background_' in label_names:
        hidechannel = 1
    else:
        hidechannel = 0

    for ch in range(nch - hidechannel):
        _, contours, hierarchy = cv2.findContours(
            (ypred[..., ch] > thresh).astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
        if prediction:
            label_name = PREFIX_PREDICT_LABEL + label_names[ch]
        else:
            label_name = label_names[ch]
        color = labelcolors[label_name].astype(np.float, copy=False) * 255
        getLogger().debug(f"@@@debug: label={label_names[ch]}, color={color}, contours={len(contours)}")
        # thickness = 4 if label_names[ch] == KEY_LABEL else 2
        thickness = 2*ratio if label_names[ch] == KEY_LABEL else 1*ratio
        outimg = cv2.drawContours(outimg, contours, -1, color, math.ceil(thickness))

    # draw legend
    drawLabelLegend(outimg, frmsize[1], frmsize[0], label_names, labelcolors, ratio, prediction)
    return outimg


def drawExpMask(mask, img, frmsize, label_names):
    nch = mask.shape[-1] if len(mask.shape) > 2 else 1
    outimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for ch in range(nch - 1):
        if label_names[ch] == KEY_LABEL:
            _, contours, hierarchy = cv2.findContours(
                (mask[..., ch]).astype(np.uint8), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            color = (255, 0, 0)  #cmap[colorlist[ch]].astype(np.float, copy=False)*255
            outimg = cv2.drawContours(outimg, contours, -1, color, -1)
    return outimg


def prepareBaseImage(index, outputdir, imgs_index, X, img_outsize):
    
    full_image_name_index = imgs_index[index]
    imagedir = os.path.join(outputdir, full_image_name_index[:-5])
    framename = full_image_name_index[-4:]
    imagename = full_image_name_index[-4:] + '.jpg'

    imagepath = os.path.join(imagedir, imagename)
    os.makedirs(imagedir, exist_ok=True)
    #print('before squeeze', X[index].shape)
    img = cv2.resize(X[index], (img_outsize, img_outsize), interpolation=cv2.INTER_AREA)
    #--inputimg = X[index].squeeze(2)
    inputimg = img
    frmsize = inputimg.shape[0]
    ratio = frmsize / 448.0 
    #print('after squeeze', inputimg.shape)
    PADheight = int(32*ratio)
    imagepad = np.zeros([PADheight, frmsize], dtype=np.uint8)
    imagepad[0, 0:-60] = 128  # split line

    inputimg = np.concatenate((inputimg, imagepad), axis=0)

    return inputimg, framename, imagepath


def drawFooterMessage(image, message, ratio=1.0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    #print(f"debug: math.ceil(20*ratio)={math.ceil(20*ratio)}")
    cv2.putText(image, message, (math.ceil(20*ratio), image.shape[0] - 10), font, 0.5*ratio,
                (0, 240, 240))  ##cv2.FILLED)


def generate_image_statistics_csv(filename, images_datas):
    """
        以每张图片的IMT结果为一行输出到csv文件. 
    """
    detailtable = []
    csv_head = ['video', 'image', 'max', 'min', 'mean', 'all']
    for img_datas in images_datas:  # iterate erery video-folder
        videoname = img_datas['videoname']
        rows = []
        for img_imt in img_datas['framedetails']:
            img_name = img_imt[0]
            _thickness = img_imt[1:] if len(img_imt) > 1 else [0]
            _max = max(_thickness)
            _min = min(_thickness)
            _mean = sum(_thickness)/len(_thickness)
            _all = repr(_thickness)
            row = [videoname, img_name, _max, _min, _mean, _all]
            rows.append(row)
        detailtable += rows
    detailtable.insert(0, csv_head)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(detailtable)
    return None


def outputIMTResult(out_dir, dset, folder_details):
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # output image dice
    image_dice_csv_file = os.path.join(
        out_dir, 'IMT-' + dset + '-' + snow + '.csv')
    generate_image_statistics_csv(image_dice_csv_file, folder_details)
    return 


def createresult(index, outputdir, img_info_touple_s, images_struct, y_groundtrouth, ypreds,
                 model_label_names, masks_label_names, labelcolors, modelPara):
    """
    根据预测得到的图片array(ypreds) 生成结果图片与csv文件. 如果图片小于96, 不生成图片. 
    param: img_info_touple_s, toupe of (index, labels) for all images
    output:
        ['frm-0001', dice, TP, TN, FP, FN]
    """
    X = images_struct['input_imgs']
    cropped_imgs = images_struct['cropped_imgs']
    img_outsize = modelPara['outsize']
    getLogger().debug(f'@@@debug X={X.shape}Y={y_groundtrouth.shape}')
    if modelPara['use_threshold'] == 0:
        label_dict = {k: model_label_names.index(k) for k in model_label_names}
        getLogger().debug(f'use softmax!\nmodel_label_names:{label_dict}')
        getLogger().debug(f'createresult0: ypreds[index].shape{ypreds[index].shape}')
        ypreds[index] = np.argmax(ypreds[index], axis=-1)
        getLogger().debug(f'createresult1: ypreds[index].shape{ypreds[index].shape}')
        ypreds[index] = transfer_mask_from_1d_to_nd(ypreds[index], label_dict)

    key_label_ch = model_label_names.index(KEY_LABEL)
    target_pred = ypreds[index][..., key_label_ch]
    getLogger().info(f"shape of target pred is:{target_pred.shape}")
    full_image_name_index = img_info_touple_s[index][0]
    framename = full_image_name_index[-4:]
    imt_result = [f'frm-{framename}']

    target_pixel_sum = np.sum(target_pred)
    getLogger().info(f"IM's target pixel sum is:{target_pixel_sum}=[{(100*target_pixel_sum/target_pred.size):2.6f}%]")
    if target_pixel_sum < 100:
        getLogger().info(f"target is too small")
        return imt_result
    
    target_indices = np.argwhere(target_pred>0.5)
    getLogger().info(f"target indices len{len(target_indices)}, first={target_indices[0]} last={target_indices[-1]}")

    _, contours, hierarchy = cv2.findContours(
            (target_pred > SEG_THRESHOLD).astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    
    getLogger().info(f"count of IM contours:{len(contours)}")
    if 1 > len(contours):  #todo: sourt and get the largest one
        return imt_result

    target_indices_r, target_indices_c = target_indices[:, 1], target_indices[:, 0]
    getLogger().info(f"r, c: {target_indices_r.shape, target_indices_c.shape}")
    im_height = np.max(target_indices_c) - np.min(target_indices_c)
    im_width = np.max(target_indices_r) - np.min(target_indices_r)
    getLogger().info(f"IM range:{np.max(target_indices_c) , np.min(target_indices_c)};{np.max(target_indices_r), np.min(target_indices_r)}")
    getLogger().info(f"height={im_height}; im_width={im_width}")
    
    if im_width > (target_pred.shape[0] * 0.25):
        devide_count = 5.0
    elif im_width > (target_pred.shape[0] * 0.1):
        devide_count = 3.0
    elif im_width > (target_pred.shape[0] * 0.05):
        devide_count = 2.0
    else:
        return imt_result
    im_length_delta = im_width / devide_count
    im_split_thickness = []
    im_split_left = np.min(target_indices_r)
    for i in range(int(devide_count)):
        im_split_right = im_split_left + im_length_delta
        im_split_right = im_split_right if im_split_right <=np.max(target_indices_r) else np.max(target_indices_r)
        getLogger().info(f"left={im_split_left}, right={im_split_right}")
        im_split_r = target_indices_r.copy()
        
        im_split_r[im_split_r < im_split_left] = 0
        im_split_r[im_split_r >= im_split_right] = 0
        im_split_c = target_indices_c[np.nonzero(im_split_r)]
        getLogger().info(f"c sets:{im_split_c.size}")
        #im_split_c[im_split_r !=0]
        #im_split = np.stack([im_split_c, im_split_r], axis=0)
        _thickness = np.max(im_split_c) - np.min(im_split_c, initial=np.min(target_indices_c))
        im_split_thickness.append(_thickness)
        im_split_left = im_split_right + 1
    getLogger().info(f"all thickness: {im_split_thickness}")
    if max(im_split_thickness) / min(im_split_thickness) > 2.0:
        getLogger().warn(f"!+++!thickness may wrong; The difference between the maximum value and the minimum value is greater than 2.0!")
        return imt_result
    getLogger().info(f"thickness-max={max(im_split_thickness)}; min={min(im_split_thickness)}; average={sum(im_split_thickness)/len(im_split_thickness)}")
    imgs_indx = img_info_touple_s[:, 0]
    full_image_name_index = imgs_indx[index]
    imagedir = os.path.join(outputdir, full_image_name_index[:-5])
    framename = full_image_name_index[-4:]
    imagename = full_image_name_index[-4:] + '.jpg'
    imagepath = os.path.join(imagedir, imagename)
    os.makedirs(imagedir, exist_ok=True)

    os.path.join(outputdir, imagename)
    cv2.imwrite(imagepath, target_pred*250)
    if len(im_split_thickness) > 1:
        imt_result.extend(im_split_thickness)
    return imt_result


def use_tflite_load(modelfile):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=modelfile)
    interpreter.allocate_tensors()
    return interpreter


def use_tflite_predict(imgs, tf_interpreter):
    interpreter = tf_interpreter
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    out_imgs = []
    for img in imgs:
        x_shape = (1, ) + img.shape
        x = img.reshape(x_shape)
        if list(x.shape) == input_shape.tolist():
            input_data = x
        else:
            raise ValueError("image shape error")
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        #out_img = cv2.resize(output_data[0], (size, size), interpolation=cv2.INTER_AREA).reshape(size, size, output_data.shape[-1])
        out_img = output_data[0]  # 因为当最后一维通道数 > 4的时候, 不能够使用opencv进行resize了 ???. 这里输出size=model.size的模型尺寸. channel=6的时候cv2.resize是ok的.
        out_imgs.append(out_img)
    return out_imgs


def HandleOneFrameFolder(image_dir, out_dir, modelPara):
    # create data
    getLogger().info("1. Create np data from input.")
    model_size = modelPara['size']
    imgs, gt_nd, img_info_touple_s, masks_1d, cropped_imgs = get_imgs_masks_labels(image_dir, modelPara['label_names'], 
        size=model_size, keyLabel=KEY_LABEL)
    masks_label_names = modelPara['label_names']
    getLogger().info(f"imgs.shape={imgs.shape}, nd.shape={gt_nd.shape}, 1d.shape={masks_1d.shape}, img_info_touple_s={img_info_touple_s.shape}")
    getLogger().info(f"label for model: {modelPara['label_names']}")
    getLogger().info(f"label in estimation data: {masks_label_names}") # label_names form true labeled data maybe no 'JV'

    total_labels = np.unique(modelPara['label_names'] + masks_label_names)
    color_map_count = len(total_labels) * 2 + 2
    if color_map_count < 16:
        color_map_count = 16
    else:
        color_map_count = 128
    cmap = label_colormap(color_map_count)
    labelcolors = {}
    for ct, label in enumerate(total_labels):
        labelcolors[label] = cmap[ct]
        labelcolors[PREFIX_PREDICT_LABEL + label] = cmap[ct+len(total_labels)+2]

    # Linknet Segmentation
    getLogger().info("2. linknet segmentation.")
    post_process = True
    if post_process:
        getLogger().info("process duration inlcudes post process step.")
    start = time.perf_counter()
    tf_interpreter = use_tflite_load(modelPara['dset'])
    ypreds = use_tflite_predict(imgs, tf_interpreter)
    if 'labels_thresh' in modelPara:
        labels_thresh = modelPara['labels_thresh']
    else:
        labels_thresh = {i:0.26 for i in modelPara['label_names']}
    getLogger().info(f"ypreds: type={ypreds[0].dtype}")
    predFrameDuration = int(
        (time.perf_counter() - start) * 1000 / imgs.shape[0])
    getLogger().info(f"duration(ms)/frame:{predFrameDuration}")

    getLogger().info("3. Ouput result images.")
    finaloutputdir = out_dir
    folder_result = []
    getLogger().info(f"output dir:{finaloutputdir}")
    images_struct = {
        'input_imgs': imgs,
        'cropped_imgs': cropped_imgs
    }

    for i in range(gt_nd.shape[0]):
        res = createresult(i, finaloutputdir, img_info_touple_s, images_struct,
                           gt_nd, ypreds, modelPara['label_names'],
                           masks_label_names, labelcolors, modelPara)
        getLogger().debug(f"@@@debug: createresult: {res}")
        folder_result.append(res)
    return folder_result, predFrameDuration


def mainProcess(img_root_dir, out_dir, modelPara):
    if os.path.isfile(modelPara['dset']):
        model_file = modelPara['dset']
        model_part = os.path.basename(model_file).split('.tflite')[0]
        model_md5 = hashlib.md5()
        with open(model_file, 'rb') as f:
            buff = f.read()
            model_md5.update(buff)
            model_md5 = model_md5.hexdigest()
    else:
        raise ValueError(f"{modelPara['dset']} must be tflite file!!!")
    out_dir = out_dir + f".{model_part}.{datetime.datetime.now().strftime('%Y%m%dT%H%M')}"
    imgdir_names = []

    dirs = sorted(os.listdir(img_root_dir))
    for item in dirs:
        path = os.path.join(img_root_dir, item)
        if os.path.isdir(path):
            imgdir_names.append(path)
    getLogger().info(f"all video directorys:\n{imgdir_names}")

    getLogger().info(f"modelPara: {modelPara}")

    total = len(imgdir_names)
    folder_details = []
    for ct, image_dir in enumerate(imgdir_names):
        print("*"*32)
        print('Handle with {0} [{1}/{2}]'.format(image_dir, ct + 1, total))
        files_in_dir = os.listdir(image_dir)
        if len(files_in_dir) < 2:
            getLogger().info(f"Empty dirctory : {image_dir}")
            continue
        onefolder_res, frameDuration = HandleOneFrameFolder(
            image_dir, out_dir, modelPara)

        folder_details.append({
            'videoname': image_dir.split(os.sep)[-1],
            'frameduration': frameDuration,
            'framedetails': onefolder_res
        })
        print("#"*32, '\n')
    outputIMTResult(out_dir, model_part + '_' + str(modelPara['size']),
                     folder_details)
    with open(os.path.join(out_dir, "paras.log"), 'w') as f:
        f.write(f"keylabel={KEY_LABEL}\nmodel_md5:{model_md5}\n{json.dumps(modelPara, indent=4)}")
        
    getLogger().info(f'output dir:{out_dir} \n\nFinish...')


def main(args, confirmed):
    if not os.path.isdir(args.test_set_dir):
        raise ValueError(f"testset dir error: {args.test_set_dir}")
    img_root = args.test_set_dir
    while os.sep == img_root[-1]:
        img_root = img_root[0: -1]
    global _logger_, _logpath_
    _logpath_ = os.path.join(img_root, 'precrop.log')

    getLogger().info(f"test set root dir: {img_root}")
    if args.keylabel not in KEY_LABELS: 
        getLogger().error(f"key label error: {args.keylabel}")
        return -1
    global KEY_LABEL
    KEY_LABEL = args.keylabel
    getLogger().info(f"key label is: {args.keylabel}\n use_threshold is:{args.use_threshold}")
    
    label_names = LabelList[args.keylabel]
    threshs = Thresholds[args.keylabel]
    labels_thresh = {i: j for i, j in zip(label_names, threshs)}

    models = [(args.model_file, args.model_size)]

    if not confirmed:
        print(f"""要针对模型:{models}\nLabel: {KEY_LABEL}\n进行评估?\n确认请输入'Y'否则'N' """)
        while True:
            confirm = input()
            if confirm.lower() == "y":
                break
            elif confirm.lower() == 'n':
                return None
            else:
                print(f"当前输入[{confirm}]无效, 请输入Y/N")
                continue
    else:
        print("already confirmed...")
    
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    getLogger().info(f"Start at {snow}")

    outdir = img_root + '.out_' + KEY_LABEL
    if models:
        for model in models:
            para = {
                'labels_thresh': labels_thresh,
                'label_names': label_names,
                'dset': model[0],
                'size': model[1],
                'channel_first': False,
                'outsize': args.out_size,
                'use_threshold': args.use_threshold
            }
            mainProcess(img_root, outdir, para)
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    getLogger().info(f"End at {snow}")

def init_argparser():
    parser = argparse.ArgumentParser(
        description="evaluate tflite model.",
        usage=f"python prog.py [options] [parameters]",
        epilog="written by bibaodi")
    parser.add_argument('--testSetDir',
                        '-t',
                        required=True,
                        dest='test_set_dir',
                        metavar='test-set-video-directory',
                        help=f"the full path directory for test set video folder")
    parser.add_argument('--keylabel',
                        '-k',
                        dest='keylabel',
                        required=True,
                        metavar='the-key-label',
                        help=f"the key label which in the labels")
    parser.add_argument('--confirmed',
                        '-y',
                        dest='confirmed',
                        action='store_true',
                        help=f"the confirm parameter")
    parser.add_argument('--model',
                        '-m',
                        required=True,
                        dest='model_file',
                        metavar='test-tflite-model_file-directory',
                        help=f"the full path directory for tflite model file")
    parser.add_argument('--model-size',
                        '-s',
                        dest='model_size',
                        type=int, default=96,
                        metavar='test-model-input-size',
                        help=f"the model input size")
    parser.add_argument('--out-size',
                        '-o',
                        dest='out_size',
                        type=int, default=96,
                        metavar='output-image-size',
                        help=f"the output image size, generally is original image size")
    parser.add_argument('--use-threshold',
                        '-u',
                        dest='use_threshold',
                        type=float, default=0.0,
                        help=f"the threshold of the model([0,1.0] default=0 is use softmax)")
    parser.add_argument('--log',
                        dest='log',
                        type=str, default='info', choices=['debug', 'info', 'warn', 'error'],
                        help=f"the log level: debug info warn error")
    return parser


if __name__ == '__main__':
    print("*"*80+"\n", "params:", sys.argv)
    parser = init_argparser()
    args = parser.parse_args()
    getLogger(args.log)
    main(args, args.confirmed)
    print('End...')
