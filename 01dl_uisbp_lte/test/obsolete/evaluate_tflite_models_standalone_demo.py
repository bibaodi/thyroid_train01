'''
evaluate models by TestSet

Usage:
    evaluate_tflite_models_standalone.py 

Argument:

OutPut:
    evaluted images, csv file which include dice and other evaluation metrics

History：
    20200202-write for xiaobaishiji
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
import csv
import datetime
import seaborn as sns
import logging
import argparse
import hashlib
import tqdm
import PIL.Image
import PIL.ImageDraw

_logpath_ = r'evaluate_models.log'
_logger_ = None

DICE_CONST = 1.0
MINIMUM = 1e-6
SEG_THRESHOLD = 0.25
KEY_LABEL = None  #'CA'  # 'Plaque'  # 'BP'
KEY_LABELS = ['Plaque', 'CA']
PREFIX_PREDICT_LABEL = 'pred'

OUTPUT_FULL_EXPECTION = False

LabelList_plaque = ['CA', 'JV', 'Plaque', '_background_']
LabelList_ca = LabelList_plaque
LabelList = {
    "Plaque": LabelList_plaque,
    "CA": LabelList_ca,
}

Thresholds_plaque = [0.1, 0.26, 1e-3, 0.26]
Thresholds_ca = Thresholds_plaque
Thresholds = {
    "Plaque": Thresholds_plaque,
    "CA": Thresholds_ca,
}

def prepareLogging():
    logger = logging.getLogger("evaluate_linket_models")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    file_handler = logging.FileHandler(_logpath_)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger

# use getLogger instead logger. In the future logger may be move to a module

def getLogger():
    global _logger_, _logpath_
    if not _logger_:
        _logger_ = prepareLogging()
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
    dice = dice if dice > 0.0001 else 0
    return dice, top, bottom, ypred_sum, ytrue_sum


def calcDice(y,
             model_label_names,
             mask,
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
        ytrue = mask[..., ch]
    # calc dice
    key_dice = get_one_image_dice_coef(ypred, ytrue)
    return key_dice


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

def shapes_to_label_withkey(img_shape,
                            shapes,
                            label_name_to_value,
                            keyLabel="",
                            para_type='class'):
    assert para_type in ['class', 'instance']

    cls_mask = np.zeros(img_shape[:2], dtype=np.int32)

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
    if zoom:
        img_array_new = cv2.resize(img_croped,
                                   dsize=(zoomed_size, zoomed_size),
                                   interpolation=cv2.INTER_AREA)
    else:
        img_array_new = img_croped
    ratio = img_array_new.shape[0] / img_croped.shape[0]
    return img_array_new, xshift, ratio


def read_image_and_correct_it(imagefile=None, zoom=False, zoomed_size=448, clahe=True):
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
    new_img, xshift, ratio = correct_image_size(img, zoom=zoom, zoomed_size=zoomed_size)

    return (new_img, xshift, ratio)


def get_raw_image_filenames(img_root_dir="", extension='png', abandon_list=[]):
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


def get_imgs_masks_labels(img_root_dir="", extension='png', size=448, keyLabel='', abandon_list=[], black_it_ratio=[]):

    assert len(keyLabel) >= 0  # keyLabel must be something
    # get filenames
    filenames, imgs_index = get_raw_image_filenames(img_root_dir=img_root_dir, extension=extension, abandon_list=abandon_list)

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
            if "UncertainZone" in label:  
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
    for fname in tqdm.tqdm(filenames, ncols=80, desc="Processing Files:"):
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
        count_i += len(black_it_ratio) + 1
        imgs.append(img)

        if keyLabel:  # keep keylabel mask when it overlapps with others
            lbl = shapes_to_label_withkey(img.shape, data['shapes'],
                                            label_dict, keyLabel)
        else:  # not consider overlap between labels
            lbl = shapes_to_label(img.shape, data['shapes'],
                                        label_dict)

        # get mask
        ct = 0
        mask = np.zeros(img.shape + (len(label_dict), ),
                        dtype=np.uint8)  
        for (key, val) in sorted(label_dict.items()):  
            if key != 'UncertainZone':
                mask[lbl == val, ct] = 1
                ct += 1
        masks.append(mask)

    # get arrays
    img_list = [img[..., None] for img in imgs]
    img_array = np.stack(img_list, axis=0)
    mask_array = np.zeros((len(masks), size, size, len(label_dict)),
                            dtype=np.uint8)
    for ct, mask in enumerate(masks):
        mask_array[ct, ...] = np.dstack([
            mask[..., ii][..., None] for ii in range(mask.shape[-1])
        ])

    return img_array, mask_array, label_dict, np.array(imgs_index)


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
    
    full_image_name_index = imgs_index[index][0]
    imagedir = os.path.join(outputdir, full_image_name_index[:-5])
    framename = full_image_name_index[-4:]
    imagename = full_image_name_index[-4:] + '.jpg'

    imagepath = os.path.join(imagedir, imagename)
    os.makedirs(imagedir, exist_ok=True)
    img = cv2.resize(X[index], (img_outsize, img_outsize), interpolation=cv2.INTER_AREA)
    inputimg = img
    frmsize = inputimg.shape[0]
    ratio = frmsize / 448.0 
    PADheight = int(32*ratio)
    imagepad = np.zeros([PADheight, frmsize], dtype=np.uint8)
    imagepad[0, 0:-60] = 128  # split line

    inputimg = np.concatenate((inputimg, imagepad), axis=0)

    return inputimg, framename, imagepath


def drawFooterMessage(image, message, ratio=1.0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, message, (math.ceil(20*ratio), image.shape[0] - 10), font, 0.5*ratio,
                (0, 240, 240)) 


def generate_image_statistics_csv(filename, images_datas):

    detailtable = []
    csv_head = ['video', 'image', 'dice']
    for img_datas in images_datas: 
        videoname = img_datas['videoname']
        row = [[videoname, *x] for x in img_datas['framedetails']
               ] 
        detailtable += row
    detailtable.insert(0, csv_head)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(detailtable)
    return None


def outputDiceResult(out_dir, dset, folder_details):
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # output image dice
    image_dice_csv_file = os.path.join(
        out_dir, 'image_dice-' + dset + '-' + snow + '.csv')
    generate_image_statistics_csv(image_dice_csv_file, folder_details)


def generate_image_result(index, outputdir, imgs_idx, X, y_groundtrouth, Ypred,
                 model_label_names, masks_label_names, labelcolors, modelPara, dice):

    img_outsize = modelPara['outsize']
    model_size = modelPara['size']
    inputimg, framename, imagepath = prepareBaseImage(index, outputdir,
                                                      imgs_idx, X, img_outsize=img_outsize)
    imgsize = inputimg.shape
    ratio = imgsize[0] / 448.0 
    # original image
    base = cv2.cvtColor(inputimg, cv2.COLOR_GRAY2BGR)
    model_name = modelPara['dset'] if not os.path.isfile(modelPara['dset']) else os.path.basename(modelPara['dset']).split('.tflite')[0]
    drawFooterMessage(base,
                      f"Model: {model_name, model_size}", ratio)

    results = [base]

    # draw expection (ground trouth)
    if OUTPUT_FULL_EXPECTION:
        base = inputimg.copy()
        yimage_ = y_groundtrouth[index]
        yimage = cv2.resize(yimage_, (img_outsize, img_outsize), interpolation=cv2.INTER_AREA)
        outimge = drawContourRes(yimage,
                                 base,
                                 imgsize,
                                 masks_label_names,
                                 labelcolors,
                                 SEG_THRESHOLD, ratio, prediction=False)
        drawFooterMessage(outimge,
                          datetime.datetime.now().strftime('%Y/%m/%d %H:%M'), ratio)
        results.append(outimge)

    # draw predication
    base = inputimg.copy()
    yimage_ = Ypred[index]
    yimage = cv2.resize(yimage_, (img_outsize, img_outsize), interpolation=cv2.INTER_AREA)
    outimgeexp = drawContourRes(yimage,
                                base,
                                imgsize,
                                model_label_names,
                                labelcolors,
                                SEG_THRESHOLD,ratio, prediction=True)

    # draw expection in prediction image only for keylabel target
    base = inputimg.copy()
    yimage_ = y_groundtrouth[index]
    yimage = cv2.resize(yimage_, (img_outsize, img_outsize), interpolation=cv2.INTER_AREA)
    outimge = drawExpMask(yimage,
                          base,
                          imgsize,
                          masks_label_names)

    # blend predication and exptection
    alpha = 0.25
    cv2.addWeighted(outimge, alpha, outimgeexp, 1 - alpha, 0, outimgeexp)

    drawFooterMessage(outimgeexp, "Frame=%s Dice=%.2f" % (framename, dice), ratio)
    results.append(outimgeexp.copy())

    # create result
    visres = np.hstack(results)
    cv2.imwrite(imagepath, visres)
    return


def createresult(index, outputdir, imgs_idx, X, y_groundtrouth, Ypred,
                 model_label_names, masks_label_names, labelcolors, modelPara):

    img_outsize = modelPara['outsize']
    getLogger().debug(f'@@@debug X={X.shape}Y={y_groundtrouth.shape}')

    dice, top, bottom, ypred_sum, _ = calcDice(Ypred[index],
                                               model_label_names,
                                               y_groundtrouth[index],
                                               masks_label_names,
                                               thresh=SEG_THRESHOLD)

    full_image_name_index = imgs_idx[index][0]
    framename = full_image_name_index[-4:]
    one_image_result = ['frm-' + framename, dice]
    #one_image_result.extend(metrics4[1])
    if img_outsize >= 96:
        generate_image_result(index, outputdir, imgs_idx, X, y_groundtrouth, Ypred,
                            model_label_names, masks_label_names, labelcolors, modelPara, dice)

    return one_image_result


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
        output_data = interpreter.get_tensor(output_details[0]['index'])
        out_img = output_data[0]  
        out_imgs.append(out_img)
    return out_imgs


def HandleOneFrameFolder(image_dir, out_dir, modelPara):
    # create data
    print("1. Create np data from input.")
    model_size = modelPara['size']
    imgs, masks, label_dict, imgs_idx = get_imgs_masks_labels(image_dir,
        size=model_size, keyLabel=KEY_LABEL)
    masks_label_names = list(sorted(label_dict.keys()))
    print(imgs.shape, masks.shape, imgs_idx.shape)
    print("label for model:", modelPara['label_names'])
    print('label in estimation data:', masks_label_names)

    total_labels = sorted(
        np.unique(modelPara['label_names'] + masks_label_names))
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
    print("2. linknet segmentation.")
    post_process = True
    if post_process:
        print("process duration inlcudes post process step.")
    start = time.perf_counter()
    tf_interpreter = use_tflite_load(modelPara['dset'])
    ypred = use_tflite_predict(imgs, tf_interpreter)
    if 'labels_thresh' in modelPara:
        labels_thresh = modelPara['labels_thresh']
    else:
        labels_thresh = {i:0.26 for i in modelPara['label_names']}
    print(f"ypred: type={ypred[0].dtype}")
    predFrameDuration = int(
        (time.perf_counter() - start) * 1000 / imgs.shape[0])
    print("duration(ms)/frame", predFrameDuration)

    print("3. Ouput result images.")
    finaloutputdir = out_dir
    folder_result = []
    print("output dir", finaloutputdir)
    for i in range(masks.shape[0]):
        res = createresult(i, finaloutputdir, imgs_idx, imgs,
                           masks, ypred, modelPara['label_names'],
                           masks_label_names, labelcolors, modelPara)
        getLogger().debug("@@@debug: createresult", res)
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
    print(f"all video directorys:\n{imgdir_names}")

    print('modelPara:', modelPara)

    total = len(imgdir_names)
    folder_details = []
    for ct, image_dir in enumerate(imgdir_names):
        print("*"*32)
        print('Handle with {0} [{1}/{2}]'.format(image_dir, ct + 1, total))
        files_in_dir = os.listdir(image_dir)
        if len(files_in_dir) < 2:
            print(f"Empty dirctory : {image_dir}")
            continue
        onefolder_res, frameDuration = HandleOneFrameFolder(
            image_dir, out_dir, modelPara)

        folder_details.append({
            'videoname': image_dir.split(os.sep)[-1],
            'frameduration': frameDuration,
            'framedetails': onefolder_res
        })
        print("#"*32, '\n')
    outputDiceResult(out_dir, model_part + '_' + str(modelPara['size']),
                     folder_details)
    with open(os.path.join(out_dir, "paras.log"), 'w') as f:
        f.write(f"keylabel={KEY_LABEL}\nmodel_md5:{model_md5}\n{modelPara}")
    print('output dir:', out_dir, "\n\nFinish...")


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
    getLogger().info(f"key label is: {args.keylabel}")
    
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
    print("Start at {}".format(snow))

    outdir = img_root + '.out_' + KEY_LABEL
    if models:
        for model in models:
            para = {
                'labels_thresh': labels_thresh,
                'label_names': label_names,
                'dset': model[0],
                'size': model[1],
                'channel_first': False,
                'outsize': args.out_size
            }
            mainProcess(img_root, outdir, para)

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
    return parser


if __name__ == '__main__':
    print("*"*80+"\n", "params:", sys.argv)
    parser = init_argparser()
    args = parser.parse_args()
    main(args, args.confirmed)
    print('finish...')
