'''
evaluate models by TestSet

Usage:
    evaluate_linket_models.py 

Argument:

OutPut:
    evaluted images, csv file which include dice and other evaluation metrics

History：
    20190902 port from estimate_mul_batch.py, and add TN TP FN FP for csv
    20190926 support argument parser and support output imgs as 96*96 by hist
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

from uisbp.preprocess import BinaryLabelData, MultiLabelData
import uisbp.post_processing as postproc
from uisbp.runner import SegmentationRunner

_logpath_ = r'evaluate_linket_models.log'
_logger_ = None

DICE_CONST = 2.0
DICE_TN = 1.0
MINIMUM = 1e-6
SEG_THRESHOLD = 0.25
KEY_LABEL = None  #'CA'  # 'Plaque'  # 'BP'
KEY_LABELS = ['Plaque', 'BP', 'CA']
PREFIX_PREDICT_LABEL = 'pred'

OUTPUT_FULL_EXPECTION = True


LabelList_bp = ['ASM', 'BP', 'MSM', 'SCM', 'Vessel', '_background_']
LabelList_plaque = ['CA', 'JV', 'Plaque', '_background_']
LabelList_ca = LabelList_plaque
LabelList_bp_ca = ['ASM', 'BP', 'CA', 'JV', 'MSM', 'SCM', '_background_']
LabelList_interscalene = ['ASM', 'Interscalene', 'MSM', 'SCM', 'Vessel', '_background_']
LabelList = {
    "Plaque": LabelList_plaque,
    "CA": LabelList_ca,
    "BP": LabelList_bp,
    "Interscalene": LabelList_interscalene,
    "BP_CA": LabelList_bp_ca,
}

Thresholds_plaque = [0.1, 0.26, 1e-3, 0.26]
Thresholds_ca = Thresholds_plaque
Thresholds_bp = [0.26, 0.26, 0.26, 0.26, 0.26, 0.26]
Thresholds = {
    "Plaque": Thresholds_plaque,
    "CA": Thresholds_ca,
    "BP": Thresholds_bp,
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
    return top / bottom, top, bottom, ypred_sum, ytrue_sum


def get_dice_coef(ypred, ytrue, smooth=MINIMUM):
    top = 2 * np.sum(ypred * ytrue, axis=(1, 2, 3)) + smooth * DICE_CONST
    bottom = np.sum(ypred, axis=(1, 2, 3)) + np.sum(ytrue,
                                                    axis=(1, 2, 3)) + smooth
    return top / bottom


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


def calculate_4_matrics(dice, labels, inference, key_label):
    """
    将dice数值转化为TP TN FP FN
    当出现推理结果中有目标并且标注中也有目标但是两个目标不相交的时候FP&FN=1
    """
    if not isinstance(dice, float):
        print("dice should be float type")
        return None
    if not isinstance(labels, set):
        print("labels should be set type")
        return None

    getLogger().debug(
        f"calculate_4_matrics: dice={dice}, labels={labels}, keylabel={key_label}, inference={inference}"
    )
    result = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    if dice > 1.1 and key_label not in labels:
        result["TN"] = 1
    elif dice < 1.0 and dice >= 0.01:
        result["TP"] = 1
    elif dice < 0.01:
        if key_label not in labels:
            result["FP"] = 1
        elif key_label in labels:
            result["FN"] = 1
            if inference:
                result["FP"] = 1

    return (result, [result["TP"], result["TN"], result["FP"], result["FN"]])


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
        以每张图片的结果为一行输出到csv文件. 包括dice以及TP,FP等指标值
    """
    detailtable = []
    csv_head = ['video', 'image', 'dice', 'TP', 'TN', 'FP', 'FN']
    for img_datas in images_datas:  # iterate erery video-folder
        videoname = img_datas['videoname']
        row = [[videoname, *x] for x in img_datas['framedetails']
               ]  # get all image-info in one video-folder
        detailtable += row
    detailtable.insert(0, csv_head)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(detailtable)
    return None


def generate_video_dice_statistics_csv(filename, images_datas):
    """
        以每个video为一行输出不同threshold的dice的统计结果
    """
    summarytable = []
    threshes = [x / 100 for x in range(30, 90, 5)]
    header = ['video', 'totalframe', 'duration(ms)/frame', 'meandice'
              ] + threshes

    for img_datas in images_datas:
        videoname = img_datas['videoname']
        dices = [
            x[1] if x[1] != DICE_CONST else DICE_TN
            for x in img_datas['framedetails']
        ]
        row = [
            videoname,
            len(dices), img_datas['frameduration'],
            statistics.mean(dices)
        ]
        for thresh in threshes:
            row.append(len([d for d in dices if d >= thresh]))
        summarytable.append(row)

    row_length = len(row)
    row = ['sum']
    for i in range(1, row_length, 1):
        row.append(math.fsum([x[i] for x in summarytable]))
    summarytable.append(row)

    colstartdicecount = 4
    diffrow = ['delta', '-', '-', '-']
    for i in range(colstartdicecount, len(row) - 1):
        diffrow.append(row[i] - row[i + 1])
    diffrow.append(row[len(row) - 1])
    summarytable.append(diffrow)
    summarytable.insert(0, header)

    #print(summarytable)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(summarytable)


def generate_model_classification_matrics_csv(filename, images_datas):
    """
        计算整个测试集的Accuracy, Precision, Recall, F1
        分每个video-folder输出与总数输出
    """

    metrics_list = []
    csv_head = [
        'video', 'dice_mean', 'accuracy', 'precision', 'recall', 'F1', 'TP',
        'TN', 'FP', 'FN'
    ]
    framedetails_names = ('image', 'dice', 'TP', 'TN', 'FP', 'FN')
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    for img_datas in images_datas:  # iterate erery video-folder
        videoname = img_datas['videoname']
        img_datas_array = np.array(
            img_datas['framedetails']
        )  # shape=(img_count, 6(image, dice, TP~*4))
        img_datas_nums = img_datas_array[:, 2:].astype(float)
        tp, tn, fp, fn = (np.sum(img_datas_nums[:, i]) for i in range(4))
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        accuracy = (tp + tn) / img_datas_nums.shape[0]
        precision = (tp) / (tp + fp) if ((tp+fp)>0) else 0
        recall = (tp) / (tp + fn) if ((tp+fn)>0) else 0
        f1 = (2 * tp) / (2 * tp + fp + fn) if ((tp+fp+fn)>0) else 0
        dice_array = img_datas_array[:, 1].astype(float)
        dice_array[dice_array == DICE_CONST] = DICE_TN
        dice_mean = np.mean(dice_array)
        row = [
            videoname, dice_mean, accuracy, precision, recall, f1, tp, tn, fp,
            fn
        ]  # get all image-info in one video-folder
        metrics_list.append(row)
    
    total_acc = (total_tp + total_tn) / (total_tp+total_fp+total_tn+total_fn)
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp > 0) else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn > 0) else 0
    total_f1 = (total_tp * 2) / (2*total_tp + total_fp + total_fn)

    metrics_array = np.array(metrics_list)[:, 1:].astype(np.float)
    total_row = ['total']
    #for i in range(metrics_array.shape[-1]):
    total_row.extend([np.mean(metrics_array[:, 0]), total_acc, total_precision,\
        total_recall, total_f1, total_tp, total_tn, total_fp, total_fn])
    metrics_list.append(total_row)

    metrics_list.insert(0, csv_head)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics_list)


def outputDiceResult(out_dir, dset, folder_details):
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # output image dice
    image_dice_csv_file = os.path.join(
        out_dir, 'image_dice-' + dset + '-' + snow + '.csv')
    generate_image_statistics_csv(image_dice_csv_file, folder_details)

    # output video dice summary
    video_dice_csv_file = os.path.join(
        out_dir, 'video_dice-' + dset + '-' + snow + '.csv')
    generate_video_dice_statistics_csv(video_dice_csv_file, folder_details)

    # output video classification matrics
    video_classification_csv_file = os.path.join(
        out_dir, 'video_matrics-' + dset + '-' + snow + '.csv')
    generate_model_classification_matrics_csv(video_classification_csv_file,
                                              folder_details)


def generate_image_result(index, outputdir, imgs_idx, X, y_groundtrouth, Ypred,
                 model_label_names, masks_label_names, labelcolors, modelPara, dice):
    """
    根据预测结果生成图片预测结果, 方面进行结果效果查看
    """
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
    """
    根据预测得到的图片array(Ypred) 生成结果图片与csv文件. 如果图片小于96, 不生成图片. 
    output:
        ['frm-0001', dice, TP, TN, FP, FN]
    """
    img_outsize = modelPara['outsize']
    getLogger().debug(f'@@@debug X={X.shape}Y={y_groundtrouth.shape}')

    dice, top, bottom, ypred_sum, _ = calcDice(Ypred[index],
                                               model_label_names,
                                               y_groundtrouth[index],
                                               masks_label_names,
                                               thresh=SEG_THRESHOLD)
    metrics4 = calculate_4_matrics(dice, imgs_idx[index][1], ypred_sum,
                                   KEY_LABEL)

    full_image_name_index = imgs_idx[index][0]
    framename = full_image_name_index[-4:]
    one_image_result = ['frm-' + framename, dice]
    one_image_result.extend(metrics4[1])
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
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        #out_img = cv2.resize(output_data[0], (size, size), interpolation=cv2.INTER_AREA).reshape(size, size, output_data.shape[-1])
        out_img = output_data[0]  # 因为当最后一维通道数 > 4的时候, 不能够使用opencv进行resize了 ???. 这里输出size=model.size的模型尺寸. channel=6的时候cv2.resize是ok的.
        out_imgs.append(out_img)
    return out_imgs


def HandleOneFrameFolder(image_dir, out_dir, modelPara):
    # create data
    print("1. Create np data from input.")
    ml = MultiLabelData(image_dir, out_dir)
    model_size = modelPara['size']
    imgs, masks, label_dict, imgs_idx = ml.get_imgs_masks_labels_2(
        size=model_size, keylabel=[KEY_LABEL])
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
    ypred, _ = postproc.post_process_imgs_multi_label(imgs=np.array(ypred), thresh=labels_thresh, pc=0.001, cc=True)
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
        raise ValueError(f"modelPara['dset'] must be tflite file!!!")
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

    # models = [ ('plaquev3.0_0122', 448), ('plaquev3.0.aug_0122', 448), ('plaquev3.0_0125', 160)]
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
    print("\n"*5, "params:", sys.argv)
    parser = init_argparser()
    args = parser.parse_args()
    main(args, args.confirmed)
    print('finish...')
