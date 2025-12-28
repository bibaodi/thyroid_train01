'''
evaluate models by TestSet

Usage:
    evaluate_linket_models.py 

Argument:

OutPut:
    evaluted images, csv file which include dice and other evaluation metrics

History：
    20190902 port from estimate_mul_batch.py, and add TN TP FN FP for csv
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

from uisbp.preprocess import BinaryLabelData, MultiLabelData
from uisbp.post_processing import *
from uisbp.runner import SegmentationRunner

_logpath_ = r'evaluate_linket_models.log'
_logger_ = None

DICE_CONST = 2.0
DICE_TN = 1.0
MINIMUM = 1e-6
SEG_THRESHOLD = 0.25
#KEY_LABEL = 'CA'  # 'Plaque'  # 'BP' # 'Plaque' # 'CA'
KEY_LABEL = 'Plaque'  # 'BP' # 'Plaque' # 'CA'

OUTPUT_FULL_EXPECTION = True


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


def label_colormap(N=16):
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
             framesize,
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


def drawLabelLegend(outimg, imgwidth, imgheight, label_names, labelcolors):
    labelwidth = 45
    linelength = 20
    labelheight = 14
    font = cv2.FONT_HERSHEY_SIMPLEX

    leglabels = label_names.copy()
    if '_background_' in leglabels:
        leglabels.remove('_background_')

    for ct, lbl in enumerate(leglabels):
        color = labelcolors[lbl].astype(np.float, copy=False) * 255

        left = int(imgwidth - labelwidth - linelength)
        top = int(imgheight - labelheight * (ct + 1.5))
        right = int(imgwidth - labelwidth - 5)
        bottom = int(imgheight - labelheight * (ct + 1))

        cv2.rectangle(outimg, (left, top), (right, bottom), color, -1)
        cv2.putText(outimg, lbl, (int(imgwidth - labelwidth),
                                  int(imgheight - labelheight * (ct + 1))),
                    font, 0.4, color, 1, cv2.LINE_AA)


def drawContourRes(mask, img, frmsize, label_names, labelcolors, thresh=0.26):

    ypred = mask
    nch = ypred.shape[-1]
    outimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if '_background_' in label_names:
        hidechannel = 1
    else:
        hidechannel = 0

    for ch in range(nch - hidechannel):
        _, contours, hierarchy = cv2.findContours(
            (ypred[..., ch] > thresh).astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

        color = labelcolors[label_names[ch]].astype(np.float, copy=False) * 255
        # thickness = 4 if label_names[ch] == KEY_LABEL else 2
        thickness = 2 if label_names[ch] == KEY_LABEL else 1
        outimg = cv2.drawContours(outimg, contours, -1, color, thickness)

    # draw legend
    drawLabelLegend(outimg, frmsize[1], frmsize[0], label_names, labelcolors)
    return outimg


def drawExpMask(mask, img, frmsize, label_names, thresh=0.26):
    ypred = mask
    nch = ypred.shape[-1]

    outimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #print(outimg.shape)

    for ch in range(nch - 1):
        if label_names[ch] == KEY_LABEL:
            _, contours, hierarchy = cv2.findContours(
                (ypred[..., ch] > thresh).astype(np.uint8), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            color = (255, 0, 0
                     )  #cmap[colorlist[ch]].astype(np.float, copy=False)*255
            outimg = cv2.drawContours(outimg, contours, -1, color, -1)

    # draw legend
    #drawLegend(outimg, frmsize, label_names, cmap, colorlist)
    return outimg


def prepareBaseImage(index, outputdir, imgs_index, X):
    frmsize = 448
    full_image_name_index = imgs_index[index][0]
    imagedir = os.path.join(outputdir, full_image_name_index[:-5])
    framename = full_image_name_index[-4:]
    imagename = full_image_name_index[-4:] + '.jpg'

    imagepath = os.path.join(imagedir, imagename)
    os.makedirs(imagedir, exist_ok=True)
    #print('before squeeze', X[index].shape)
    inputimg = X[index].squeeze(2)
    #print('after squeeze', inputimg.shape)
    PADheight = 32
    imagepad = np.zeros([PADheight, frmsize], dtype=np.uint8)
    imagepad[0, 0:-60] = 128  # split line

    inputimg = np.concatenate((inputimg, imagepad), axis=0)

    return inputimg, framename, imagepath


def drawFooterMessage(image, message):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, message, (20, image.shape[0] - 10), font, 0.5,
                (0, 240, 240), 1)


def createresult(index, outputdir, imgs_idx, X, Y_Original, Y, Ypred,
                 model_label_names, masks_label_names, labelcolors, modelPara):
    """
    根据预测得到的图片array(Ypred) 生成结果图片与csv文件.
    output:
        ['frm-0001', dice, TP, TN, FP, FN]
    """
    inputimg, framename, imagepath = prepareBaseImage(index, outputdir,
                                                      imgs_idx, X)
    imgsize = inputimg.shape

    # original image
    base = cv2.cvtColor(inputimg, cv2.COLOR_GRAY2BGR)
    drawFooterMessage(base,
                      "Model: %s-%d" % (modelPara['dset'], modelPara['size']))

    results = [base]

    if OUTPUT_FULL_EXPECTION:
        base = inputimg.copy()
        yimage = Y_Original[index]
        outimge = drawContourRes(yimage,
                                 base,
                                 imgsize,
                                 masks_label_names,
                                 labelcolors,
                                 thresh=SEG_THRESHOLD)
        drawFooterMessage(outimge,
                          datetime.datetime.now().strftime('%Y/%m/%d %H:%M'))
        results.append(outimge)

    # draw predication
    base = inputimg.copy()
    yimage = Ypred[index]
    outimgeexp = drawContourRes(yimage,
                                base,
                                imgsize,
                                model_label_names,
                                labelcolors,
                                thresh=SEG_THRESHOLD)

    # draw expection
    base = inputimg.copy()
    yimage = Y[index]
    outimge = drawExpMask(yimage,
                          base,
                          imgsize,
                          masks_label_names,
                          thresh=SEG_THRESHOLD)

    # blend predication and exptection
    alpha = 0.25
    cv2.addWeighted(outimge, alpha, outimgeexp, 1 - alpha, 0, outimgeexp)

    dice, top, bottom, ypred_sum, _ = calcDice(Ypred[index],
                                               model_label_names,
                                               Y[index],
                                               masks_label_names,
                                               448,
                                               thresh=SEG_THRESHOLD)
    metrics4 = calculate_4_matrics(dice, imgs_idx[index][1], ypred_sum,
                                   KEY_LABEL)

    drawFooterMessage(outimgeexp, "Frame=%s Dice=%.2f" % (framename, dice))

    results.append(outimgeexp.copy())

    # create result
    visres = np.hstack(results)
    cv2.imwrite(imagepath, visres)

    one_image_result = ['frm-' + framename, dice]
    one_image_result.extend(metrics4[1])
    return one_image_result


def HandleOneFrameFolder(image_dir, out_dir, modelPara, seg_model):
    # create data
    print("1. Create np data from input.")
    ml = MultiLabelData(image_dir, out_dir)
    imgs, masks, label_dict, imgs_idx = ml.get_imgs_masks_labels_2(
        size=448, keyLabel=KEY_LABEL)
    masks_label_names = list(sorted(label_dict.keys()))
    print(imgs.shape, masks.shape, imgs_idx.shape)
    print("label for model:", modelPara['label_names'])
    print('label in estimation data:', masks_label_names)

    masks_unmergeBP = masks.copy()

    total_labels = sorted(
        np.unique(modelPara['label_names'] + masks_label_names))
    cmap = label_colormap()
    labelcolors = {}
    for ct, label in enumerate(total_labels):
        labelcolors[label] = cmap[ct]

    # Linknet Segmentation
    print("2. linknet segmentation.")
    post_process = True
    if post_process:
        print("process duration inlcudes post process step.")
    start = time.perf_counter()
    ypred = seg_model.predict(imgs,
                              post_process=True,
                              channel_first_model=modelPara['channel_first'])
    predFrameDuration = int(
        (time.perf_counter() - start) * 1000 / imgs.shape[0])
    print("duration(ms)/frame", predFrameDuration)

    print("3. Ouput result images.")
    finaloutputdir = out_dir
    folder_result = []
    print("output dir", finaloutputdir)
    for i in range(masks.shape[0]):
        res = createresult(i, finaloutputdir, imgs_idx, imgs, masks_unmergeBP,
                           masks, ypred, modelPara['label_names'],
                           masks_label_names, labelcolors, modelPara)
        #print("@@@debug: createresult", res)
        folder_result.append(res)
    return folder_result, predFrameDuration


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
    for img_datas in images_datas:  # iterate erery video-folder
        videoname = img_datas['videoname']
        img_datas_array = np.array(
            img_datas['framedetails']
        )  # shape=(img_count, 6(image, dice, TP~*4))
        img_datas_nums = img_datas_array[:, 2:].astype(float)
        tp, tn, fp, fn = (np.sum(img_datas_nums[:, i]) for i in range(4))
        accuracy = (tp + tn) / img_datas_nums.shape[0]
        precision = (tp) / (tp + fp) if tp else MINIMUM
        recall = (tp) / (tp + fn) if tp else MINIMUM
        f1 = (2 * tp) / (2 * tp + fp + fn) if tp else MINIMUM
        dice_array = img_datas_array[:, 1].astype(float)
        dice_array[dice_array == DICE_CONST] = DICE_TN
        dice_mean = np.mean(dice_array)
        row = [
            videoname, dice_mean, accuracy, precision, recall, f1, tp, tn, fp,
            fn
        ]  # get all image-info in one video-folder
        metrics_list.append(row)

    metrics_array = np.array(metrics_list)[:, 1:].astype(np.float)
    total_row = ['total']
    for i in range(metrics_array.shape[-1]):
        total_row.append(np.mean(metrics_array[:, i]))
    metrics_list.append(total_row)

    metrics_list.insert(0, csv_head)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics_list)


def outputDiceResult(out_dir, dset, folder_details):
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
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


def mainProcess(img_root_dir, out_dir, modelPara):
    out_dir = os.path.join(out_dir,
                           modelPara['dset'] + '_' + str(modelPara['size']))
    imgdir_names = []

    dirs = sorted(os.listdir(img_root_dir))
    print(dirs)

    for item in dirs:
        path = os.path.join(img_root_dir, item)
        if os.path.isdir(path):
            imgdir_names.append(path)

    print('modelPara:', modelPara)
    if KEY_LABEL == 'BP' or KEY_LABEL == 'Plaque':  # small object
        seg_model = SegmentationRunner(thresh=SEG_THRESHOLD,
                                       pc=0.001,
                                       do_clahe=True)
    else:  # CA # large object
        seg_model = SegmentationRunner(thresh=SEG_THRESHOLD,
                                       pc=0.05,
                                       do_clahe=True)

    seg_model.load_model(model='linknet',
                         dset=modelPara['dset'],
                         img_size=modelPara['size'])

    total = len(imgdir_names)
    folder_details = []
    for ct, image_dir in enumerate(imgdir_names):
        print("*"*32)
        print('Handle with {0} [{1}/{2}]'.format(image_dir, ct + 1, total))
        onefolder_res, frameDuration = HandleOneFrameFolder(
            image_dir, out_dir, modelPara, seg_model)

        folder_details.append({
            'videoname': image_dir.split(os.sep)[-1],
            'frameduration': frameDuration,
            'framedetails': onefolder_res
        })
        print("#"*32, '\n')
    outputDiceResult(out_dir, modelPara['dset'] + '_' + str(modelPara['size']),
                     folder_details)

    print('output dir:', out_dir)


def main(confirmed):
    #change this to point to "images" directory from newest data
    img_root = r'F:\workspace\empty_scan_test'
    global _logger_, _logpath_
    _logpath_ = os.path.join(img_root, 'precrop.log')
    img_subdirs = [
        'testSet5.0',
        #'ValidationSet5.0',
        #  'PlaqueV5.0.crop0507.Test.positivelabel',
        #  'PlaqueV5.0.crop0507.Test.total',
        #  'PlaqueV5.0.crop0507.Trainset.positivelabel',
        #  'PlaqueV5.0.crop0507.Trainset.total'
    ]

    img_dirs = [os.path.join(img_root, x) for x in img_subdirs]

    modelPara = {
        'label_names': ['ASM', 'BP', 'MSM', 'SCM', 'Vessel', '_background_'],
        'dset': 'BPV1.9.L2R.aug.c1st.b',
        'size': 96,
        'channel_first': True
    }

    #labellist = ['ASM', 'BP', 'MSM', 'SCM', 'Vessel', '_background_']
    labellist = ['CA', 'JV', 'Plaque', '_background_']
    # labellist = ['ASM', 'Interscalene', 'MSM', 'SCM', 'Vessel', '_background_']

    # models = [ ('plaquev3.0_0122', 448), ('plaquev3.0.aug_0122', 448), ('plaquev3.0_0125', 160)]
    # models = []
    models = [('Plaque.V5.0.posi.aug_0508', 160),
              ('Plaque.V5.0.posi_0508', 160),
              ('Plaque.V5.0.total.aug_0508', 160),
              ('Plaque.V5.0.total_0508', 160)]

    models = [('Plaque.V5.0.total.aug0509', 96), ('plaquev6.0.aug0719', 96),
              ('plaquev6.0.aug0726', 96), ('plaquev6.0.aug0726.2', 96),
              ('plaquev6.0.aug0729', 96)]
    models = [('Plaque.V5.0.total.aug0730', 96)]
    models = [('Plaque.V5.0.total.0731', 96)]
    models = [('emptyScan0.1_6.0.noaug0812', 96),
              ('emptyScan0.5_6.0.noaug0812', 96),
              ('emptyScan6.0.noaug0806', 96),
              ('Plaque.V5.0.total.aug0509', 96)]
    models = [('emptyScan0.1_6.0.noaug.noclahe0815', 96)]

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

    for img_dir in img_dirs:
        #result directory
        outdir = img_dir + '_out_' + KEY_LABEL + '_' + snow

        if models:
            for model in models:
                para = {
                    'label_names': labellist,
                    'dset': model[0],
                    'size': model[1],
                    'channel_first': False
                }
                #outdir += str(para['size'])
                mainProcess(img_dir, outdir, para)
        else:
            mainProcess(img_dir, outdir, modelPara)


if __name__ == '__main__':
    print("\n"*5, "params:", sys.argv)
    confirmed = False
    if len(sys.argv) > 1:
        para = sys.argv[1]
        if para.lower() == '-y':
            confirmed = True
    main(confirmed)
