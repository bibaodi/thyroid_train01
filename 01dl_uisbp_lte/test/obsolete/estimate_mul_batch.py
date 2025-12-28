'''
batch mask result for review

Usage:
    estimate_apr_batch.py 

Argument:

History：
    2018/7/26 full flow
    2018/7/27 handle data folder one by one to avoid memory overflow
    2018/10/31 handle mulitple models
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

_logpath_ = r'estimate_multi_data.log'
_logger_ = None

SEG_THRESHOLD = 0.25
#KEY_LABEL = 'CA'  # 'Plaque'  # 'BP' # 'Plaque' # 'CA'
KEY_LABEL = 'Plaque'  # 'BP' # 'Plaque' # 'CA'
MERGE_BP = False  # True
OUTPUT_FULL_EXPECTION = True


def prepareLogging():
    logger = logging.getLogger("estimate_multi_data")

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


def get_one_image_dice_coef(ypred, ytrue, smooth=1e-6):
    top = 2 * np.sum(ypred * ytrue) + smooth
    bottom = np.sum(ypred) + np.sum(ytrue) + smooth
    return top / bottom, top, bottom


def get_dice_coef(ypred, ytrue, smooth=1e-6):
    top = 2 * np.sum(ypred * ytrue, axis=(1, 2, 3)) + smooth
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

        # if label_names[ch] == KEY_LABEL:
        #     color = [255,255,255]
        #     # thickness = 4 if label_names[ch] == KEY_LABEL else 2
        #     thickness = 2 if label_names[ch] == KEY_LABEL else 1
        #     outimg = cv2.drawContours(outimg, contours, -1, color, thickness)

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


def prepareBaseImage(index, outputdir, ids, X):
    frmsize = 448
    imagedir = os.path.join(outputdir, ids[index][:-5])
    framename = ids[index][-4:]
    imagename = ids[index][-4:] + '.jpg'

    imagepath = os.path.join(imagedir, imagename)
    os.makedirs(imagedir, exist_ok=True)

    inputimg = X[index].squeeze(2)

    PADheight = 32
    imagepad = np.zeros([PADheight, frmsize], dtype=np.uint8)
    imagepad[0, 0:-60] = 128  # split line

    inputimg = np.concatenate((inputimg, imagepad), axis=0)

    return inputimg, framename, imagepath


def drawFooterMessage(image, message):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, message, (20, image.shape[0] - 10), font, 0.5,
                (0, 240, 240), 1)


def createresult(index, outputdir, ids, X, Y_Original, Y, Ypred,
                 model_label_names, masks_label_names, labelcolors, modelPara):

    inputimg, framename, imagepath = prepareBaseImage(index, outputdir, ids, X)
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

        if MERGE_BP:
            base = inputimg.copy()
            yimage = Y[index]
            outimge = drawContourRes(yimage,
                                     base,
                                     imgsize,
                                     masks_label_names,
                                     labelcolors,
                                     thresh=SEG_THRESHOLD)
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

    dice, top, bottom = calcDice(Ypred[index],
                                 model_label_names,
                                 Y[index],
                                 masks_label_names,
                                 448,
                                 thresh=SEG_THRESHOLD)

    drawFooterMessage(outimgeexp, "Frame=%s Dice=%.2f" % (framename, dice))

    results.append(outimgeexp.copy())

    # create result
    if MERGE_BP:
        rows = [np.hstack(results[:2]), np.hstack(results[2:])]
        visres = np.vstack(rows)
    else:
        visres = np.hstack(results)

    cv2.imwrite(imagepath, visres)
    #print(imagepath)
    return ['frm-' + framename, dice]


def HandleOneFrameFolder(image_dir, out_dir, modelPara, seg_model):

    # create data
    print("1. Create np data from input.")
    ml = MultiLabelData(image_dir, out_dir)
    imgs, masks, masks_label_names, label_dict, ids = ml.get_imgs_masks_labels(
        size=448, use_uncertain=False, merge_bp=MERGE_BP, keyLabel=KEY_LABEL)
    print(imgs.shape, masks.shape, ids.shape)
    print("label for model:", modelPara['label_names'])
    print('label in estimation data:', masks_label_names)

    if MERGE_BP:
        imgs, masks_unmergeBP, masks_label_names, label_dict, ids = ml.get_imgs_masks_labels(
            size=448, use_uncertain=False)
    else:
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
        res = createresult(i, finaloutputdir, ids, imgs, masks_unmergeBP,
                           masks, ypred, modelPara['label_names'],
                           masks_label_names, labelcolors, modelPara)
        folder_result.append(res)

    return folder_result, predFrameDuration


def outputDiceResult(out_dir, dset, folder_details):

    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    # output image dice
    detailtable = [[dset, snow], ['video', 'image', 'dice']]
    for fd in folder_details:
        videoname = fd['videoname']
        row = [[videoname, x[0], x[1]] for x in fd['framedetails']]
        detailtable += row

    with open(os.path.join(out_dir,
                           'image_dice-' + dset + '-' + snow + '.csv'),
              'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerows(detailtable)

    # output vidoe dice summary
    summarytable = []

    # threshes = [0.40, 0.50, 0.55, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68,
    #             0.69, 0.70, 0.72, 0.75, 0.80, 0.85]
    threshes = [x / 100 for x in range(30, 90, 5)]
    header = [[dset, snow],
              ['video', 'totalframe', 'duration(ms)/frame', 'meandice'] +
              threshes]

    for fd in folder_details:
        videoname = fd['videoname']
        dices = [x[1] for x in fd['framedetails']]
        row = [
            videoname,
            len(dices), fd['frameduration'],
            statistics.mean(dices)
        ]

        for thresh in threshes:
            row.append(len([d for d in dices if d >= thresh]))

        summarytable.append(row)

    row = ['threshold_count']
    colframenum = 1
    colvideodice = 3
    colstartdicecount = 4
    total = math.fsum([x[colframenum] for x in summarytable
                       ])  # sum product of frame count and video mean dice
    row.append(total)
    row.append('')  # column duration

    dicesum = math.fsum(
        [x[colframenum] * x[colvideodice] for x in summarytable])
    row.append(dicesum / total)

    for i in range(len(threshes)):
        row.append(math.fsum([x[i + colstartdicecount] for x in summarytable]))
    summarytable.append(row)

    diffrow = ['count in range', '', '', '']
    for i in range(colstartdicecount, len(row) - 1):
        diffrow.append(row[i] - row[i + 1])
    diffrow.append('')
    summarytable.append(diffrow)

    summarytable.insert(0, header[0])
    summarytable.insert(1, header[1])

    #print(summarytable)
    with open(os.path.join(out_dir,
                           'video_dice-' + dset + '-' + snow + '.csv'),
              'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerows(summarytable)


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
        print('Handle with %s [%d/%d]' % (image_dir, ct + 1, total))
        onefolder_res, frameDuration = HandleOneFrameFolder(
            image_dir, out_dir, modelPara, seg_model)

        folder_details.append({
            'videoname': image_dir.split(os.sep)[-1],
            'frameduration': frameDuration,
            'framedetails': onefolder_res
        })

    outputDiceResult(out_dir, modelPara['dset'] + '_' + str(modelPara['size']),
                     folder_details)

    print('output dir:', out_dir)


def main_old():
    milthresold = 0
    argument_cnt = len(sys.argv) - 1
    if argument_cnt == 4:
        milthresold = float(sys.argv[3])

    if argument_cnt == 3 or argument_cnt == 4:
        mainProcess(sys.argv[1], sys.argv[2], milthresold)
    else:
        print("Wrong command line argument.")
        print(__doc__)


def main():
    #change this to point to "images" directory from newest data
    img_root = r'F:\workspace\empty_scan_test'
    global _logger_, _logpath_
    _logpath_ = os.path.join(img_root, 'precrop.log')
    img_subdirs = [
        'testSet5.0_raw',
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
    models = [
        ('emptyScan0.1_6.0.noaug0812', 96),
        ('emptyScan0.5_6.0.noaug0812', 96),
        ('emptyScan6.0.noaug0806', 96),
        ('Plaque.V5.0.total.aug0509', 96)
    ]
    models = [('emptyScan0.1_6.0.noaug.noclahe0815', 96)]
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
    main()



