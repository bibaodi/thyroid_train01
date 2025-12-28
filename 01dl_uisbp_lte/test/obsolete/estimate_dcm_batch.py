'''
batch mask result for review

Usage:
    estimate_apr_batch.py dicom_file

Argument:
    dicom_file: the dicom file which contain jpeg images not videos
History：
    2018/7/26 full flow
    2018/7/27 handle data folder one by one to avoid memory overflow
    2018/10/31 handle mulitple models
    2019/05/07 support use dicom as input file to predict image target
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import glob
import re
import time
import math
import json
import csv, statistics
import datetime
import seaborn as sns
from tqdm import tqdm

import matplotlib.pyplot as plt
import pydicom
import sys
from pydicom.data import get_testdata_files
from typing import Any
sys.path.append('..')

#from uisbp.preprocess import BinaryLabelData, MultiLabelData
from uisbp.post_processing import *
from uisbp.runner import SegmentationRunner
from uisbp.transform_utils import crop, resize, adjust_hist

IMG_FILENAME_PATTERN = re.compile(r"/(?P<hospital_id>\d+)_(?P<patient_number>\d+)_(?P<patient_name>[a-zA-Z]+)_(?P<scan>BP(?:(?:_L)|(?:_R))?)_(?P<sequence>\d+)/(?P<img_type>[a-zA-Z_0-9]+)/(?P<filename>[a-zA-Z_0-9]+)\..*")
SEG_THRESHOLD = 0.25
KEY_LABEL = 'Plaque'
MERGE_BP = False
OUTPUT_FULL_EXPECTION = True

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

def label_colormap(N=16):
    colors = sns.color_palette("bright", n_colors=N)
    col = np.array(colors)
    return col

def get_one_image_dice_coef(ypred, ytrue, smooth=1e-6):
    top = 2 * np.sum(ypred * ytrue) 
    bottom = np.sum(ypred) + np.sum(ytrue) + smooth
    return top / bottom, top, bottom

def get_dice_coef(ypred, ytrue, smooth=1e-6):
    top = 2 * np.sum(ypred * ytrue, axis=(1,2,3)) + smooth
    bottom = np.sum(ypred, axis=(1,2,3)) + np.sum(ytrue, axis=(1,2,3)) + smooth
    return top / bottom 

def get_f1(ypred, ytrue):
    top = 2 * np.sum(ypred * ytrue) 
    bottom = np.sum(ypred) + np.sum(ytrue) + 1e-6
    return top / bottom

def calcDice(y, model_label_names, mask, masks_label_names, framesize, thresh=0.26):

    # get pred result region 
    ch = model_label_names.index(KEY_LABEL)
    ypred = y[...,ch]>thresh
    # get mask region
    # grey
    ch = masks_label_names.index(KEY_LABEL)
    ytrue = mask[...,ch]
    # calc dice
    return get_one_image_dice_coef( ypred, ytrue)


def drawLabelLegend(outimg, imgwidth, imgheight, label_names, labelcolors):
    labelwidth = 45
    linelength = 20
    labelheight = 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    leglabels = label_names.copy()
    if '_background_' in leglabels:
        leglabels.remove('_background_')

    for ct, lbl in enumerate(leglabels):
        color = labelcolors[lbl].astype(np.float, copy=False)*255
        
        left = int(imgwidth-labelwidth-linelength)
        top =  int(imgheight-labelheight*(ct+1.5))
        right =  int(imgwidth-labelwidth-5)
        bottom = int(imgheight-labelheight*(ct+1))  

        cv2.rectangle(outimg, (left,top), (right, bottom), color, -1)
        cv2.putText(outimg, lbl, (int(imgwidth-labelwidth), int(imgheight-labelheight*(ct+1))),
                font, 0.4, color, 1, cv2.LINE_AA)

def drawContourRes(mask, img, frmsize, label_names, labelcolors, thresh=0.26):

    ypred = mask
    nch = ypred.shape[-1]
    outimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if '_background_' in label_names:
        hidechannel = 1
    else:
        hidechannel = 0

    for ch in range(nch-hidechannel):
        _, contours, hierarchy = cv2.findContours((ypred[...,ch]>thresh).astype(np.uint8),  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        color =labelcolors[label_names[ch]].astype(np.float, copy=False)*255
        thickness = 2 if label_names[ch] == KEY_LABEL else 1
        outimg = cv2.drawContours(outimg, contours, -1, color, thickness)


    drawLabelLegend(outimg, frmsize[1], frmsize[0], label_names, labelcolors)
    return outimg

def drawExpMask(mask, img, frmsize, label_names, thresh=0.26):
    ypred = mask
    nch = ypred.shape[-1]

    outimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #print(outimg.shape)

    for ch in range(nch-1):
        if label_names[ch] == KEY_LABEL:
            _, contours, hierarchy = cv2.findContours((ypred[...,ch]>thresh).astype(np.uint8),  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            color = (255,0,0) #cmap[colorlist[ch]].astype(np.float, copy=False)*255
            outimg = cv2.drawContours(outimg, contours, -1, color, -1)
    return outimg

def prepareBaseImage(index, outputdir, ids, X):
    frmsize = 448
    #imagedir = os.path.join(outputdir, ids[index][:-5])
    imagedir = outputdir
    framename = ids[index][-4:]
    imagename = ids[index][-4:] + '.jpg'
    
    imagepath = os.path.join(imagedir, imagename)
    os.makedirs(imagedir, exist_ok=True)
    
    inputimg = X[index].squeeze(2)
    
    PADheight = 32
    imagepad = np.zeros([PADheight, frmsize], dtype=np.uint8)
    imagepad[0,0:-60] = 128 # split line

    inputimg = np.concatenate((inputimg, imagepad),axis=0)

    return inputimg, framename, imagepath

def drawFooterMessage(image, message):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, message, (20, image.shape[0]-10),
            font, 0.5, (0,240,240), 1)   

def createresult(index, outputdir, ids, X, Y_Original, Y, Ypred, 
    model_label_names, masks_label_names, labelcolors, modelPara):
    #print("####", type(Y_Original), Y_Original.shape)
    inputimg, framename, imagepath = prepareBaseImage(index, outputdir, ids, X)
    imgsize = inputimg.shape

    # original image
    base = cv2.cvtColor(inputimg, cv2.COLOR_GRAY2BGR)
    drawFooterMessage(base, "Model: %s-%d"%(modelPara['dset'], modelPara['size']))
   
    results = [base]

    # draw predication
    base = inputimg.copy()
    yimage = Ypred[index]
    outimgeexp = drawContourRes(yimage, base, imgsize, model_label_names, labelcolors, thresh=SEG_THRESHOLD)
    drawFooterMessage(outimgeexp, datetime.datetime.now().strftime('%Y/%m/%d %H:%M'))

    results.append(outimgeexp.copy())

    # create result
    if MERGE_BP:
        rows = [np.hstack(results[:2]), np.hstack(results[2:])]
        visres = np.vstack(rows)
    else:
        visres = np.hstack(results) 

    cv2.imwrite(imagepath, visres)
    #print(imagepath)
    return ['frm-'+framename, 0]


def generateImagesFromDicom(dataset, indent=0, filename="demo"):
    """Go through all items in the dataset and print them with custom format

    Modelled after Dataset._pretty_str()
    """
    dont_print = ['Pixel Data', 'File Meta Information Version']
    dont_print = []

    indent_string = "   " * indent
    next_indent_string = "   " * (indent + 1)

    for data_element in dataset:
        if data_element.VR == "SQ":   # a sequence
            print(indent_string, data_element.name)
            for sequence_item in data_element.value:
                generateImagesFromDicom(dataset=sequence_item, indent=indent + 1,filename=filename)
                print(next_indent_string + "---------")
        else:
            if data_element.name in dont_print:
                print("""<item not printed -- in the "don't print" list>""")
            elif data_element.name=='Pixel Data' : 
                # check frame number and tranfersyntex here

                frmpath = filename+'_frms'
                os.makedirs(filename+'_frms', exist_ok=True)
                frmindex = 1
                print(indent_string, 'Extract frames from [Pixel Data]:')
                for frm in pydicom.encaps.generate_pixel_data_frame(data_element.value):
                
                    frmname = os.path.join(frmpath, "frm-%04d.png"%(frmindex,))
                    frmindex += 1
                    print("generateImagesFromDicom: %s %s : % 8d bytes"%(next_indent_string, frmname, len(frm)))

                    with open(frmname, 'wb+') as f:
                        f.write(frm)
            else:
                repr_value = repr(data_element.value)
                if len(repr_value) > 50:
                    repr_value = repr_value[:50] + "..."
                print("{0:s} {1:s} = {2:s}".format(indent_string,
                                                   data_element.name,
                                                   repr_value))
    if indent ==0:
        return frmpath



class BinaryLabelData:

    def __init__(self, img_root_dir: str, outdir: str, img_filename_pattern: Any=IMG_FILENAME_PATTERN):
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
        pattern = os.path.join(self.img_root_dir, f"**/{img_type}/*{extension}")
        img_filenames = sorted(glob.glob(pattern, recursive=True))

        return img_filenames

    def get_imgs_output_filepath(self, img_filepath, root_output_dir, mask=False):
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
        return os.path.join(root_output_dir, img_info['img_type'], output_filename)

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

    def process_raw_images_to_file(self, img_type='positive', output_dir='processed_images', clahe=False):
        """
        Process raw images to cropped images with just ultrasound.

        Args:
            img_root_dir (str): Full path to root directory of stored images
            img_type (str): Image type options are ['1', '0', 'mark']
            output_dir (str): Root output directory for cropped images
        """
        output_dir = f'{self.outdir}{os.path.sep}{output_dir}'
        os.makedirs(f'{output_dir}{os.path.sep}{img_type}', exist_ok=True)

        size_dict = defaultdict(lambda: 0)

        # get both images and masks
        if img_type == 'positive':
            img_files = self.get_raw_image_filenames(img_type=img_type, extension='jpg')
            mask_files = self.get_raw_image_filenames(img_type=img_type, extension='bboxes.tsv')

            for img_file, mask_file in zip(img_files, mask_files):

                if img_file.split(os.path.sep)[-1].split('.')[0] != mask_file.split(os.path.sep)[-1].split('.')[0]:
                    raise ValueError(f'Image file {img_file} and mask file {mask_file} do not match!')

                try:
                    img_outdir = self.get_imgs_output_filepath(img_file, root_output_dir=output_dir, mask=False)
                    mask_outdir = self.get_imgs_output_filepath(mask_file, root_output_dir=output_dir, mask=True)
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
            img_files = self.get_raw_image_filenames(img_type=img_type, extension='jpg')
            for img_file in img_files:
                try:
                    img_outdir = self.get_imgs_output_filepath(img_file, root_output_dir=output_dir, mask=False)
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
        imgfiles = glob.glob(f'{datapath}/positive/*bmp')
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
            imgs.append(resize(crop_img(cv2.imread(imgfile, 0))[..., None], size=size))
            masks.append(
                resize(crop_img(cv2.imread(maskfile, 0))[..., None], size=size, interpolation=cv2.INTER_NEAREST))
            ids.append(imgfile.split(os.path.sep)[-1].split('.')[0])

        # non-nerve images and masks
        imgfiles = glob.glob(f'{datapath}/negative/*bmp')
        x = sorted(list(filter(lambda x: 'mask' not in x.lower(), imgfiles)))
        for imgfile in x:
            imgs.append(resize(crop_img(cv2.imread(imgfile, 0))[..., None], size=size))
            ids.append(imgfile.split(os.path.sep)[-1].split('.')[0])
            masks.append(np.zeros((size, size, 1), dtype=np.uint8))

        img_array = np.array(imgs)
        mask_array = np.array(masks)
        print(img_array.shape, mask_array.shape, len(ids))
        return img_array, mask_array, np.array(ids)


class MultiLabelData:

    def __init__(self, img_root_dir: str, outdir: str):
        self.img_root_dir = img_root_dir
        self.outdir = outdir

    def get_raw_image_filenames(self, extension='png'):
        """
        Returns full path to all image files of type `img_type`

        Args:
            extension (str): File-type extension [default 'jpg']

        Returns:
            (list): List of image filenames
        """
        pattern = os.path.join(self.img_root_dir, f"**/*{extension}")
        img_filenames = sorted(glob.glob(pattern, recursive=True))

        ids = []
        for fn in img_filenames:
            base = fn.split(os.path.sep)[-2].split('_frms')[0]
            frame = fn.split(os.path.sep)[-1].split('.')[0].split('-')[-1]
            ids.append('_'.join([base, frame]))
        print("####get_raw_image_filenames 0-3:", img_filenames[0:3], ids[0:3])
        print("####get_raw_image_filenames: --end")
        return img_filenames, ids

    def get_imgs_masks_labels_for_mobile(self, extension='png', use_uncertain=False, size=448, keyLabel=''):

            assert len(keyLabel)>=0 # keyLabel must be something
            # get filenames
            filenames, ids = self.get_raw_image_filenames(extension=extension)

            # get label names
            label_dict = {'_background_': 0}
            ct = 0

            label_names = list(sorted(label_dict.keys()))
            uncertain = 0
            if not use_uncertain and "UncertainZone" in label_names:
                label_names.pop(label_names.index('UncertainZone'))
                uncertain = 1

            # make masks
            imgs, masks = [], []
            for fn in tqdm(filenames, desc="Processing Files", ncols=80):
                img = cv2.imread(fn, 0)
                imgs.append(img)

            # get arrays
            img_array = np.array([resize(crop_img(img[..., None]), size) for img in imgs])
            mask_array = np.zeros((len(masks), size, size, len(label_names)), dtype=np.uint8)
            for ct, mask in enumerate(masks):
                mask_array[ct, ...] = np.dstack([resize(crop_img(mask[..., ii][..., None]), size) for ii in range(mask.shape[-1])])

            return img_array, mask_array, label_names, label_dict, np.array(ids)

def HandleOneFrameFolder(image_dir, out_dir, modelPara, seg_model):

    # create data 
    print("1. Create np data from input.")
    ml = MultiLabelData(image_dir, outdir)
    imgs, masks, masks_label_names, label_dict, ids = ml.get_imgs_masks_labels_for_mobile(size=448, 
        use_uncertain=False, keyLabel=KEY_LABEL)
    print(imgs.shape, masks.shape, ids.shape) # 这里得到图像,但是没有得到mask
    print("label for model:", modelPara['label_names'] )
    print('label in estimation data:', masks_label_names)

    masks_unmergeBP = imgs.copy()

    total_labels = sorted(np.unique(modelPara['label_names'] + masks_label_names))
    cmap = label_colormap()
    labelcolors = {}
    for ct, label in enumerate(total_labels):
        labelcolors[label] = cmap[ct]

    # Linknet Segmentation
    print("2. linknet segmentation.")
    post_process = True
    if post_process: print("process duration inlcudes post process step.")
    start = time.perf_counter()
    ypred = seg_model.predict(imgs, post_process=True, channel_first_model=modelPara['channel_first'])
    predFrameDuration = int((time.perf_counter() - start)*1000/imgs.shape[0])
    print("duration(ms)/frame", predFrameDuration)

    print("3. Ouput result images.")
    finaloutputdir = out_dir
    folder_result = []
    print("output dir", finaloutputdir)
    import copy
    masks = copy.deepcopy(imgs)
    print("masks:", masks.shape)
    for i in range(imgs.shape[0]):
        res = createresult(i, finaloutputdir, ids, imgs, masks_unmergeBP, masks, 
                ypred, modelPara['label_names'], masks_label_names, labelcolors, modelPara)
        folder_result.append(res)

    return folder_result, predFrameDuration

def mainProcess(dcm_imges_dir, out_dir, modelPara):
    #out_dir = os.path.join(out_dir, modelPara['dset']+'_'+str(modelPara['size']))
    imgdir_names = []
    print("dcm_imges_dir:", dcm_imges_dir)
    dirs = sorted(os.listdir(dcm_imges_dir))   
    #print(dirs)
    for item in dirs:
        path = os.path.join(dcm_imges_dir, item)
        if os.path.isdir(path):
            imgdir_names.append(path)
    if not len(imgdir_names):
        imgdir_names.append(dcm_imges_dir)
    
    print('modelPara:', modelPara)
    if KEY_LABEL == 'BP' or KEY_LABEL == 'Plaque': # small object
        seg_model = SegmentationRunner(thresh=SEG_THRESHOLD, pc=0.001, do_clahe=True)
    else:
        seg_model = SegmentationRunner(thresh=SEG_THRESHOLD, pc=0.05, do_clahe=True) 

    seg_model.load_model(model='linknet', dset=modelPara['dset'] , img_size=modelPara['size'])


    total = len(imgdir_names)
    folder_details = []
    for ct, image_dir in enumerate(imgdir_names):
        print('Handle with %s [%d/%d]' %(image_dir, ct+1, total))
        onefolder_res, frameDuration = HandleOneFrameFolder(image_dir, out_dir, modelPara, seg_model)

        folder_details.append( { 'videoname': image_dir.split(os.sep)[-1], 
                                'frameduration':frameDuration, 
                                'framedetails':onefolder_res} )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print('...Finish...\noutput dir is:\n', out_dir)

def pre_process_dicom(dicom_file):
    print("dcmfile:", dicom_file)
    if os.path.isfile(dicom_file):
        print("dcmfile is file? :", dicom_file)
        ds = pydicom.dcmread(dicom_file)
        image_dir = generateImagesFromDicom(ds, filename=dicom_file)
        return image_dir
    else:
        return ''

def main():
    milthresold = 0
    argument_cnt = len(sys.argv) - 1
    if argument_cnt == 4:
        milthresold = float(sys.argv[3])

    if argument_cnt == 3 or argument_cnt == 4:
        mainProcess(sys.argv[1], sys.argv[2], milthresold)    
    else:
        print("Wrong command line argument.")
        print(__doc__)


if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("param not enough: please use 'python app dicom'")
        exit(-1)
    dcm_root = sys.argv[1]
    if not os.path.exists(dcm_root):
        print("dicom path not correct")
        exit(-1)
    dcm_files = []
    if os.path.isfile(dcm_root):
        dcm_files.append(dcm_root)
    elif os.path.isdir(dcm_root):
        for i in os.listdir(dcm_root):
            f = os.path.join(dcm_root, i)
            if '.dcm' in i and os.path.isfile(f):
                dcm_files.append(f)
    img_dirs = []
    print('dcm_files:', dcm_files)
    for df in dcm_files:
        img_dir = pre_process_dicom(dicom_file=df)
        img_dirs.append(img_dir)
    print('img_dirs:', img_dirs)

    modelPara = {
         'label_names':  ['CA', 'JV', 'MSM', 'Plaque', 'Vessel', '_background_'],
         'dset': 'BPV1.9.L2R.aug.c1st.b',
         'size': 96,
         'channel_first':True
    }

    labellist = ['CA', 'JV', 'Plaque', '_background_']

    models = [('apr', 96)]

    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    print("Start at {}".format(snow, ))
    for img_dir in img_dirs:
        outdir = img_dir + '_out_' + KEY_LABEL + '_' + snow
        if models:
            for model in models:
                para = {
                    'label_names' : labellist,
                    'dset': model[0],
                    'size': model[1],
                    'channel_first': False
                }
                mainProcess(img_dir, outdir, para)
        else:
            mainProcess(img_dir, outdir, modelPara)
        import shutil
        shutil.rmtree(img_dir)