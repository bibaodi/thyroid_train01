'''
get_square_image  -  
2019/11/11 bibaodi:
    crop useless zone ande resize label with images alone with the width. if width > height then padding zeros at image bottom.主要是用来查看生成方型图片后的效果, 防止算法出现严重错误后难以察觉
20191112 bibaodi:
    支持添加参数, 使得正常的方型图片可以在指定比例进行底部涂黑. (为了能够验证当前正常的模型对标准测试集合进行涂黑后的识别精度)
'''

import logging
import sys, os
import glob
import csv
import json
from collections import deque
import shutil
import cv2
import numpy as np
import math
import argparse

_logpath_ = r'precrop.log'
_logger_ = None

def prepareLogging():
    logger = logging.getLogger("precrop")
    
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
    return _logger_ 

def validOutputDir(path):
    '''make sure it is empty dirctory'''    
    isEmptyDir = False
    try:
       os.makedirs(path, exist_ok=True)
       items = os.listdir(path)
       isEmptyDir = len(items) == 0
    except OSError:
        pass
    return isEmptyDir

def readFileQueue(root, fileext):
    filelist = glob.glob(os.path.join(root, '**/frm*'+ fileext), recursive=True)
    filelist.sort()
    return deque(filelist)

def getLastPathPart(p):
    ''' Get last level directory and file '''
    return os.sep.join(p.split(os.sep)[-2:])
 
def generateOutputPath(inputfile, outputDir):
    newpath = os.path.join(outputDir, getLastPathPart(inputfile))
    destdir = os.path.dirname(newpath)
    if not os.path.isdir(destdir):
        os.makedirs(destdir, exist_ok=True)
    return newpath

def CopyFileRecurisively(srcfile, destfile):
    destdir = os.path.dirname(os.path.abspath(destfile))
    if not os.path.isdir(destdir):
        os.makedirs(destdir, exist_ok=True)
    shutil.copyfile(srcfile, destfile)

def loadcsvTable(tableFile):
    dicttable = {}
    with open(tableFile, encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            dicttable[row[0].strip()] = int(row[1].strip())
    return dicttable


def CopyFileToResult(inputfile, outdir):
        outfile = generateOutputPath(inputfile, outdir)
        getLogger().info('Copy {0} to {1}'.format(inputfile, outfile))
        CopyFileRecurisively(inputfile, outfile)  

def resize(img: np.ndarray, size: int=None, interpolation: int=cv2.INTER_AREA) -> np.ndarray:
    r, c, nch = img.shape
    if size is None:
        size = min(r, c)
    return cv2.resize(img, (size, size), interpolation=interpolation).reshape(size, size, nch)

def crop_img(img, size):
    
    r, c, ch = img.shape
    row_start = 0
    if r >= c:
        col_start = 0
        size = c
    else:
        col_start = int(math.ceil((c - size) / 2))
    #print(size, row_start, col_start)

    if r < size:
        newimg = np.zeros((size, c, ch), img.dtype)
        newimg[:r, :, :] = img[:r, :, :]
        img = newimg
    return img[row_start:row_start+size, col_start:col_start+size]
###--start

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
    empty_json = """{"shapes": [],"lineColor": [0,255,0,128],
            "fillColor": [
                255,0,0,128
            ],
            "imagePath": "{}",
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
        image_name = os.path.splitext(os.path.basename(json_file))[0] + '.png'
        data = json.loads(empty_json.replace('{}', image_name))
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
        getLogger().info(f"padding to image: {padding_size}")

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

    def generate_black_bottom_imgs(black_it_ratio, img):
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

    def generate_black_bottom_img_names(black_it_ratio, img_filename):
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

###--end
def CropImage(inputfile, outdir, finalsize, black_it_ratio:list=[]):
        outfile = generateOutputPath(inputfile, outdir)
        getLogger().info('Convert {0} to {1}'.format(inputfile, outfile))

        img_datas = read_image_and_correct_it(inputfile, zoom=True, zoomed_size=finalsize)
        img_black = img_datas[0]
        if len(black_it_ratio) > 0:
            non_black_height = math.ceil(img_black.shape[0] * (1 - black_it_ratio[0]))  # 以第一个比率进行涂黑
            padding_size = (img_black.shape[0] - non_black_height, *img_black.shape[1:])
            img_black[non_black_height:, ...] = np.zeros(padding_size, dtype=np.uint8)
        xshift = img_datas[1]
        ratio = img_datas[2]
        cv2.imwrite(outfile, img_black)

        return xshift, ratio


def modiOneShape(shape, xshift, ratio, finalsize):

    labelname = shape['label']
    #print("modify", labelname)

    for point in shape['points']:
        x = round((point[0]-xshift) * ratio)
        y = round(point[1] * ratio)
        #print('old point', point)
        errormax = 10
        if 0 - x > errormax or 0 - y > errormax  \
            or x - finalsize > errormax or y - finalsize > errormax:
            getLogger().warn('Out of range shape position {} : ({}, {}) to ({}, {})'.format(labelname, point[0], point[1],  x, y))

        # print('x, y:', x, y)
        point[0] = min(max(x, 0),finalsize-1) 
        point[1] = min(max(y, 0),finalsize-1) 

        #print('new point', point)

def ModiJsonfile(jsonname, imagename, resultDir, xshift, ratio, finalsize):
    with open(jsonname) as ifjson:
        labels = json.load(ifjson)

    for shape in labels["shapes"]:
        modiOneShape(shape, xshift, ratio, finalsize)
    
    outputjson = generateOutputPath(jsonname, resultDir)
    os.makedirs(os.path.dirname(outputjson), exist_ok=True)

    getLogger().info('Convert {0} to {1}'.format(jsonname, outputjson))
    with open(outputjson, 'w') as f:
        json.dump(labels, f)

def handleOneFrame(jsonname, imagename, outputdir, finalsize, black_it_ratio:list=[]):
    resultDir = outputdir
    # handle image
    xshift, ratio = CropImage(imagename, resultDir, finalsize, black_it_ratio)
    print('xshift', xshift, 'ratio', ratio)
    _, json_data = get_json_from_file(jsonname, need_correct=True,
                                        xshift=xshift,
                                        ratio=ratio,
                                        finalsize=finalsize)
    outputjson = generateOutputPath(jsonname, resultDir)
    os.makedirs(os.path.dirname(outputjson), exist_ok=True)

    getLogger().info('Convert {0} to {1}'.format(jsonname, outputjson))
    with open(outputjson, 'w') as f:
        json.dump(json_data, f)

def process(inputdir, outputdir, black_it_ratio:list=[]):
    imageQueue = readFileQueue(inputdir, '.png')
    jsonQueue = readFileQueue(inputdir, '.json')
    getLogger().info("queue image {} json {}".format(len(imageQueue), len(jsonQueue)))
    finalsize = 448
    
    while len(imageQueue):
        imagename = imageQueue.pop()
        imagebasename = os.path.splitext(imagename)[0]
        jsonname = imagebasename + '.json'
        imagedir = os.path.basename(os.path.dirname(imagename))
        getLogger().debug("Process {} and {}".format(imagename, jsonname))
        handleOneFrame(jsonname, imagename, outputdir, finalsize, black_it_ratio)
        

def main(args):
    indir = args.rootdir
    outputDir = args.outdir
    black_it_ratio = [(38.4-30)/34.8, (38.4-20)/38.4]
    if args.black_it == 1:
        black_it_ratio = black_it_ratio[:1]
    elif args.black_it == 2:
        black_it_ratio = black_it_ratio[1:]
    else:
        black_it_ratio = []
    if validOutputDir(outputDir):
        global _logger_, _logpath_
        _logpath_ = os.path.join(outputDir, 'precrop.log')
        _logger_ = prepareLogging()
        getLogger().info("Runnning: " + ' '.join(sys.argv))
        process(indir, outputDir, black_it_ratio)
    else:
        print("outputDir must be an empty directory!")

def init_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--rootdir', '-i', dest='rootdir', type=str, required=True,
                        help='Full path to root of data folder')
    parser.add_argument('--dataset', dest='dataset', type=str, default='',
                        help='Dataset type name. Valid options are [binary, multi]')
    parser.add_argument('--imagesize', dest='imagesize', type=int, default=448,
                        help='the target image size will store to npd file')
    parser.add_argument('--outdir', '-o', dest='outdir', type=str, required=True,
                        help='Full path to output directory')
    parser.add_argument('--merge_bp', dest='merge_bp', action='store_true',
                        help='Option to merge BP contours')
    parser.add_argument('--keylabel', dest='keylabel', type=str, default='',
                        help='High priority label when overlap each other')
    parser.add_argument("--abandon", dest='abandon_list', type=str, default='',
                        help='the abandoned videos list file. (contain bad labeled and bad image.etc.)')
    parser.add_argument("--black_it", dest="black_it", type=int, default=0,
                        help='fill balck color to the image bottom, for that width > height. this will 2 times size that before')
    return parser

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    main(args)