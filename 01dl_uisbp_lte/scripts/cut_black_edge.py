'''
cut the left and right black edge of image (pocus1.0.2 generated)

Usage:
    python cut_black_edge.py {imagedir} **argvs
Argument:
    imagedir : the input dir which include multi directory which contain image or image+json
OutPut:
    image or image+json (json's coordinate will fixed and image will be cutted balck edge left and right)

History：
    20200701-first released

TODO:
    None
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
import glob
import time
import math
import json
import csv, statistics
import datetime
import logging
import argparse
import hashlib
import tqdm

_logpath_ = r'black_edge_cuting.log'
_logger_ = None


def prepareLogging(level=logging.INFO):
    logger = logging.getLogger(_logpath_)
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
        if column.max() - column.min() > 5:
            left = col
            break
    for col in range(gray_img_t.shape[0] - 1, -1, -1):
        column = gray_img_t[col]
        if column.max() - column.min() > 5:
            right = col
            break
    return (left, right)


def get_image_actual_width_index_range(imagefile=None):
    """
        获取图片的实际有效宽度的索引范围
    """
    if not imagefile:
        raise ValueError("imagefile is None")
    if not os.path.isfile(imagefile):
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


def correct_image_size(img_array, zoom=False, zoomed_size=0, round=False):
    """
        crop+resize. 返回处理好的图片并返回X轴偏移以及缩放比例. 
        round: 是否保证图片尺寸都保持10像素误差的一致.
    """
    if not img_array.any() or not isinstance(img_array, np.ndarray):
        raise ValueError("param is not np.ndarray")
    if len(img_array.shape) != 2 and len(img_array.shape) != 3:
        raise ValueError(
            "image must in IMREAD_GRAYSCALE or IMREAD_COLOR(height, row, gray)!"
        )
    if zoom and zoomed_size < 1:
        raise ValueError(f"zoomed size must visualized. current is {zoomed_size}")
    true_width = get_image_actual_width_index_range_mem(img_array.copy())
    img_croped, xshift, _ = crop_img_and_remove_black_edge(
        img_array, true_width[0], true_width[1])
    getLogger().debug(f"img cropped size={img_croped.shape}")
    cropped_size = math.ceil(img_croped.shape[0] / 10) * 10 if round else img_croped.shape[0]
    if cropped_size != img_croped.shape[0]:
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


def read_image_and_correct_it(imagefile=None, zoom=False, zoomed_size=448, round=False, clahe=True):
    """
    读取图片, 如果图片不是448*448的那么进行crop, resize, final_size=zoomed_size
    注意: 此函数调用后需要配套修改json的坐标
    round: 是否保证图片尺寸都保持10像素误差的一致.
    """
    if not imagefile:
        raise ValueError("imagefile is None")
    if not os.path.isfile(imagefile):
        msg = f"imagefile {imagefile} not exist"
        raise ValueError(msg)
    img = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(7, 7))
        img = clahe.apply(img)
    if not img.size:
        raise ValueError("imagefile read None")
    new_img, xshift, ratio, cropped_img = correct_image_size(img, zoom=zoom, zoomed_size=zoomed_size, round=round)
    getLogger().debug(f"{imagefile} cropped_img(resize by ceil) size:{cropped_img.shape}")
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


def save_image_and_json(rootdir, imagepath, imagebuf, jsonbuf):
    imagedir = os.path.join(rootdir, os.path.basename(os.path.dirname(imagepath)))
    imagename = os.path.basename(imagepath)
    jsonname = imagename.split('.')[0]+'.json'
    os.makedirs(imagedir, exist_ok=True)
    cv2.imwrite(os.path.join(imagedir, imagename), imagebuf)
    if jsonbuf:
        with open(os.path.join(imagedir, jsonname), 'w') as f:
            json.dump(jsonbuf, f)
    return 


def process_then_save(img_root_dir:str, out_dir:str, params:dict):
    # get filenames
    filenames, imgs_index = get_raw_image_filenames(img_root_dir=img_root_dir, \
        extension=params['extension'], abandon_list=params['abandon_list'])

    for fname in tqdm.tqdm(filenames, ncols=80, desc="Processing Files:"):
        json_file = fname.split('.png')[0] + '.json'
        img_datas = read_image_and_correct_it(fname, zoom=False, zoomed_size=0, round=params['round'], clahe=False)
        img = img_datas[0]
        xshift = img_datas[1]
        ratio = img_datas[2]
        cropped_img = img_datas[3]
        assert img is cropped_img
        # open jsonfile
        json_exist, data = get_json_from_file(json_file,
                                        need_correct=True,
                                        xshift=xshift,
                                        ratio=ratio,
                                        finalsize=img.shape[0])
        if not json_exist:
            data = None
        save_image_and_json(out_dir, fname, img, data)
    return 0


def HandleOneFrameFolder(image_dir, out_dir, params):
    start = time.perf_counter()
    ret = process_then_save(image_dir, out_dir, params)
    
    predFrameDuration = int((time.perf_counter() - start) * 1000)
    getLogger().info(f"duration(ms)/frame:{predFrameDuration}")
    return ret, predFrameDuration


def mainProcess(img_root_dir, out_dir, params):
    out_dir = out_dir + f".{datetime.datetime.now().strftime('%Y%m%dT%H%M')}"
    imgdir_names = []
    dirs = sorted(os.listdir(img_root_dir))
    for item in dirs:
        path = os.path.join(img_root_dir, item)
        if os.path.isdir(path):
            imgdir_names.append(path)
    getLogger().info(f"all video directorys:\n{imgdir_names}\nmodelPara: {params}")

    total = len(imgdir_names)
    folder_details = []
    for ct, image_dir in enumerate(imgdir_names):
        print("*"*32)
        print('Handle with {0} [{1}/{2}]'.format(image_dir, ct + 1, total))
        files_in_dir = os.listdir(image_dir)
        if len(files_in_dir) < 1:
            getLogger().info(f"Empty dirctory : {image_dir}")
            continue
        onefolder_res, frameDuration = HandleOneFrameFolder(
            image_dir, out_dir, params)

        folder_details.append({
            'videoname': image_dir.split(os.sep)[-1],
            'frameduration': frameDuration,
            'framedetails': onefolder_res
        })
        print("#"*32, '\n')
    getLogger().info(f'output dir:{out_dir} \n\nFinish...')
    return 0


def main(args, confirmed):
    if not os.path.isdir(args.input_image_dir):
        raise ValueError(f"input dir error: {args.input_image_dir}")
    img_root = args.input_image_dir
    while os.sep == img_root[-1]:
        img_root = img_root[0: -1]
    getLogger().info(f"input root dir: {img_root}")

    if not confirmed:
        print(f"""要针对{img_root}进行黑边处理?\n确认请输入'Y'否则'N' """)
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

    outdir = f"{img_root}.cutBlackEdge"
    para = {
        'round': args.round,
        'extension':'png',
        'abandon_list':[],
        'black_it_ratio':[]
    }
    mainProcess(img_root, outdir, para)
    snow = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    getLogger().info(f"End at {snow}")


def init_argparser():
    parser = argparse.ArgumentParser(
        description="cut image black edge in left and rhght.",
        usage=f"python prog.py [options] [parameters]",
        epilog="written by bibaodi")
    parser.add_argument('--imagedir',
                        '-i',
                        required=True,
                        dest='input_image_dir',
                        help=f"the full path directory for input image folder")
    parser.add_argument('--confirmed',
                        '-y',
                        dest='confirmed',
                        action='store_true',
                        help=f"the confirm parameter")
    parser.add_argument('--round',
                        '-c',
                        dest='round',
                        action='store_true',
                        help=f"in order to keep all images in one folder in same shape.")
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
