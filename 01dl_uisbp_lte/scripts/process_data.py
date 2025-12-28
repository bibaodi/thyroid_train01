#!/usr/bin/env python
#20190919 支持使用abandoned_list定向删除数据集中的视频文件夹
#20191106 添加参数支持将图像生成numpy矩阵时候, 增添部分为图片底部的(38.4-30)与(38.4-20)进行黑色处理. 为了支持线阵的深度小于宽度时候AI预测. 
#20200616 支持keylabel生成mask的时候, 支持多个进行顺序覆盖(plaque, im)

import argparse
import os

import numpy as np

from uisbp.preprocess import BinaryLabelData, MultiLabelData
from uisbp.data import split_data

def validOutputDir(path):
    '''make sure it is an exist empty dirctory''' 
    isEmptyDir = False
    path = os.path.abspath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"{str(e)}")
            return isEmptyDir, path
    items = os.listdir(path)
    isEmptyDir = len(items) == 0
    return isEmptyDir, path

def save_test_train_split(imgs, masks, ids, outdir):
    try:
        os.makedirs(outdir, exist_ok=True)
    except OSError as e:
        print(f"save_test_train_split_ERROR:{str(e)}")
        return False
    X_train, X_test, Y_train, Y_test, train_ids, test_ids = split_data(imgs, masks, ids, seed=144)
    np.save(f'{outdir}{os.sep}imgs_train.npy', X_train)
    np.save(f'{outdir}{os.sep}imgs_mask_train.npy', Y_train)
    np.savetxt(f'{outdir}{os.sep}imgs_fname_train.txt', train_ids, fmt='%s')

    np.save(f'{outdir}{os.sep}imgs_test.npy', X_test)
    np.save(f'{outdir}{os.sep}imgs_mask_test.npy', Y_test)
    np.savetxt(f'{outdir}{os.sep}imgs_fname_test.txt', test_ids, fmt='%s')
    return True

def init_parser():
    parser = argparse.ArgumentParser(description='Process data for use in segmentation and classification networks')

    parser.add_argument('--rootdir', dest='rootdir', type=str, required=True,
                        help='Full path to root of data folder')
    parser.add_argument('--dataset', dest='dataset', type=str, default='',
                        help='Dataset type name. Valid options are [binary, multi]')
    parser.add_argument('--imagesize', dest='imagesize', type=int, default=448,
                        help='the target image size will store to npd file')
    parser.add_argument('--outdir', dest='outdir', type=str, required=True,
                        help='Full path to output directory')
    parser.add_argument('--merge_bp', dest='merge_bp', action='store_true',
                        help='Option to merge BP contours')
    parser.add_argument('--keylabels', dest='keylabels', type=str, default='', nargs='+',
                        help='High priority labels when overlap each other, first will be covered by after')
    parser.add_argument("--abandon", dest='abandon_list', type=str, default='',
                        help='the abandoned videos list file. (contain bad labeled and bad image.etc.)')
    parser.add_argument("--black_it", dest="black_it", action='store_true', default=False,
                        help='fill balck color to the image bottom, for that width > height. this will 2 times size that before')
    return parser


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    out_path = args.outdir
    valied, out_path = validOutputDir(out_path)
    while not valied:
        print(f"OutPut Folder is not Empty!\n{out_path}\noverwrite it input ['Y'], else input new path.")
        input_str = input()
        if 'y' != input_str.lower():
            out_path = input_str
        else:
            break
        valied, out_path = validOutputDir(out_path)
    if args.dataset == 'binary':
        bl = BinaryLabelData(args.rootdir, args.outdir)
        bl.process_raw_images_to_file(img_type='positive')
        bl.process_raw_images_to_file(img_type='negative')
        imgs, masks, ids = BinaryLabelData.get_np_files_from_images(f'{args.outdir}{os.sep}processed_images', size=448)
        save_test_train_split(imgs, masks, ids, f'{args.outdir}{os.sep}np_data{os.sep}')

    elif args.dataset == 'multi':
        black_it_ratio = [(38.4-30)/34.8, (38.4-20)/38.4] if args.black_it else []  #预设置黑色涂抹占的比例
        ml = MultiLabelData(args.rootdir, args.outdir)
        size = args.imagesize
        print(f"size={size}, dset={args.rootdir}, keylabels={args.keylabels}, black_it_ratio={black_it_ratio}")
        if os.path.isfile(args.abandon_list):
            abandonlist_file = args.abandon_list
            with open(abandonlist_file, 'r') as f:
                abandon_list = f.readlines()
            for i in abandon_list:
                if i.startswith('#'):
                    abandon_list.remove(i)
            abandon_list = [i.strip('\n \r\t') for i in abandon_list]
        else:
            abandon_list = []
        #imgs, masks, label_names, label_dict, ids = ml.get_imgs_masks_labels(size=448, use_uncertain=False,
        #        merge_bp=args.merge_bp, keylabels=args.keylabels, abandon_list=abandon_list)
        imgs, masks, label_dict, touple4index_shapeslabels = ml.get_imgs_masks_labels_2(
            size=size, keylabels=args.keylabels, abandon_list=abandon_list, black_it_ratio=black_it_ratio)
        label_names = list(sorted(label_dict.keys()))
        imgs_serial_nums = touple4index_shapeslabels[:, 0]
        print(
            f"imgs{imgs.shape};masks {masks.shape};imgs_serial_nums {imgs_serial_nums.shape};label_names {label_names}")

        save_test_train_split(imgs, masks, imgs_serial_nums, f'{out_path}{os.sep}np_data{os.sep}')
        np.savetxt(f'{out_path}{os.sep}np_data{os.sep}labels.txt', label_names, fmt='%s')

    else:
        raise ValueError(f'Unknown data type {args.dataset}. Allowed options are [binary, multi]')