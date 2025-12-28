#!/usr/bin/env python
#20191022 初版. 为了能够看到数据裁剪后的效果, 创建此程序-bibaodi

import argparse
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from uisbp import preprocess as prep

def init_parser():
    parser = argparse.ArgumentParser(description='Process data for use in segmentation and classification networks')

    parser.add_argument('--rootdir', '-i', dest='rootdir', type=str, required=True,
                        help='Full path to root of data folder')
    parser.add_argument('--dataset', dest='dataset', type=str, default='multi',
                        help='Dataset type name. Valid options are [binary, multi]')
    parser.add_argument('--outdir', '-o', dest='outdir', type=str, required=True,
                        help='Full path to output directory')
    parser.add_argument('--merge_bp', dest='merge_bp', action='store_true',
                        help='Option to merge BP contours')
    parser.add_argument('--keylabel', dest='keylabel', type=str, default='BP',
                        help='High priority label when overlap each other')
    parser.add_argument("--abandon", dest='abandon_list', type=str, default='',
                        help='the abandoned videos list file. (contain bad labeled and bad image.etc.)')
    parser.add_argument("--size", dest='size', type=int, default=448,
                        help='the target image size')
    return parser

def main(args):
    rootdir = args.rootdir
    outdir = args.outdir
    size = args.size
    while os.sep == rootdir[-1]:
        rootdir = rootdir[0: -1]
    while os.sep == outdir[-1]:
        outdir = outdir[0: -1]
    
    if args.dataset == 'binary':
        print("not support! Sorry. ")
        return 
    elif args.dataset == 'multi':
        mld = prep.MultiLabelData(rootdir, outdir)
        filenames, imgs_index = mld.get_raw_image_filenames(extension='png')
        for fn in tqdm(filenames, ncols=80, desc="Processing Files:"):
            json_file = fn.split('.png')[0] + '.json'

            img_datas = prep.read_image_and_correct_it(fn, zoom=False, zoomed_size=size)
            img = img_datas[0]
            xshift = img_datas[1]
            ratio = img_datas[2]
            # open jsonfile
            _, new_json = prep.get_json_from_file(json_file,
                                         need_correct=True,
                                         xshift=xshift,
                                         ratio=ratio,
                                         finalsize=img.shape[0])

            new_img_file = fn.replace(rootdir, outdir)
            new_json_file = json_file.replace(rootdir, outdir)
            if not os.path.exists(os.path.dirname(new_img_file)):
                os.makedirs(os.path.dirname(new_img_file))
            cv2.imwrite(new_img_file, img)
            with open(new_json_file, 'w') as f:
                json.dump(new_json, f, indent=2)
        return 
    else:
        usage = 'python app -i [rootdir] -o [outdir] '
        return print(usage)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    main(args)