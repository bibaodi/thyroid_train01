# 20190827:获取斑块+CA的面积, 斑块+CA的回声强度, 用于精细分类图片
# 20190917:支持多文件夹的同时计算以及支持命令参数进行程序调用; 将csv写入与图像处理进行逻辑代码区分; 添加argparse处理; 支持数据统计
# bibaodi
import os
import json
import sys
import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import csv
import argparse
import tqdm


class Const_Value(object):
    slabel = 'label'
    label_shapes = 'shapes'
    label_points = 'points'
    label_plaque = 'Plaque'
    label_plaque2 = '斑块:Plaque'
    label_ca = 'CA'
    label_ca2 = '颈动脉:CA'
    area_percent = 'area_percent'
    grayscale = 'grayscale'
    area = 'area'


class EffictiveAnnotation(object):
    def __init__(self, image_file, json_file=None, final_size=448,
                 clahe=False):
        self.final_size = final_size
        self.image_file = image_file
        self.image_suffix = "." + self.image_file.split('.')[-1]
        self.json_file = json_file if json_file else self.image_file.split(
            self.image_suffix)[0] + ".json"
        self.image_data = None
        self.json_data = None
        self.clahe = clahe
        self.xshift = 0
        self.ratio = 1

    def crop_img_and_remove_black_edge(self, img, true_width_left: int,
                                       true_width_right: int):
        r, c = img.shape[:2]
        true_width = true_width_right - true_width_left + 1
        if true_width < 1:
            raise ValueError("true width wrong!")
        row_start = 0
        if true_width > r:
            raise ValueError(
                f"true width can not be cuted r{r}c{c}t{true_width}")
        else:
            col_start = true_width_left
            size = true_width
        xshift = col_start
        yshift = row_start
        return img[row_start:row_start + size, col_start:col_start +
                   size], xshift, yshift

    def get_image_actual_width_index_range_mem(self, imagebuff=None):
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

    def get_image_actual_width_index_range(self, imagefile=None):
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
            return self.get_image_actual_width_index_range_mem(img)

    def correct_image_size(self, img_array, zoom=False):
        """
            crop+resize. 返回处理好的图片并返回X轴偏移以及缩放比例. 
        """
        if not img_array.any() or not isinstance(img_array, np.ndarray):
            raise ValueError("param is not np.ndarray")
        if len(img_array.shape) != 2 and len(img_array.shape) != 3:
            raise ValueError(
                "image must in IMREAD_GRAYSCALE or IMREAD_COLOR(height, row, gray)!"
            )
        true_width = self.get_image_actual_width_index_range_mem(
            img_array.copy())
        img_croped, xshift, _ = self.crop_img_and_remove_black_edge(
            img_array, true_width[0], true_width[1])
        if zoom:
            img_array_new = cv2.resize(img_croped,
                                       dsize=(self.final_size,
                                              self.final_size),
                                       interpolation=cv2.INTER_AREA)
        else:
            img_array_new = img_croped
        ratio = img_array_new.shape[0] / img_croped.shape[0]
        return img_array_new, xshift, ratio

    def read_image_and_correct_it(self, imagefile=None, zoom=False):
        """
        读取图片, 如果图片不是448*448的那么进行crop, resize
        注意: 此函数调用后需要配套修改json的坐标
        """
        #print(f"read_image_and_correct_it: file is {imagefile}")
        if not imagefile:
            imagefile = self.image_file
        if not imagefile:
            raise ValueError("imagefile is None")
        if not os.path.isfile:
            msg = f"imagefile {imagefile} not exist"
            raise ValueError(msg)
        img = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
        if not img.size:
            raise ValueError("imagefile read None")
        #print(f"read_image_and_correct_it: array shape is {img.shape}, type({type(img)})")
        xshift = 0
        ratio = 1
        if img.shape[0:2] == (self.final_size, self.final_size):
            new_img = img
        else:
            new_img, xshift, ratio = self.correct_image_size(img, zoom)
        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(7, 7))
            new_img = clahe.apply(new_img)

        self.xshift = xshift
        self.ratio = ratio
        self.image_data = new_img

        return (new_img, xshift, ratio)

    def write_new_image_to_file(self, newfile=None):
        if not newfile:
            newfile = self.image_file.split(self.image_suffix)[0] + "new.png"
        cv2.imwrite(newfile, self.image_data)
        print(f" write new image to file {newfile}")
        return

    def write_new_json_to_file(self, newfile=None):
        if not newfile:
            newfile = self.json_file.split('.json')[0] + "new.json"
        self.json_data['imagePath'] = os.path.basename(
            newfile.split('.json')[0] + self.image_suffix)
        with open(newfile, 'w') as f:
            json.dump(self.json_data, f)

    def correct_json_amend_one_shape(self, shape, xshift, ratio, finalsize):
        #print(f"correct_json_amend_one_shape: {xshift, ratio, finalsize}")
        labelname = shape[Const_Value.slabel]
        for point in shape['points']:
            x = round((point[0] - xshift) * ratio)
            y = round(point[1] * ratio)
            errormax = 10
            if 0 - x > errormax or 0 - y > errormax or \
                    x - finalsize > errormax or y - finalsize > errormax:
                print('Out of range shape position {} : ({}, {}) to ({}, {})'.
                      format(labelname, point[0], point[1], x, y))
            point[0] = min(max(x, 0), finalsize - 1)
            point[1] = min(max(y, 0), finalsize - 1)

    def get_json_from_file(self,
                           json_file=None,
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
        if not json_file:
            json_file = self.json_file
        if xshift == 0 and ratio == 1.0:
            xshift, ratio = self.xshift, self.ratio
        file_exist = True
        if os.path.isfile(json_file):
            with open(json_file) as fin:
                data = json.load(fin)
                if need_correct:
                    for shape in data["shapes"]:
                        self.correct_json_amend_one_shape(
                            shape, xshift, ratio, finalsize)
        else:
            file_exist = False
            data = json.loads(empty_json)

        self.json_data = data
        return file_exist, data


######


def shape_to_mask(img_size: tuple, points, line_width=1, point_size=1):
    """
        根据多边形的点在空白图中画出多边形, 填充像素为127的灰度. 返回一个bool值的矩阵, 矩阵中处多边形覆盖区域外全部为False
        input:
            img_size: original image size(width*heigth)
            points: all points of the polygon
        return: image mask (ndarray:dtype=bool)
    """
    img_mask = np.zeros(img_size[:2], dtype=np.uint8)
    img_mask = PIL.Image.fromarray(img_mask)
    draw = PIL.ImageDraw.Draw(img_mask)
    xy = [tuple(point) for point in points]
    if len(xy) < 3:
        errmsg = 'Polygon must have points more than 2'
        raise ValueError(errmsg)
    draw.polygon(xy=xy, outline=128, fill=128)
    img_mask = np.array(img_mask, dtype=bool)
    return img_mask


def get_polygon_area(polygon_array):
    """
        根据polygon的mask得到这个polygon的像素数量, 用来代表面积
        input:
            polygon_array: 在空白画布上画出polygon后转化为bool的矩阵
        return:
            (pixel_count, pixel_percentage)
    """
    if polygon_array.dtype != np.bool:
        errmsg = 'Polygon array must in bool dtype'
        raise ValueError(errmsg)
    pixel_count = np.count_nonzero(polygon_array)
    pixel_percentage = pixel_count / polygon_array.size
    return (pixel_count, pixel_percentage)


def get_polygon_pixel_mean_value(img_array, polygon_array):
    """
        根据polygon的mask获取polygon的所有像素的灰度值的求和与均值
        input:
            img_array: 原始的图片的灰度值的矩阵 shape=(width, heigth)
            polygon_array: polygon的mask的矩阵, 值为bool.
        return:
            (polygon_pixel_mean, polygon_pixel_sum): polygon覆盖区域像素灰度值的求和与均值
    """
    if polygon_array.dtype != np.bool:
        errmsg = 'Polygon array must in bool dtype'
        raise ValueError(errmsg)
    if img_array.dtype != np.uint8:
        errmsg = 'image array must in uint8 dtype'
        raise ValueError(errmsg)

    if img_array.shape != polygon_array.shape:
        errmsg = 'image array must in same shape'
        raise ValueError(errmsg)

    polygon_img_array = img_array.copy()
    polygon_img_array[np.logical_not(polygon_array)] = 0

    polygon_pixel_sum = np.sum(polygon_img_array)
    polygon_pixel_mean = polygon_pixel_sum / (
        np.count_nonzero(polygon_img_array) + 1)

    return (polygon_pixel_mean, polygon_pixel_sum)


def crop_sub_mask(polygon_origin, polygon_sub):
    """
        将origin的mask中剪掉sub的mask. 适用于血管中有大斑块的情况, 这样会导致血管的像素灰度值均值受到影响. 
        要求两个矩阵shape相等
        input:
            polygon_origin: 大的polygon的mask的矩阵, 原始的矩阵
            polygon_sub: 需要被剪掉的部分的矩阵
        return:
            polygon_new: 将polygon_origin中polygon_sub为True的部分改为False的矩阵
    """
    for array in (polygon_origin, polygon_sub):
        if array.dtype != np.bool:
            errmsg = 'Polygon array must in bool dtype'
            raise ValueError(errmsg)

    polygon_new = polygon_origin.copy()
    polygon_new[polygon_sub] = False

    return polygon_new


def calculate_polygon_area_and_gray(image_file=None):
    """
        计算图片的标注中的面积(像素数量)和标注区域的灰度均值. 
        因为plaque位于CA上面, 所以计算CA的面积的时候会去掉Plaque的面积
        input:
            image_file: 图片文件
        output:
            [{image: 0001, label: plaque, area: 42.0, area_percent: 0.022, grayscale: 24.0}] 针对每个标签计算面积与灰度的均值.
    """
    if not image_file:
        print(f"image file must here")
        return None
    #print(f'process file {image_file}')
    ea = EffictiveAnnotation(image_file=image_file, final_size=448, clahe=True)
    data, x, r = ea.read_image_and_correct_it(zoom=True)
    fe, json_data = ea.get_json_from_file(need_correct=True,
                                          finalsize=data.shape[0])
    polygon_values = []
    ca_mask = None
    plaque_masks = []
    for shape in json_data[Const_Value.label_shapes]:
        if shape[Const_Value.slabel] in [
                Const_Value.label_ca, Const_Value.label_ca2
        ]:
            #print("ca_mask:", shape[Const_Value.slabel])
            ca_mask = shape_to_mask(data.shape, points=shape['points'])
        if shape[Const_Value.slabel] in [
                Const_Value.label_plaque, Const_Value.label_plaque2
        ]:
            #print("plaque_mask:", shape[Const_Value.slabel])
            plaque_masks.append(
                shape_to_mask(data.shape, points=shape['points']))
    if ca_mask is None:
        print(f"no CA label in file {image_file}")
        return None
    for pmask in plaque_masks:
        parea = get_polygon_area(pmask)
        pmean, psum = get_polygon_pixel_mean_value(ea.image_data, pmask)
        polygon_values.append({
            Const_Value.slabel: Const_Value.label_plaque,
            Const_Value.area: parea[0],
            Const_Value.area_percent: parea[1],
            Const_Value.grayscale: pmean
        })
        #print(f"plaque:, mean:{pmean}, sum:{psum}")
        if np.count_nonzero(np.logical_and(ca_mask, pmask)):
            ca_mask = crop_sub_mask(ca_mask, pmask)

    caarea = get_polygon_area(ca_mask)
    camean, casum = get_polygon_pixel_mean_value(ea.image_data, ca_mask)
    polygon_values.append({
        Const_Value.slabel: Const_Value.label_ca,
        Const_Value.area: caarea[0],
        Const_Value.area_percent: caarea[1],
        Const_Value.grayscale: camean
    })
    #print(f"ca:, mean:{camean}, sum:{casum}")
    #print(f"final result, {polygon_values}")
    img_index = os.path.basename(image_file).split('.')[0].strip('frm-')
    img_item = {'image': img_index}
    for v in polygon_values:
        v.update(img_item)

    #ea.write_new_image_to_file()
    #ea.write_new_json_to_file()
    return polygon_values


def get_specific_files_in_dir(root_dir, suffix=".png"):
    if not os.path.exists(root_dir):
        print("dicom path not correct")
        raise NotADirectoryError(root_dir)
    files = []
    if os.path.isfile(root_dir):
        files.append(root_dir)
    elif os.path.isdir(root_dir):
        for i in os.listdir(root_dir):
            f = os.path.join(root_dir, i)
            if suffix in i and os.path.isfile(f):
                files.append(f)
    return files


def write_one_video_statistic_result_to_csv(results, csvfilename):
    """
    将统计结果信息写入到csv文件, 记录三个文件: 全部数据+只有斑块数据+只有ca数据
    """
    def write2csv(rows, csv_file):
        with open(csv_file, 'w', newline='') as csvf:
            fieldnames = [
                'image', Const_Value.slabel, Const_Value.area,
                Const_Value.area_percent, Const_Value.grayscale
            ]
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write2csv(results, csvfilename)
    all_values = results
    all_ca_values = get_one_type_statistic_result(all_values,
                                                  Const_Value.label_ca)
    all_plaque_values = get_one_type_statistic_result(all_values,
                                                      Const_Value.label_plaque)
    write2csv(all_ca_values, csvfilename.split('.csv')[0] + "_ca.csv")
    write2csv(all_plaque_values, csvfilename.split('.csv')[0] + "_pla.csv")
    return all_ca_values, all_plaque_values


def get_one_type_statistic_result(all_values, onetype):
    one_type_values = []
    for ot in all_values:
        if ot and ot[Const_Value.slabel] == onetype:
            one_type_values.append(ot)
    return one_type_values


def calculate_polygon_in_onevideo(video_dir=None):
    files = get_specific_files_in_dir(video_dir)
    all_values = []
    for f in files:
        values = calculate_polygon_area_and_gray(f)
        all_values.extend(values) if values and len(values) else None

    video_name = os.path.basename(video_dir).split('_frms')[0]
    csv_file = os.path.join(os.path.dirname(video_dir),
                            video_name + "_annotation_statistic.csv")
    ca_values, plaque_values = write_one_video_statistic_result_to_csv(
        all_values, csv_file)
    video_values = {video_name: all_values}
    ca_values = {video_name: ca_values}
    plaque_values = {video_name: plaque_values}
    return video_values, ca_values, plaque_values


def write_mean_statistic_to_csv(rows, csv_file):
    with open(csv_file, 'w', newline='') as csvf:
        fieldnames = [
            'video', Const_Value.area_percent, Const_Value.grayscale
        ]
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return


def get_video_mean_statistic(all_videos_values):
    """
        根据得到的所有视频的信息, 获取每个视频的均值, 默认只针对一种polygon进行处理, 这里不进一步区分
        input: dict. 每个item的key是videoname, value是以image为单位统计得到的信息的字典组成的list
        output: list of dict. 每个dict的key是videoname:area_percentage:gray, value是对应数值, 包含了各个需要统计的字段的均值.
    """
    if not isinstance(all_videos_values, dict):
        raise ValueError(f"all_videos_values must be dict")
    videos_mean = []
    mean_items = [Const_Value.area_percent, Const_Value.grayscale]
    for videoitem in all_videos_values.items():
        videoname = videoitem[0]
        video_values = videoitem[1]
        one_polygon_2info = np.zeros((len(video_values), 2))
        for i, img_value in enumerate(video_values):
            #print(f"@@@debug: img_value:{img_value}")
            for j, mean_item in enumerate(mean_items):
                one_polygon_2info[i, j] = img_value[mean_item]
        one_video_mean = np.mean(one_polygon_2info, axis=0).tolist()
        dict_info = {
            "video": videoname,
            mean_items[0]: one_video_mean[0],
            mean_items[1]: one_video_mean[1]
        }
        videos_mean.append(dict_info)
    return videos_mean


def process_multi_videos(multi_video_dir=None):
    """
        支持多个视频进行多边形的计算
    """
    if not os.path.isdir(multi_video_dir):
        raise ValueError(f"videos directory [{multi_video_dir}] not exist")
    videos = os.listdir(multi_video_dir)
    full_videos = [os.path.join(multi_video_dir, i) for i in videos]
    all_values, all_ca_values, all_plaque_values = {}, {}, {}
    for v in tqdm.tqdm(full_videos, ncols=80):
        if not os.path.isdir(v):
            continue
        if "ScanEmpty".lower() in v.lower():
            continue
        values, ca_values, plaque_values = calculate_polygon_in_onevideo(v)
        all_values.update(values)
        all_ca_values.update(ca_values)
        all_plaque_values.update(plaque_values)
    ca_means = get_video_mean_statistic(all_ca_values)
    plaque_means = get_video_mean_statistic(all_plaque_values)
    write_mean_statistic_to_csv(ca_means, os.path.join(multi_video_dir, "all_ca_mean.csv"))
    write_mean_statistic_to_csv(plaque_means, os.path.join(multi_video_dir, "all_plaque_mean.csv"))
    #return all_values, all_ca_values, all_plaque_values
    return ca_means


def init_argparser():
    parser = argparse.ArgumentParser(
        description="calculate target's(e.g. Plaque) gray-scale area.",
        usage=f"python prog.py [options] [parameters]",
        epilog="written by bibaodi")
    parser.add_argument('--onevideo',
                        '-v',
                        dest='videodir',
                        metavar='one-video-directory',
                        help=f"the full path directory for one video folder")
    parser.add_argument('--multivideo',
                        '-m',
                        dest='videosdir',
                        metavar='multi-videos-directory',
                        help=f"the full path directory for multi-video folder")
    return parser


if __name__ == "__main__":
    parser = init_argparser()
    print(sys.argv)
    if len(sys.argv) < 2:
        parser.print_help()
    else:
        args = parser.parse_args()
        if args.videodir:
            video_values = calculate_polygon_in_onevideo(args.videodir)
        elif args.videosdir:
            video_values = process_multi_videos(args.videosdir)
        else:
            parser.print_help()
        print(video_values)
