# bibaodi
# 20190810 从一个文件夹中选择百分比的文件作为子集
# 20191012 添加根据文件列表获取子集的函数.

import numpy as np
import os
import sys
import glob
import tqdm
import shutil


def get_percent_items(items: list, percentage: float, seed=42):
    """
        从一个列表中随机选取前百分比的条目
    """
    random_state = np.random.RandomState(seed)

    # randomly permuted ids
    random_items = random_state.permutation(items)

    sub_counts = int(percentage * len(random_items))
    sub_items = random_items[:sub_counts]
    sub_items.sort()
    return sub_items.tolist()


def get_sub_dirs(root_dir: str):
    """
    获取目录下所有的子目录
    """
    return [
        os.path.join(root_dir, name) for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]


def get_all_specific_files(directory: str, extension: str = "png"):
    pattern = os.path.join(directory, f"*{extension}")
    imagefiles = glob.glob(pattern)

    return imagefiles


def get_all_image_files(directory: str):
    return get_all_specific_files(directory, "png")


def get_sub_set_by_percentage(root_dir: str, out_dir: str, percentage: float):
    """
    获取一个图片数据集的子集, 从每个最小文件夹中分别获取固定比例
    """
    print("out:", out_dir)
    sub_dirs = get_sub_dirs(root_dir)
    for folder in sub_dirs:
        img_files = get_all_image_files(folder)
        img_sub_files = get_percent_items(img_files, percentage)
        json_sub_files = [imagename.split('.png')[0] + '.json' for imagename in img_sub_files]

        sub_files = img_sub_files + json_sub_files

        sub_files_part = [subfile.split(root_dir)[-1] for subfile in sub_files]
        sub_files_correct = map(lambda f: f[1:]
                                if f.startswith(os.sep) else f, sub_files_part)
        new_files = [
            os.path.join(out_dir, subfile) for subfile in sub_files_correct
        ]
        for src, dst in zip(sub_files, new_files):
            dst_dir = os.path.dirname(dst)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            if 'json' in src and not os.path.exists(src):
                continue
            shutil.copyfile(src, dst)

        print("newfile_list:", new_files)
        return new_files


def get_sub_set_by_list_file(root_dir, subsetdir, listfile):
    """
    从总的数据集中获取一个子集, 子集的名称在listfile中指定.
    rootdir: 绝对路径, 是数据所在目录
    subsetdir: 绝对路径, 新数据集合的目录
    listfile: 数据条目的名称列表文件
    return: True 成功, False失败
    """
    if not os.path.isdir(root_dir):
        print(f"rootdir:{root_dir} invalid")
        return False
    if not os.path.isdir(subsetdir):
        os.makedirs(subsetdir)
        print(f"subsetdir:{subsetdir} created!")
    if not os.path.isfile(listfile):
        print(f"listfile:{listfile} invalid")
        return False
    
    with open(listfile, 'r') as f:
        buf = f.read()
        sub_set = buf.split()
    for item in tqdm.tqdm(sub_set, ncols=80, desc="Processing Files:"):
        s = os.path.join(root_dir, item)
        d = os.path.join(subsetdir, item)
        #print(f"copy from {s} to {d}")
        shutil.copytree(s, d, copy_function=shutil.copyfile)
    return True


if __name__ == "__main__":
    example = """
        python D:\srcs\dl2_uisbp_lte\test\get_sub_image_set.py F:\workspace\empty_scan_test\testSet5.0_lit F:\workspace\empty_scan_test\testSet5.0_lit6 0.4
    """
    usage = "python get_sub_image_set.py in_dir out_dir rate|listfile"
    print(sys.argv)

    print(f"""要百分比还是根据文件列表? \n百分比方式确认请输入'P', 列表文件方式请输入'F' """)
    while True:
        process_opt = input()
        if process_opt.lower() == "p":
            process_fn = get_sub_set_by_percentage
            param3 = float(sys.argv[3])
        elif process_opt.lower() == 'n':
            process_fn = get_sub_set_by_list_file
            param3 = sys.argv[3]
        else:
            print(f"当前输入[{process_opt}]无效, 请输入P/F")
            continue
    
    if len(sys.argv) == 4:
        process_fn(sys.argv[1], sys.argv[2], param3)
    else:
        print(usage)
    pass