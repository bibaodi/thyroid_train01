"""
convert the files which annotated by dong in xin1zhipu extend from Xiaobai-format to labelme format
author: eton@250304
version: 0.1
"""

import os, sys
import json
import pathlib
import typing
import numpy
import numpy as np
# Add the utils directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog
from  glog import glogger as logger
import LabelmeJson

def parseXbfmtV2For301PxThyNodu(json_file:typing.Union[str, pathlib.Path], target_json_dir:typing.Union[str, pathlib.Path]):
    """
        deal with one xiaobai's json which have multi-labelme json
        - if points less than 4, ignore it. --eton@241202
    """
    
    if not os.path.exists(target_json_dir):
        print(f"json target directory not exist.")
        return -1
    # 01-read from nodule-json-file
    with open(json_file, 'r') as f:
        json_ = json.load(f)
        all_imgs_datas = json_['data']
        one_data_item = all_imgs_datas[0]

        if False:#debug
            print("json has item number=", len(json_))
            for i in json_:
                print(f"\titem[{i}]")
            print("total has data number=", len(all_imgs_datas), type(all_imgs_datas))
            print(type(all_imgs_datas[0]))

            print(f"\titems in \\data\\")
            for i in one_data_item:
                print(f"\t\t{i}")
            #print(f"\t\t lesions in data.")
            print("\t\t one_data_item['frameNumber']:", one_data_item['frameNumber'])

        ## extract infomation --all images.
        for one_data_item in all_imgs_datas:
            frameNumber = one_data_item['frameNumber']
            #--02 extract json per image
            polygonPoints=[]
            filename = f"frm-{(frameNumber+1):04d}"
            #--02.1 remove the exist json file
            target_json_name = os.path.join(target_json_dir, f"{filename}.json")
            if os.path.exists(target_json_name):
                os.remove(target_json_name)
                #print(f">>>>>delete file{target_json_name}")

            #--03 create target labelme format json
            target_json=LabelmeJson.getOneTargetObj()
            target_json['imagePath']=f"{filename}.png"
            target_json['shapes'].clear()

            lesions = one_data_item["lesions"]
            isCurrentImageHasLabelmeJsonFile=False
            if(len(lesions)>0):
                for lesion_idx in range(len(lesions)):
                    polygonPoints=lesions[lesion_idx]["polygonPoint"]

                    points_len = int((len(polygonPoints)))
                    if points_len < 4:
                        print(f"\tpoints not enough 4:[{target_json_dir}/{filename}], ignore it.")
                        continue

                    all_points = numpy.zeros((points_len, 2))
                    all_points = polygonPoints
                    #--03.2 delete zeros
                    available_pts = []
                    for pt in range(points_len):
                        #print(f"point [{pt}]={all_points[pt]}")
                        if all_points[pt][0] is None or all_points[pt][0]<10:
                            #print(f"point[{pt}] is {all_points[pt]} delete it...")
                            continue
                        else:
                            available_pts.append(all_points[pt])

                    if len(available_pts) < 4:
                        print(f"\tavailable points not enough 4:[{target_json_dir}/{filename}], ignore it.")
                        continue
                    #--04 create target labelme format json
                    #if lesion_idx>0:
                    target_json["shapes"].append(LabelmeJson.getOneShapeObj())

                    target_json['shapes'][lesion_idx]['label']='ThyNodu'
                    target_json['shapes'][lesion_idx]['points']=available_pts
                    isCurrentImageHasLabelmeJsonFile=True
            else:
                target_json['shapes'].clear()

            if False == isCurrentImageHasLabelmeJsonFile:
                continue

            #--05 write to disk
            target_json_name = os.path.join(target_json_dir, f"{filename}.json")

            if os.path.exists(target_json_name):
                os.remove(target_json_name)
                #print(f">>>>>delete file{target_json_name}")
            with open(target_json_name, 'w') as jfp:
                json.dump(target_json, jfp)
    print(f"\t<<<one case done.")

    return 0



def test():
    jfile=r'/mnt/f/241129-xin1zhipu-thyroid-datas/01-mini-batch/dong5k-20/02.202408120464.01.20176.0001.08152100144.json'
    ojpath=r'/mnt/f/241129-xin1zhipu-thyroid-datas/01-mini-batch/dong5k-20/labelme'
    parseXbfmtV2For301PxThyNodu(jfile, ojpath)

if __name__ == "__main__":
    glog.glogger = glog.initLogger("convert301PX_xbfmt_2Lbmefmt")
        
    test()

    if len(sys.argv)<2:
        print(f"Usage: App Image")
    else:
        glog.glogger = glog.initLogger("convert301PX_xbfmt_2Lbmefmt")

        casesFolder=sys.argv[1]
        #casesFolder=r"/mnt/f/241129-zhipu-thyroid-datas/10-received-datas/241216-staticPACS_censoredOut/censor_out_pre/02.202401031860.01"
        logger.info(f"ProcessHome:{casesFolder}")
        parseXbfmtV2For301PxThyNodu(casesFolder)
