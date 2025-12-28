#!/bin/python3

import json
import os

def findThePointInOneShapeLessN(json_file:str, leastPointCount:int=4):
    """
        deal with one xiaobai's json which have multi-labelme json
        - if points less than 4, ignore it. --eton@241202
    """
    #print(f"\t\tdebug:{json_file}")
    if not os.path.basename(json_file).startswith("frm-"):
        return
    if not os.path.exists(json_file):
        print(f"\tjson target not exist.[{json_file}]")
        return -1
    # 01-read from nodule-json-file
    with open(json_file, 'r') as f:
        lbm_json = json.load(f)
        lbm_shapes=lbm_json['shapes']
        if lbm_shapes is not None:
            for ishape, lbm_shapeItem in enumerate(lbm_shapes):
                lbm_pointsInOneShape=lbm_shapeItem["points"]
                pointCntInShape=len(lbm_pointsInOneShape)
                #print(f"\t{json_file} shape[{ishape}] has point in shape is: {pointCntInShape}.")
                if pointCntInShape < leastPointCount:
                    print(f"\tErr:{json_file}_shape[{ishape}] has point in shape less then {leastPointCount}>{pointCntInShape}.")
                    return -1
    return 0

def list_folders(directory):
    try:
        folders = []
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_dir():
                    folders.append(entry.name)
        return folders
    except Exception as e:
        print(f"Error: {e}")
        return []

def list_folders(directory):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out only the directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders
    except Exception as e:
        print(f"Error: {e}")
        return []

def list_File_withSuffix(directory, suffix:str=".json"):
    try:
        # List all entries in the directory
        entries = sorted(os.listdir(directory))
        # Filter out only the directories
        folders = [entry for entry in entries if entry.endswith(suffix)]
        return folders
    except Exception as e:
        print(f"Error: {e}")
        return []

def processMultiFolders(working_dir:str):
    casefolders = sorted(list_folders(working_dir))
    totalCaseCount=len(casefolders)
    #caseindex=0

    for caseindex, icase in  enumerate(casefolders):
        print(f"[{caseindex}/{totalCaseCount}] : processing... {icase}")
        icasedir=os.path.join(working_dir, icase)
        jsonInCase=list_File_withSuffix(icasedir)
        #print(foldersInCase, jsonInCase)

        if len(jsonInCase)<1 :
            print(f"caseDir:[{icasedir}] missing json/dicom-img-folder, ignore....")
            continue
        for ijsonIncase in jsonInCase:
            json_file_name = os.path.join(icasedir, ijsonIncase)
            failed = findThePointInOneShapeLessN(json_file_name, 4)
        print(f"\t\t{len(jsonInCase)} json checked.")
        # if 0 != failed:
        #     print(f"\tconvert failed!!!")
        # else:
        #     print("success,,,")



if "__main__" == __name__:
    working_dir=r"/mnt/f/240926-RayShap/241129-thyroid-datas/52-debug/thyroidNodules_axp-086.dcm_frms/"
    working_dir=r'/data/raw_data/thyroidNodules/thyNodu241202/bad-datas'
    working_dir=r'/data/raw_data/thyroidNodules/thyNodu241202/thyroidNodulesAixMarkGood31V1'
    #working_dir=r'/tmp/debug33'
    processMultiFolders(working_dir)
