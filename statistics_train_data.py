#!/bin/python3
# statistics the data features about train and test.
# eton@241207

import os
import sys
from pathlib import Path
import pandas as pd

trainResFolderParent=r'/mnt/f/241129-zhipu-thyroid-datas/51-trainResults'
resFolder=r'res_optimizeThyNod1.6_mul_nodules1.5_axpG63_224_20241206T0943_sz224/'



trainDataListfName='imgs_fname_train.txt'
validateDataListfName='imgs_fname_test.txt'

datasetTypes=[validateDataListfName, trainDataListfName]

#thyroidNodules_axp-013.dcm_0001
dcmlistSuffix=".dcm"
dcmfolderSuffix=dcmlistSuffix+"_frms" 


# List of image file extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
json_extension="*.json"
dataActualFolder=r'/mnt/f/241129-zhipu-thyroid-datas/31-labelmeFormatOrganized/241207_all82AixThyroidNodules'

def statis_emptyJson_and_total(onecasename:str):
    onedcmfolder=onecasename+dcmfolderSuffix
    dcmpath=os.path.join(dataActualFolder, onedcmfolder)
    if not os.path.isdir(dcmpath):
        print(f"Err:file:{onedcmfolder} not exist, ignore {dcmpath}")
        return
    files_in_dir = os.listdir(dcmpath)
    
    if len(files_in_dir) < 2:
        print(f"Empty dirctory : {dcmpath}")
        return
    # Define the folder path
    folder_path = Path(dcmpath)
    # List to store image file paths
    image_files = []
    json_files=[]

    # Loop through each extension and find matching files
    for extension in image_extensions:
        image_files.extend(folder_path.glob(extension))
    json_files.extend(folder_path.glob(json_extension))

    lmstartJsonfile=[ii for ii in json_files if "lmstart.json" in str(ii)]
    if len(lmstartJsonfile)>0:
        json_files.remove(lmstartJsonfile[0])

    #print(f"debug: images count:{len(image_files)}, json count{len(json_files)}")
    return (len(image_files), len(json_files))

def processImglistLines(imgnamelines:list):
    dcmnameset=set()
    dcmname=None
    for imgline in imgnamelines:
        nameParts=imgline.split(dcmlistSuffix)
        if type(nameParts) is list and len(nameParts)>1:
            if len(nameParts[0]) > 2:
                dcmname=nameParts[0]
        if type(dcmname) is not None:
            dcmnameset.add(dcmname)
    #print(imgnameline.split(".dcm"))
    sortedList=sorted(dcmnameset)
    #print(f"debug: dcm name set:{type(sortedList)}")
    return sortedList

def readImgListFile(trainResFolder:str):
    #iOneResfolder=os.path.join(trainResFolder, resFolder)
    iOneResfolder=trainResFolder
    print(iOneResfolder)

    dirItems = sorted(os.listdir(iOneResfolder))
    print(dirItems)
    
    for iitem in dirItems:
        statis_list=[]
        if iitem in datasetTypes:
            imglists=None
            
            iitemPath=os.path.join(iOneResfolder, iitem)
            if not os.path.isfile(iitemPath):
                print(f"Err:file:{iitem} not exist, ignore")
            with open(iitemPath, 'r') as fp:
                imglists=fp.readlines()
            if type(imglists) == None:
                print(f"Err: read from file[{iitem}] failed.!")
                return
            print(f"len of file:[{iitem}]={len(imglists)}, first img={imglists[0] if len(imglists)>0 else imglists}")
            dcmNameset=processImglistLines(imglists)
            print(f"{iitem} contains[{len(dcmNameset)}]:")
            for dcmname in dcmNameset:
                imgCnt, jsonCnt=statis_emptyJson_and_total(dcmname)
                print(f"{dcmname}: {imgCnt}, {jsonCnt}") 
                statis_list.append([dcmname, imgCnt, jsonCnt])
            
            statis_list.insert(0, ["caseName","ImgCnt", "AnnoCnt"])
            df = pd.DataFrame(statis_list)
            
            df.to_csv(f"{iitem}.csv", mode='a', header=True, index=False)
# function
#===============================
# start


UsageStr="python statistics_train_data.py  /train/history_train/res_thyNod1.7PurgeNoLabelImgs_mul_nodules1.6_axpG63purge_224_20241208T1229_sz224/ /data/raw_data/thyroidNodules/thyNodu241205/241208-thyNodulesG63P/"
if __name__ == "__main__":            
    if len(sys.argv)<3: 
        print(f"Err: Usage:\n\tApp Train-Result-Folder dataset-folder\n{UsageStr}")
    else:
        trainResFolder=sys.argv[1]
        dataActualFolder=sys.argv[2]
        if os.path.exists(trainResFolder) and os.path.exists(dataActualFolder):
            readImgListFile(sys.argv[1])
        else:
            print(f"Err: please confirm trainRes and dataset exist.")
