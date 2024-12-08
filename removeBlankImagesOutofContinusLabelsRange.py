#!/bin/python3

#remove the blank images which index < minJsonIndex or index > maxJsonIndex.
# assumption:
# a1. there is not so much blank image(not labeled) in the range [minJsonIdx, maxJsonIdx];
# a2. similart to a1, thers is only one continuous range contains the labled images, implict express that not too many no-labeld images between min,max;
# eton@241208

import os
from pathlib import Path
import re

# List of image file extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
json_extension="*.json"
dataActualFolder=r'/mnt/f/241129-zhipu-thyroid-datas/31-labelmeFormatOrganized/241207_all82AixThyroidNodules'
dcmfolderSuffix=""

def removeOnefile(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been removed successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied: Unable to remove '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def getNumberFromName(filename:str):
    match = re.search(r'frm-(\d+)\.\S+', filename)
    number=-1
    if match:
        number = int(match.group(1))
        #print(f"matched number={number}")  # Output: 3
    return number

def findminmaxJsonIndex(onecasename:Path):
    onedcmfolder=onecasename #+dcmfolderSuffix
    
    dcmpath=onecasename
    if not  onedcmfolder.is_absolute():
        dcmpath=os.path.join(dataActualFolder, onedcmfolder)
    
    #if not os.path.isdir(dcmpath):
    #    print(f"Err:file:{onedcmfolder} not exist, ignore {dcmpath}")
    #    return
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

    if len(image_files) <3 or len(json_files) <2:
        print(f"Err: image or json not enough...")
        return

    jsonSuffix=".json"
    lmstartJsonfile=[ii for ii in json_files if "lmstart.json" in str(ii)]
    if len(lmstartJsonfile)>0:
        firstItem=lmstartJsonfile[0]
        jsonSuffix=firstItem.suffix
        firstname=firstItem.name
        print(f"debug: type={type(firstItem)}, {firstItem}, {firstname}, {type(firstname)}")
        json_files.remove(lmstartJsonfile[0])
    json_files=sorted(json_files)
    print(f"json first:{json_files[0].name}, last:{json_files[-1].name}, jsonSuffix={jsonSuffix} ")
    jsonMinMaxIdxList=[json_files[0].name, json_files[-1].name]
    
    imgSuffix=image_files[0].suffix
    print(f"debug: jsonSuffix={jsonSuffix} , imgSuffix={imgSuffix}")
    imgMinMaxIdxList=[ijson.replace(jsonSuffix, imgSuffix) for ijson in jsonMinMaxIdxList]
    imgMinMaxIdxList=[getNumberFromName(ijson) for ijson in jsonMinMaxIdxList]
    
    print(f"debug: min max index={imgMinMaxIdxList}")
    #print(f"debug: images count:{len(image_files)}, json count{json_files}")

    minimgidx=imgMinMaxIdxList[0]
    maximgidx=imgMinMaxIdxList[1]
    for iidx,iitem in enumerate(sorted(image_files)):
        #print(f"debug: [{iidx}]\t{iitem.name}")
        iimgidx = getNumberFromName(iitem.name)
        print(f"debug: {iimgidx}")
        if iimgidx < minimgidx or iimgidx > maximgidx:
            print(f"delete un-labeled img:{iitem}")
            removeOnefile(iitem)
        
    #print(f"debug: images count:{len(image_files)}, json count{len(json_files)}")
    return (len(image_files), len(json_files))

#dcmfoldername=f"thyroidNodules_axp-001.dcm_frms"
#findminmaxJsonIndex(dcmfoldername)
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
        
def processMultiFolders(working_dir:str):
    casefolders = list_folders(working_dir)
    totalCaseCount=len(casefolders)
    #caseindex=0

    for caseindex, icase in  enumerate(casefolders):
        icasedir=os.path.join(working_dir, icase)
        print(f"[{caseindex}/{totalCaseCount}] : processing... {icase} {type(icasedir)}")
        findminmaxJsonIndex(Path(icasedir))
        #break

working_dir="/mnt/f/241129-zhipu-thyroid-datas/31-labelmeFormatOrganized/241207_all82AixThyroidNodules"
working_dir="/tmp/sss"
processMultiFolders(working_dir)

if __name__ == "__main__":            
    dcmfoldername=f"thyroidNodules_axp-001.dcm_frms"
    findminmaxJsonIndex(dcmfoldername)