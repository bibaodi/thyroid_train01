#!/bin/python3
# statistics the data features about train and test.
# eton@241207

import os

trainResFolderParent=r'/mnt/f/241129-zhipu-thyroid-datas/51-trainResults'
resFolder=r'res_optimizeThyNod1.6_mul_nodules1.5_axpG63_224_20241206T0943_sz224/'



trainDataListfName='imgs_fname_train.txt'
validateDataListfName='imgs_fname_test.txt'

def processImglistLines(imgnamelines:list):
    dcmnameset=set()
    dcmname=None
    for imgline in imgnamelines:
        nameParts=imgline.split(".dcm")
        if type(nameParts) is list and len(nameParts)>1:
            if len(nameParts[0]) > 2:
                dcmname=nameParts[0]
        if type(dcmname) is not None:
            dcmnameset.add(dcmname)
    #print(imgnameline.split(".dcm"))
    sortedList=sorted(dcmnameset)
    #print(f"debug: dcm name set:{type(sortedList)}")
    return sortedList

def readImgListFile():
    iOneResfolder=os.path.join(trainResFolderParent, resFolder)
    print(iOneResfolder)

    dirItems = sorted(os.listdir(iOneResfolder))
    print(dirItems)
    
    for iitem in dirItems:
        if trainDataListfName==iitem:
            imglists=None
            
            iitemPath=os.path.join(iOneResfolder, iitem)
            if not os.path.isfile(iitemPath):
                print(f"Err:file:{iitem} not exist, ignore")
            with open(iitemPath, 'r') as fp:
                imglists=fp.readlines()
            if type(imglists) == None:
                print(f"Err: read from file[{iitem}] failed.!")
                return
            print(f"len of file:[{iitem}]={len(imglists)}, fist img={imglists[0] if len(imglists)>0 else imglists}")
            dcmNameset=processImglistLines(imglists)
            print(f"{iitem} contains[{len(dcmNameset)}]:")
            for dcmname in dcmNameset:
                print(dcmname) 

#
#===============================
#



if __name__ == "__main__":            
    readImgListFile()

