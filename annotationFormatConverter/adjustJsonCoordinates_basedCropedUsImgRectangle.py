import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import pathlib

class AnnotationJsonAdjustor:
    def __init__(self, casedir:pathlib.Path):
        self.m_casename = casedir
        self.m_usimgRect=[]#[topleft, rightbottom]

    def getCropInfoFromLmstartJson(self, var:str=""):
        imgPath=pathlib.Path(self.m_casename)
        lmstartBaseName=r'lmstart.json'
        lmstartFile=imgPath.joinpath(lmstartBaseName)

        if not lmstartFile.exists():
            print(f"Error: json file Not Exist:{lmstartFile}")
            return -1
        with open(lmstartFile, 'r') as f:
            jims=json.load(f)

        rectinfo=jims["USImgRectLRTD"]
        if type(rectinfo) is list and 2 ==len(rectinfo):
            print(f"debug: case [{self.m_casename}] usRect={rectinfo}")
            self.m_usimgRect=rectinfo
            return 0
        return -2

    def processOneJson(self, jsonfile:pathlib.Path):
        if not jsonfile.exists():
            print(f"Error: json file Not Exist:{jsonfile}")
            return -1
        with open(jsonfile, 'r') as f:
            jims=json.load(f)

        jsonShapes=jims['shapes']
        if (type(jsonShapes) is not list) or (len(jsonShapes)<1):
            print(f"Error:no valid shapes in json:{jsonfile.stem}")
            return -1

        for idx, ashape in enumerate(jsonShapes):
            shapepts=ashape["points"]
            if (type(shapepts) is not list) or (len(shapepts)<3):
                print(f"Warning:shape[{idx}] in json Error")
                continue
            hasOnePtReported=False
            for ptIdx, apoint in enumerate(shapepts):
                if (type(apoint) is not list) or (len(apoint)!=2):
                    print(f"Warning:point[{ptIdx}] in shape[{idx}] Error")
                    continue
                ptX=apoint[0]
                ptY=apoint[1]
                
                topleft, rightBottom = self.m_usimgRect
                xnewLeft, ynewTop=topleft
                x_end, y_end = rightBottom
                
                if (False == hasOnePtReported):
                    if (ptY > y_end) or (ptY < ynewTop) or (ptX < xnewLeft) or (ptX > x_end):
                        print(f"Warning: {jsonfile.stem} [{apoint}]annotation OverFlow US-Image Region[{self.m_usimgRect}].")
                        hasOnePtReported=True

                apoint[0]-=xnewLeft
                apoint[1]-=ynewTop
                apoint[0]=max(apoint[0], 1)
                apoint[1]=max(apoint[1], 1)
        #write back
        with open(jsonfile, 'w') as jfp:
                json.dump(jims, jfp)
        return 0
                

    
def processJsonInCases(casesFolder):
    working_dir=pathlib.Path(casesFolder)
    casefolders = working_dir.iterdir()
    for icase in casefolders:
        icasedir=os.path.join(working_dir, icase)
        
        dcmfolder=pathlib.Path(icasedir)
        print(f"\n\nProcess:{dcmfolder}")
        annotJsonAdj=AnnotationJsonAdjustor(dcmfolder)
        failed =annotJsonAdj.getCropInfoFromLmstartJson("")

        if 0 != failed:
            print(f"\tread Rect Info failed!!!")
        else:
            print(" read Rect Info success,,,")

        allAnnotedJsons=sorted(dcmfolder.glob('frm-*.json'))
        for oneJson in allAnnotedJsons:
            annotJsonAdj.processOneJson(oneJson)


if __name__ == "__main__":
    if len(sys.argv)<2:
        print(f"App Image")
    else:
        #fp = '/media/eton/hdd931g/42-workspace4debian/10-ExtSrcs/ITKPOCUS/itkpocus/tests/data/83CasesFirstImg/thyroidNodules_axp-042_frm-0001.png'
        alldcmfolder=sys.argv[1]
        print(f"\n\nProcess:{alldcmfolder}")
        processJsonInCases(alldcmfolder)

"""
python /mnt/d/000-srcs/210822-thyroid_train/annotationFormatConverter/adjustJsonCoordinates_basedCropedUsImgRectangle.py `pwd`

for i in `find ../testAddCropInfo_4cases -name "lmstart.json"` ; 
    do n=`sed 's/\.\.\/testAddCropInfo_4cases\///g' <<< $i ` ; cmd="cp $i $n"; echo $cmd && eval $cmd;
done

for i in `find ../241208-incremental4cases -name "frm-*.json"` ; 
    do n=`sed 's/\.\.\/241208-incremental4cases\///g' <<< $i ` ; cmd="cp $i $n"; echo $cmd && eval $cmd; 
done
"""