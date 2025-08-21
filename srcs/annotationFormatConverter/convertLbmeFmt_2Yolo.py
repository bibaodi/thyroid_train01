#!/usr/bin/env python
# This script is used to convert the Labelme format to the YOLO format.
# author: eton@250105
# version: 0.1 first version
# eton@250311 version:0.2, support no spreadsheet file.
# eton@250314 version:0.3, support both detect and segment task data format. import log from glog.
# eton@250820 version:0.4, support prefixStartWithFrm as option and refine usage;
# yolo-segment: <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn> ; range is (0,1)
# yolo-detect: <class-index> <x> <y> <width> <height> ; range is (0,1)

import typing
import sys
import json
import os
import datetime
import logging
import time
from tqdm import tqdm
from multimethod import multimethod

# Add the utils directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog
import LabelmeJson
from TaskTypeDefine import TaskTypes as TaskType
logger = glog.glogger

###-------------------excel file operation
import pandas 
import re
## !pip install xlrd

class GetInfoFromExternalSpreadSheetFile:
    static_member4NotMatched=int(9527)

    def __init__(self,excelFile:str, sheetName:str,colName_select:str, outputColName:str):
        self.excelFile=excelFile
        self.sheetName=sheetName
        self.sheet1Dataframe=None

        colName_tirads=outputColName #u'ti_rads' 
        data = {}
        with pandas.ExcelFile(excelFile) as xls:
            data[sheetName] = pandas.read_excel(xls, sheetName, usecols=[colName_tirads, colName_select])
        
        self.sheet1Dataframe= data[sheetName]

    def isMatchedValueVaild(self, matchedValue:int):
        if  not isinstance(matchedValue, (int, float)):
            return False
        if self.static_member4NotMatched == matchedValue:
            return False
        return True

    @staticmethod
    def mapBethesda06ToBengNMalign(inclass:int=0):
        if inclass <=1: #NULL, 0,1->0
            return 0
        elif inclass<=2:# 2->Benign
            return 1
        else:#3-6->Malignant
            return 2
    
    @staticmethod
    def convert_leading_digits_to_number(tirads_str:str, default_class:int=0):
        if type(tirads_str) is not str or len(tirads_str)<1:
            logger.error(f"Err: input tirads_str is not string or empty:{tirads_str}")
            return default_class
            #raise ValueError("Input string is empty")

        # Initialize an empty string to collect digits
        digits = ""

        # Iterate through the characters in the string
        for char in tirads_str:
            if char.isdigit():
                digits += char
            else:
                break
        logger.info(f"debug: tirads_str={tirads_str}, digits={digits}")
        # Check if any digits were collected
        if digits:
            return int(digits)
        else:
            return default_class

    def extractAllMatchedFileName(self, excelFile:str, sheetName:str, colName_select:str, targetKeyToMatch:str, outputColName:str):
        colName_tirads=outputColName #u'ti_rads' 
        sheet1Dataframe= self.sheet1Dataframe#data[sheetName]
        #print(type(sheet1Dataframe),sheet1Dataframe.shape,sheet1Dataframe[0:5][:], sheet1Dataframe[colName_select][0:5])

        if type(targetKeyToMatch) is not str:
            targetKeyToMatch=str(targetKeyToMatch)
        
        keyCol = sheet1Dataframe[colName_select]
        valCol = sheet1Dataframe[outputColName]
        # Target value to match in ColumnA
        corresponding_value = self.static_member4NotMatched
        
        # Find the row where ColumnA matches the target value
        if 0: # match by equal
            matching_row = sheet1Dataframe[keyCol == targetKeyToMatch]
        matching_row = sheet1Dataframe[keyCol.str.contains( targetKeyToMatch)]
        #print(f"debug: matching_row: {type(matching_row)}, {matching_row.shape}, {matching_row}, {type(matching_row[outputColName])} <<<<<")
        #print(f"debug: matching_row: {type(matching_row[outputColName])}, {matching_row[outputColName].shape}, {matching_row[outputColName].index} <<<<<")

        # Get the corresponding value in ColumnB
        if not matching_row.empty:
            corresponding_value =matching_row[outputColName]# matching_row.at[outputColName]#sheet1Dataframe.get(targetKeyToMatch, 9990)#
            corresponding_value = corresponding_value.to_list()
            logger.info(f"debug: The [{targetKeyToMatch}] corresponding value in {outputColName} is: {corresponding_value}")
        else:
            logger.warning(f"Warning: No match found for the target value: {targetKeyToMatch}")
        
        if  type(corresponding_value) is list and  len(corresponding_value)>0:
            if type(corresponding_value[0]) is not str:
                return str(corresponding_value[0])
            else:
                return corresponding_value[0]
        else:
            return str(self.static_member4NotMatched)
    
    @staticmethod
    def testIt():
        exlfile=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/dataHasTIRADS_250105.xls'
        sheetName="origintable"
        selectColName='access_no'
        matchKey='02.202401010411.01'
        outputColName=u'ti_rads'

        spreadsheetReader = GetInfoFromExternalSpreadSheetFile(excelFile, sheetName, selectColName, outputColName)
        matchedTRs=spreadsheetReader.extractAllMatchedFileName(exlfile,sheetName, selectColName, matchKey, outputColName)
        matchedTRi=GetInfoFromExternalSpreadSheetFile.convert_leading_digits_to_number(matchedTRs)
        logger.info(f"matchedTR={matchedTRs}, {matchedTRi}")

###--------excel file operation.end
import pathlib
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt


class PacsCaseName_LabelmeCaseName_mapper:
    self_sprefix='301PACS'
    self_smarker='-'
    
    @staticmethod
    def getPath4AfterConverted(casepath:pathlib.Path):
        pathParent = casepath.parent
        caseName=casepath.name
        substring=".20" ##02.202410281498.01 -> 301PACS02-2410281498.01
        if substring in caseName:
            caseName=caseName.replace(substring, "-")
    
        substring=".000000" ##22.0000001629044 -> 301PACS22-1629044
        if substring in caseName:
            caseName=caseName.replace(substring, "-")
        caseName=f"301PACS{caseName}"
        labelmeFmtfoler=pathParent.joinpath(caseName)
    
        return labelmeFmtfoler
    
    @staticmethod
    def mapNameInLabelmeFmtToOriginAAccessionNum(casenameInlblme:str):
        """
        remove the prefix, keep the part which will be matched in origin accession number
        """
        accession_no=''
        
        parts=casenameInlblme.split(PacsCaseName_LabelmeCaseName_mapper.self_smarker)
        
        if len(parts)>1:
            accession_no=parts[-1]
    
        return accession_no
    

from PIL import Image, UnidentifiedImageError

class ImageOperation:
    @staticmethod
    def getImageSizeWithoutReadWholeContents(image_path:pathlib.Path):
        imgsize=(0,0)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                logger.info(f"{image_path} Image size: {width}x{height}")
                imgsize=(width, height)
                return imgsize
        except UnidentifiedImageError:
            logger.error("Failed to identify the image file.")
        except IOError:
            logger.error("Failed to open the image file.")

        return imgsize

    @staticmethod
    def showRectInImg(image_file, shape_rect, lbm_pointsInOneShape):
        img=cv2.imread(image_file)
        rectColor=(0,0,255)#GRB
        rectLineWeight=3
        pttopleft=tuple(shape_rect[0:2])
        ptBR=tuple(shape_rect[2:4])
        cv2.rectangle(img,pttopleft,ptBR,rectColor,rectLineWeight)

        polygonPts=np.array([lbm_pointsInOneShape], np.int32)#list of list of points
        polygonPts4cv=polygonPts.reshape((-1, 1, 2))

        polygonColor=(255,0,0)
        cv2.polylines(img, [polygonPts4cv], isClosed=True, color=polygonColor, thickness=1)
        plt.imshow(img)

class LabelmeFormat2YOLOFormat:
    static_taskType_detect=TaskType.Detect
    static_taskType_segment=TaskType.Segment
    static_taskType_unsupport=TaskType.UNKNOWN
    def __init__(self, taskType:TaskType , exlfile:pathlib.Path, sheetName:str, selectColName:str, outputColName:str):
        self.outputYoloPath=None
        self.taskType = taskType
        self.exlfile=exlfile if isinstance(exlfile, pathlib.Path) else pathlib.Path(exlfile)
        self.sheetName=sheetName
        self.selectColName=selectColName
        self.outputColName=outputColName
        if self.exlfile.is_file():
            self.spreadSheetReader = GetInfoFromExternalSpreadSheetFile(exlfile, sheetName, selectColName, outputColName)
        else:
            logger.error(f"Err: exlfile not exist:{exlfile}, make reader as None.")
            self.spreadSheetReader = None

    def isDetectTask(self):
        return self.taskType == self.static_taskType_detect
    
    def isSegmentTask(self):
        return self.taskType == self.static_taskType_segment

    @staticmethod
    def rectFromPixelToYoloFormat_CenterXYWH_inPercent(rectInPos:list, imgWidth:int, imgHeight:int):
        """
        input a rectangle in list of points(x,y);
        output:
            1. a rectangle in list of points(topleft.xp, toplet.yp, w.xp, h.yp)
            2. Empty list if failed
        """
        yoloRect=[]
        if type(rectInPos) is not list or len(rectInPos) !=4:
            logger.error(f"Err: input rect Err:{rectInPos}")
            return yoloRect
        for coord in rectInPos:
            if coord > imgWidth and coord > imgHeight:
                logger.error(f"Err: input rect coordinate not in range:{rectInPos}, {imgWidth}, {imgHeight}")
                return yoloRect

        rectLeft, rectTop, rectRight, rectBottom=rectInPos

        rectW=rectRight - rectLeft
        rectH=rectBottom - rectTop

        rectWp = rectW / imgWidth
        rectHp = rectH / imgHeight
        
        rectCx=(rectRight + rectLeft)/2
        rectCy=(rectBottom + rectTop)/2

        rectCxp=rectCx/imgWidth
        rectCyp=rectCy/imgHeight

        yoloRect=[rectCxp, rectCyp, rectWp, rectHp]
        return yoloRect
    
    @staticmethod
    def contourPointsPixelToYoloFormat_inPercent(contourInPos:list, imgWidth:int, imgHeight:int):
        """
        input a polygon in list of  points(x,y) [[x,y], [x,y],...];
        output: list of points from pixel to percentage [x1,y1, x2,y2,...];
        """
        yoloPercentList=[]
        if type(contourInPos) is not list or len(contourInPos) <3:
            logger.error(f"Err: input not enough 3 pt Err:{contourInPos}")
            return yoloPercentList
        for coord in contourInPos:
            if coord[0] > imgWidth or coord[1] > imgHeight:
                logger.error(f"Err: input rect coordinate not in range:{contourInPos}, {imgWidth}, {imgHeight}")
                return yoloPercentList

        for pt in contourInPos:
            Xp = pt[0] / imgWidth #x
            Yp = pt[1] / imgHeight #y
            yoloPercentList.append(Xp)
            yoloPercentList.append(Yp)
        return yoloPercentList

    @staticmethod
    def polygonToRectangle(polygonPts:list):
        """
        input a polygon in list of  points(x,y);
        output:
            1. a rectangle in list of points(topleft.x, toplet.y, bottomright.x, bottomright.y)
            2. Empty list if failed
        eton@250104
        """
        rectangle=[]
        if len(polygonPts)<2:
            return rectangle
        pts = np.array(polygonPts, dtype=np.int32)
        Xs=pts[:, 0]
        Ys=pts[:, 1]
        #print(f"debug: pts in npX:{Xs},Ys={Ys} ")
        x_min=np.min(Xs)
        y_min=np.min(Ys)

        x_max=np.max(Xs)
        y_max=np.max(Ys)

        rectangle=[x_min, y_min,x_max,  y_max]
        #print(f"debug: pts in rectangle:{rectangle} ")
        return rectangle

    
    @staticmethod
    def saveImageAndLabelsToTargetPath(outputYoloPath:pathlib.Path, inImgPath:pathlib.Path, inLabelsList:list):
        if type(outputYoloPath) is str:
            outputYoloPath=pathlib.Path(outputYoloPath)

        outputImgStem=r'images'
        outputLabelStem=r'labels'
        outputYoloPath_img=outputYoloPath.joinpath(outputImgStem)
        outputYoloPath_txt=outputYoloPath.joinpath(outputLabelStem)
        for ipath in [outputYoloPath_img, outputYoloPath_txt]:
            if not ipath.is_dir():
                ipath.mkdir(mode=0o766, parents=True, exist_ok=True)
                print(f"debug: create not exist folder:[{ipath}]")

        if not inImgPath.is_file():
            print(f"\timage not exist.[{inImgPath}]")
            return -1

        imgfileSuffix=inImgPath.suffix
        imgfileStem=inImgPath.name
        caseStem=inImgPath.parent.name
        
        filenameStemInYolo=caseStem+'_'+imgfileStem
        newImagePath=outputYoloPath_img.joinpath(filenameStemInYolo)

        try:
            shutil.copyfile(inImgPath, newImagePath)
            logger.info(f"debug: File copied from {inImgPath} to {newImagePath} with metadata")
        except FileNotFoundError:
            logger.info(f"Err:Source file not found: {inImgPath}")
        except PermissionError:
            logger.info(f"Err:Permission denied: Cannot copy [{inImgPath}] to {newImagePath}")
        except Exception as e:
            logger.info(f"Err: An error occurred: {e}")

        txtname=filenameStemInYolo.replace(imgfileSuffix, '.txt')
        newLabelfilePath=outputYoloPath_txt.joinpath(txtname)
        with open(newLabelfilePath, 'w') as txtfp:
            for oneRect in inLabelsList:
                txtfp.write(' '.join(map(str, oneRect))+ '\n')
        logger.info(f"debug: end of file create {newImagePath}")
        return 0

    def parselabelmeJson(self, json_file:pathlib.Path, leastPointCount:int=4, **kwargs):
        """
            1. deal with one xiaobai's json;
            2. get all shape, create with rectangle;
            3. save rectangles in YOLO format;
            4. all files in one folder, no sub-folder is supported.
            eton@250104
            support detect and segment task. eton@250314;
        """
        if type(json_file) is str:
            json_file=pathlib.Path(json_file)
            
        if not json_file.is_absolute():
            logger.error(f"\tjson target not absolute path:[{json_file}]")
            return -3
        if not json_file.is_file():
            logger.error(f"\tjson target not exist.[{json_file}]")
            return -1
        
        jsonbasename=json_file.name
        
        prefixStartWithFrm=True
        for key, value in kwargs.items():
            if key == 'prefixStartWithFrm':
                prefixStartWithFrm = value
                break
        if prefixStartWithFrm and not jsonbasename.startswith("frm-"):
            logger.error(f"\tfile prefix not match:[{jsonbasename}]")
            return -2
        imagefolder=json_file.parent
        caseNameinLblme=imagefolder.name
        image_file=json_file.with_suffix(".png")

        lbm_shapes=None
        # 01-read from nodule-json-file
        with open(json_file, 'r') as f:
            lbm_json = json.load(f)
            if 'imagePath' not in lbm_json:
                logger.error(f"Err: imagePath not in json file.[{json_file}](maybe lmstart)")
                return -5
            imagepath=lbm_json['imagePath']
            image_file = imagefolder.joinpath(imagepath)
            if not image_file.is_file():
                logger.error(f"Err: image file in json not found.[{image_file}]")
                return -4
            
            lbm_shapes=lbm_json['shapes']
        # process shapes in json
        allRects=[]
        if lbm_shapes is None:
            return 2005

        for ishape, lbm_shapeItem in enumerate(lbm_shapes):
            lbm_pointsInOneShape=lbm_shapeItem["points"]
            lbm_classInOneShape=lbm_shapeItem["label"]
            pointCntInShape=len(lbm_pointsInOneShape)
            #print(f"\t{json_file} shape[{ishape}] has point in shape is: {pointCntInShape}.")
            if pointCntInShape < leastPointCount:
                logger.error(f"\tErr:{json_file}_shape[{ishape}] has point in shape less then {leastPointCount}>{pointCntInShape}.")
                return -1
            #print(f"polygon[{lbm_classInOneShape}] is:{lbm_pointsInOneShape}")
            if self.isDetectTask():
                #01 polygon to rectangle;
                yoloObjShape = LabelmeFormat2YOLOFormat.polygonToRectangle(lbm_pointsInOneShape)
            elif self.isSegmentTask():
                yoloObjShape=lbm_pointsInOneShape
            #02 show in image
            debug_this=False
            if type(debug_this) is not None and debug_this:
                if self.isDetectTask():
                    ImageOperation.showRectInImg(image_file, yoloObjShape, lbm_pointsInOneShape) 
            
            #03 rectangle to YOLO
            image_op_start = time.time()
            imgW, imgH = ImageOperation.getImageSizeWithoutReadWholeContents(image_file)
            image_op_end = time.time()
            logging.info(f"performance: Image operation took {image_op_end - image_op_start:.4f} seconds")
            if self.isDetectTask():
                yoloObjShape1d=LabelmeFormat2YOLOFormat.rectFromPixelToYoloFormat_CenterXYWH_inPercent(yoloObjShape, imgW, imgH)
            else:
                yoloObjShape1d=LabelmeFormat2YOLOFormat.contourPointsPixelToYoloFormat_inPercent(yoloObjShape, imgW, imgH)
            #03.2-add class to label
            noSpreadSheet=False
            for key, value in kwargs.items():
                if key == 'noSpreadSheet':
                    noSpreadSheet = value
                    break
            if False == noSpreadSheet:
                matchKey = PacsCaseName_LabelmeCaseName_mapper.mapNameInLabelmeFmtToOriginAAccessionNum(caseNameinLblme)
                exlfile=self.exlfile
                selectColName=self.selectColName
                outputColName=self.outputColName
                sheetName=self.sheetName

                spreadsheet_op_start = time.time()
                spreadSheetReader = self.spreadSheetReader #GetInfoFromExternalSpreadSheetFile(exlfile, sheetName, selectColName, outputColName)
                matchedTRs=spreadSheetReader.extractAllMatchedFileName(exlfile,sheetName, selectColName, matchKey, outputColName)
                spreadsheet_op_end = time.time()
                logging.info(f"performance: Spreadsheet operation took {spreadsheet_op_end - spreadsheet_op_start:.4f} seconds")
                matchedTRi=GetInfoFromExternalSpreadSheetFile.convert_leading_digits_to_number(matchedTRs)
                #if not spreadSheetReader.isMatchedValueVaild(matchedTRi) or matchedTRi <1:
                #    logger.error(f"Err: Value Not Vaild, matchedTRs={matchedTRs},remove TI0")
                #    continue
                BenignMalign3Class= matchedTRi #GetInfoFromExternalSpreadSheetFile.mapBethesda06ToBengNMalign(matchedTRi)
                logger.info(f"matchedTRs={matchedTRs}, matchedTRi={matchedTRi}, BenignMalign3Class={BenignMalign3Class}") 
            else:
                BenignMalign3Class=0
                if '1malign' in str(imagefolder).lower():
                    BenignMalign3Class=1
                elif '0benign' in str(imagefolder).lower():
                    BenignMalign3Class=0
                else:
                    logger.error(f"Err: No matched folder name for BenignMalign3Class:{imagefolder}")
                logger.info(f"BenignMalign3Class={BenignMalign3Class}") 
            yoloObjShape1d.insert(0, BenignMalign3Class)
            
            allRects.append(yoloObjShape1d)
            #print(f"debug: shape_rect={shape_rect}, shape_yolorect={shape_yolorect}")

            #04 save image and Yolo.txt to new folder
        if len(allRects)>0:
            outputYoloFolder=self.outputYoloPath
            LabelmeFormat2YOLOFormat.saveImageAndLabelsToTargetPath(outputYoloFolder, image_file, allRects)
                    
        return 0

    def processOnePACSfolder(self, casepath:pathlib.Path):
        if type(casepath) is str:
            casepath=pathlib.Path(casepath)
        if  not casepath.is_dir():
            logger.info(f"not exist dir:{casepath}")
            return -1
            
        labelmefolderpath=casepath
        
        imgs=[iimg for iimg in sorted(labelmefolderpath.glob('*.{jpg,png,jpeg,bmp}'))]
        jsons=[ijson for ijson in sorted(labelmefolderpath.glob("*.json"))]

        if len(jsons) < 1:
            logger.info("Err: json file not found in casefolder:{labelmefolderpath}")
            return -1
        
        for ijsonpath in tqdm(jsons, desc="processing Jsons:"):
            ret  = self.parselabelmeJson(ijsonpath.absolute(), noSpreadSheet= True, prefixStartWithFrm=False)
            logger.info(f"debug: process [{ijsonpath}], ret={ret}")
            if ret <0:
                logger.info(f"Err: parse json failed")
            
        return 0

    def process_multiPACScases(self, casesFolder:typing.Union[str|pathlib.Path]):
        if type(casesFolder) is str:
            casesFolder=pathlib.Path(casesFolder)
        working_dir=casesFolder

        if self.isSegmentTask():
            outputYoloPath=working_dir.with_suffix('.ylSegFmt')
        elif self.isDetectTask():
            outputYoloPath=working_dir.with_suffix('.ylDetectFmt')
        else:
            logger.error(f"Err: not support task type:{self.taskType}")
            return -1
        self.outputYoloPath=outputYoloPath

        casefolders = working_dir.iterdir()

        for icase in tqdm(casefolders, desc="PACS LabelmeFormat Converting2YOLO:"):
            icasepath=icase
            caseName=icasepath.name
            if icasepath.is_file():
                continue
            if caseName.startswith("YOLO"):
                continue
            logger.info(f"^^^Process:{icasepath}")
            failed = self.processOnePACSfolder(icasepath)

            if 0 != failed:
                logger.error(f"process pacs folder failed!!!")
                break
            else:
                logger.info("process pacs folder success,,,")


def main_entrance():
    if len(sys.argv)<3:
        print(f"App ImageFolder taskType(segment|detect) spreadsheetFile.xls(optional)")
        return 
    
    glog.glogger = glog.initLogger("gen301PX_yoloDS_fromLbmefmt")
    global logger
    logger = glog.glogger
    imgfolder=pathlib.Path(sys.argv[1])
    if False == imgfolder.is_dir():
        print(f"Error: please confirm folder exist[{str(imgfolder)}]!!!")
        return -1
    glog.glogger.info(f"Processing:{imgfolder}...")

    taskType=sys.argv[2]
    if taskType == "segment":
        taskType=LabelmeFormat2YOLOFormat.static_taskType_segment
    elif taskType == "detect":
        taskType=LabelmeFormat2YOLOFormat.static_taskType_detect
    
    exlfile= sys.argv[3] #r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/dataHasTIRADS_250105.xls'
    selectColName='access_no'
    outputColName=u'bom' #'ti_rads'#u'bom' 
    sheetName="origintable"
    
    fmtConverter=LabelmeFormat2YOLOFormat(taskType, exlfile, sheetName,selectColName, outputColName)
    fmtConverter.process_multiPACScases(imgfolder)

def test_it():
    rootFolder=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat'
    oneCasefolder=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/301PACS02-2401020402.01'

    outputYoloFolder=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/yoloDataset01'
    exlfile=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/dataHasTIRADS_250105.xls'
    selectColName='access_no'
    outputColName=u'ti_rads'
    sheetName="origintable"

    #outputYoloPath=imgfolder.with_suffix('.yolofmt')
    fmtConverter=LabelmeFormat2YOLOFormat(LabelmeFormat2YOLOFormat.static_taskType_detect, exlfile, sheetName,selectColName, outputColName)
    fmtConverter.process_multiPACScases( rootFolder)


if __name__ == "__main__":
    _ver = sys.version_info
    if _ver.minor < 10:
        print(f"WARNING: this Program develop in Python3.10.12, Current Version May has Problem in `pathlib.Path` to `str` convert.")
    main_entrance()
    print(f"Done.")

#250112 run $ python annotationFormatConverter/convertLabelmeFormat2YoLo.py /mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat /mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/301PACS_database_v241229.xlsx
#250113 run $ python annotationFormatConverter/convertLabelmeFormat2YoLo.py \
# /mnt/f/241129-zhipu-thyroid-datas/17--labelmeFormatOrganized/301pacsDataInLbmfmtRangeY22-24 \
# /mnt/f/241129-zhipu-thyroid-datas/17--labelmeFormatOrganized/301pacsDataInLbmfmtRangeY22-24/301PACS_database_RangeY22-24_V250113.xlsx
