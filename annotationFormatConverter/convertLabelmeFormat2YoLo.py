#!/usr/bin/env python
# This script is used to convert the Labelme format to the YOLO format.
# author: eton@250105
# version: 0.1 first version

import datetime
import logging

from tqdm import tqdm
from multimethod import multimethod


logger = logging.getLogger(__name__)

def initLogger():
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string
    formatted_date_time = now.strftime("%y%m%dT%H%M%S")
    # Create the log file name
    log_file_name = f"convertTo_YOLOformart_{formatted_date_time}.log"
    _ver = sys.version_info
    if _ver.minor < 10:
        logger.warning(f"WARNING: this Program develop in Python3.10.12, Current Version May has Problem in `pathlib.Path` to `str` convert.")
        logging.basicConfig(filename=log_file_name,  level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_file_name, encoding='utf-8', level=logging.DEBUG)

###-------------------excel file operation
import pandas 
import re
## !pip install xlrd

class GetInfoFromExternalSpreadSheetFile:
    def __init__(self):
        pass
    
    @staticmethod
    def convert_leading_digits_to_number(tirads_str:str, default_class:int=0):
        if type(tirads_str) is not str or len(tirads_str)<1:
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

        # Check if any digits were collected
        if digits:
            return int(digits)
        else:
            return default_class

    @staticmethod
    def extractAllMatchedFileName(excelFile:str, sheetName:str, colName_select:str, targetKeyToMatch:str, outputColName:str):
        colName_tirads=outputColName #u'ti_rads'
        
        data = {}
        with pandas.ExcelFile(excelFile) as xls:
            data[sheetName] = pandas.read_excel(xls, sheetName, usecols=[colName_tirads, colName_select])
        
        sheet1Dataframe= data[sheetName]
        #print(type(sheet1Dataframe),sheet1Dataframe.shape,sheet1Dataframe[0:5][:], sheet1Dataframe[colName_select][0:5])
        eleNum=sheet1Dataframe.shape[0]

        if type(targetKeyToMatch) is not str:
            targetKeyToMatch=str(targetKeyToMatch)
        
        keyCol = sheet1Dataframe[colName_select]
        valCol = sheet1Dataframe[outputColName]
        # Target value to match in ColumnA
        corresponding_value = '0'
        
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
            logger.warning(f"Warning: No match found for the target value: {target_value}")
        if  type(corresponding_value) is list and  len(corresponding_value)>0:
            return corresponding_value[0]
        else:
            return '0'

    def testIt(self):
        exlfile=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/dataHasTIRADS_250105.xls'
        sheetName="origintable"
        selectColName='access_no'
        matchKey='02.202401010411.01'
        outputColName=u'ti_rads'

        matchedTRs=GetInfoFromExternalSpreadSheetFile.extractAllMatchedFileName(exlfile,sheetName, selectColName, matchKey, outputColName)
        matchedTRi=GetInfoFromExternalSpreadSheetFile.convert_leading_digits_to_number(matchedTRs)
        logger.info(f"matchedTR={matchedTRs}, {matchedTRi}")

###--------excel file operation.end
import sys
import json
import os
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
                logger.info(f"Image size: {width}x{height}")
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
    def __init__(self, outputYoloPath:pathlib.Path , exlfile:pathlib.Path, sheetName:str, selectColName:str, outputColName:str):
        self.outputYoloPath=outputYoloPath
        self.exlfile=exlfile
        self.sheetName=sheetName
        self.selectColName=selectColName
        self.outputColName=outputColName
        
    @staticmethod
    def rectFromPixelToYoloFormat_CenterXYWH_inPercent(rectInPos:list, imgWidth:int, imgHeight:int):
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
                ipath.mkdir(mode=0o666, parents=True, exist_ok=True)
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
            logger.info(f"Err:Permission denied: Cannot copy to {newImagePath}")
        except Exception as e:
            logger.info(f"Err: An error occurred: {e}")

        txtname=filenameStemInYolo.replace(imgfileSuffix, '.txt')
        newLabelfilePath=outputYoloPath_txt.joinpath(txtname)
        with open(newLabelfilePath, 'w') as txtfp:
            for oneRect in inLabelsList:
                txtfp.write(' '.join(map(str, oneRect))+ '\n')
        logger.info(f"debug: end of file create {newImagePath}")
        return 0

    def parseXiaobaoJson(self, json_file:pathlib.Path, leastPointCount:int=4):
        """
            1. deal with one xiaobai's json;
            2. get all shape, create with rectangle;
            3. save rectangles in YOLO format;
            4. all files in one folder, no sub-folder is supported.
            eton@250104
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
        if not jsonbasename.startswith("frm-"):
            logger.error(f"\tfile prefix not match:[{jsonbasename}]")
            return -2
        imagefolder=json_file.parent
        caseNameinLblme=imagefolder.name
        image_file=json_file.with_suffix(".png")

        lbm_shapes=None
        # 01-read from nodule-json-file
        with open(json_file, 'r') as f:
            lbm_json = json.load(f)
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
            #01 polygon to rectangle;
            shape_rect = LabelmeFormat2YOLOFormat.polygonToRectangle(lbm_pointsInOneShape)
            #02 show in image
            debug_this=False
            if type(debug_this) is not None and debug_this:
                ImageOperation.showRectInImg(image_file, shape_rect, lbm_pointsInOneShape)
            #03 rectangle to YOLO
            imgW, imgH = ImageOperation.getImageSizeWithoutReadWholeContents(image_file)
            shape_yolorect=LabelmeFormat2YOLOFormat.rectFromPixelToYoloFormat_CenterXYWH_inPercent(shape_rect, imgW, imgH)
            #03.2-add class to label
            matchKey = PacsCaseName_LabelmeCaseName_mapper.mapNameInLabelmeFmtToOriginAAccessionNum(caseNameinLblme)

            outputYoloFolder=self.outputYoloPath
            exlfile=self.exlfile
            selectColName=self.selectColName
            outputColName=self.outputColName
            sheetName=self.sheetName

            matchedTRs=GetInfoFromExternalSpreadSheetFile.extractAllMatchedFileName(exlfile,sheetName, selectColName, matchKey, outputColName)
            matchedTRi=GetInfoFromExternalSpreadSheetFile.convert_leading_digits_to_number(matchedTRs)
            shape_yolorect.insert(0, matchedTRi)
            
            allRects.append(shape_yolorect)
            #print(f"debug: shape_rect={shape_rect}, shape_yolorect={shape_yolorect}")

            #04 save image and Yolo.txt to new folder
        if len(allRects)>0:
            LabelmeFormat2YOLOFormat.saveImageAndLabelsToTargetPath(outputYoloFolder, image_file, allRects)
                    
        return 0

    def processOnePACSfolder(self, casepath:pathlib.Path):
        if type(casepath) is str:
            casepath=pathlib.Path(casepath)
        if  not casepath.is_dir():
            logger.info(f"not exist dir:{casepath}")
            return -1
            
        labelmefolderpath=casepath
        
        imgs=[iimg for iimg in sorted(casepath.glob('*.{jpg,png,jpeg,bmp}'))]
        jsons=[ijson for ijson in sorted(casepath.glob("*.json"))]

        if len(jsons) < 1:
            logger.info("Err: json file not found in casefolder:{casepath}")
            return -1
        
        for ijsonpath in jsons:
            ret  = self.parseXiaobaoJson(ijsonpath)
            logger.info(f"debug: ret={ret}")
            if ret <0:
                logger.info(f"Err: parse json failed")
            
        return 0

    def process_multiPACScases(self, casesFolder):
        working_dir=pathlib.Path(casesFolder)
        casefolders = working_dir.iterdir()

        for icase in tqdm(casefolders, desc="PACS Format Converting2YOLO:"):
            icasepath=icase
            caseName=icasepath.name
            if caseName.startswith("YOLO"):
                continue
            logger.info(f"\n\n^^^Process:{icasepath}")
            failed = self.processOnePACSfolder(icasepath)

            if 0 != failed:
                logger.info(f"\tprocess pacs folder failed!!!")
                break
            else:
                logger.info(" process pacs folder success,,,")



def main_entrance():
    initLogger()
    if len(sys.argv)<2:
        print(f"App ImageFolder")
    else:
        imgfolder=pathlib.Path(sys.argv[1])
        if False == imgfolder.is_dir():
            print(f"Error: please confirm folder exist[{str(imgfolder)}]!!!")
            return -1
        logger.info(f"\n\nProcessing:{imgfolder}...")

        exlfile=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/dataHasTIRADS_250105.xls'
        selectColName='access_no'
        outputColName=u'ti_rads'
        sheetName="origintable"
        outputYoloPath=imgfolder.with_suffix('.yolofmt')
        fmtConverter=LabelmeFormat2YOLOFormat(outputYoloPath, exlfile, sheetName,selectColName, outputColName)
        fmtConverter.process_multiPACScases( imgfolder)

def test_it():
    rootFolder=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat'
    oneCasefolder=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/301PACS02-2401020402.01'

    outputYoloFolder=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/yoloDataset01'
    exlfile=r'/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/forObjectDetect_PACSDataInLabelmeFormatConvert2YoloFormat/dataHasTIRADS_250105.xls'
    selectColName='access_no'
    outputColName=u'ti_rads'
    sheetName="origintable"

    outputYoloPath=imgfolder.with_suffix('.yolofmt')
    fmtConverter=LabelmeFormat2YOLOFormat(outputYoloPath, exlfile, sheetName,selectColName, outputColName)
    fmtConverter.process_multiPACScases( imgfolder)


if __name__ == "__main__":
    _ver = sys.version_info
    if _ver.minor < 10:
        print(f"WARNING: this Program develop in Python3.10.12, Current Version May has Problem in `pathlib.Path` to `str` convert.")
    main_entrance()
    print(f"Done.")