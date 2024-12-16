import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import pathlib
from pathlib import Path 

"""
- eton@241216 finish get crop rectangle and write to lmstart.json.

"""


def removeAllNonGrayscalePixels(img:np.ndarray):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where the grayscale image is equal to the original image
    mask = np.all(img == gray_image[:, :, np.newaxis], axis=-1)
    
    # Set non-grayscale pixels to zero
    img[~mask] = 0
    #plt.imshow(img)
    return img

def tryRmAllNonGrayscalePixels(imgBgr:np.ndarray):
    """
    if the us image is not all grayscale, then ignore this, otherwise all image will be black/blank;
    """
    rowCntC=int(imgBgr.shape[0]/2)
    colCntC=int(imgBgr.shape[1]/2)
    
    center9pixels=imgBgr[(rowCntC-1): (rowCntC+2), (colCntC-1):(colCntC+2),:]
    pixelsNum=(center9pixels.shape[0] * center9pixels.shape[1])
    #print(f"debug: center9pixels={center9pixels}")
    center9pixels=center9pixels.reshape(pixelsNum, 3)
    #print(f"debug: center9pixels={center9pixels}")
    
    equalItems=0
    shouldEqualCnt=center9pixels.shape[0]
    for irgb in range(shouldEqualCnt):
        if len(np.unique(center9pixels[irgb,:])) == 1:
            equalItems+=1
    if (shouldEqualCnt - equalItems) / shouldEqualCnt > 0.2:
        print(f"WARNING: pixel not grayscale , cannot remove color pixels.!!!")
        img_removedColors=imgBgr
    else:
        img_removedColors=removeAllNonGrayscalePixels(imgBgr)
    return img_removedColors

def getOpeningImg(gray_image:np.ndarray):
    #gray_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    kernel=np.ones((5,5), np.uint8)
    openingImg = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    return openingImg

class CropUsImageClass:
    def __init__(self, imgfile:str):
        self.m_imgname = imgfile
        self.m_percentage=0.3
        self.m_oriImage=None
        self.m_usimgRect=[]

    def getUSimgRectByGradientPhase(self, gray_image): 
        # Load the image in grayscale
        print(f"debug:image shape={gray_image.shape}")
        kernelSize=3
        outputImgDepth=cv2.CV_32F #-1
        # Compute the gradient in the x direction
        ##grad_x = cv2.Sobel(gray_image, outputImgDepth, 1, 0, ksize=kernelSize)
        #grad_x = cv2.Scharr(gray_image, outputImgDepth, 1, 0)
        #np.savetxt('output-gradX.csv', grad_x, delimiter=',' ,fmt='%.1f')    
        # Compute the gradient in the y direction
        ##grad_y = cv2.Sobel(gray_image, outputImgDepth, 0, 1, ksize=kernelSize)
        grad_y = cv2.Scharr(gray_image, outputImgDepth, 0, 1)
        #plt.imshow(grad_y)
        #print(f"debug:image gradient shape={grad_y.shape}")
        #np.savetxt('output-gradY.csv', grad_y, delimiter=',', fmt='%.1f')    

        #print(f"debug:iRowVals[ 23]={grad_y[24,:]}")
               
        topRow=-1
        bottomRow=-1
        leftCol=-1
        rightCol=-1

        rowCnt=grad_y.shape[0]
        colCnt=grad_y.shape[1]
        left_one_third=0.3*rowCnt
        bottom_one_third=0.66*rowCnt
        rowValThreshold=colCnt*self.m_percentage
        
        rowStartTopIndex=5 #assume the top 5line not include the header to us-image edge;
        for irow in range(rowStartTopIndex, rowCnt):
            iRowVals=grad_y[irow, :]
            #print(f"debug:iRowVals={iRowVals}") if irow ==-1 else None
            gtZeroCntTop=np.sum(iRowVals>0)
            nonZeroCntBottom=np.sum(iRowVals<0)
            #print(f"debug:row[{irow}]: has non zero item is: {gtZeroCntTop}, THRESOLD={rowValThreshold}")
            #01-topline and left, right.
            if gtZeroCntTop > rowValThreshold:
                if topRow<3:
                    upper2RowPixelsVals=gray_image[irow-2,:] #this for trapezoid
                    upperRowPixelsVals=gray_image[irow-1,:]
                    thisRowPixelsVals=gray_image[irow,:]
                    upper2RowPixelsMean=np.mean(upper2RowPixelsVals)
                    upperRowPixelsMean=np.mean(upperRowPixelsVals)
                    thisRowPixelsMean=np.mean(thisRowPixelsVals)
                    if upperRowPixelsMean > 7 and upper2RowPixelsMean>7:
                        print(f"debug:row[{irow-1} and {irow-2}]: image mean value is: {upperRowPixelsMean},{upper2RowPixelsVals}. mostly a wrong line in header of screenshot. ignore this line.")
                        continue
                    #print(f"debug:row[{irow}]: has non zero item is: {gtZeroCntTop}, THRESOLD={rowValThreshold}")
                    #print(f"debug:row[{irow}]: image value is: {thisRowPixelsVals}")
                    topRow=irow
                    thisRowVal=iRowVals

                    for xinRow, xval in enumerate(thisRowVal):
                        if xval >0 :
                            setThisAsLeft=True
                            for iextend in range(xinRow, xinRow+10, 1):
                                if (iextend >= colCnt) or thisRowVal[iextend] <=0:
                                    setThisAsLeft=False
                                    topRow=-1
                                    break
                            if setThisAsLeft:
                                #print(f"debug: found xleft in top line={xinRow}")
                                leftCol=xinRow
                                break
                    for xinRow in range(len(thisRowVal)-1, 0, -1):#right to left
                        xval=thisRowVal[xinRow]
                        if xval >0 :
                            setThisAsRight=True
                            for iextend in range(xinRow, xinRow+10, 1):
                                if  (iextend >= colCnt) or thisRowVal[iextend] <=0:
                                    setThisAsRight=False
                                    break
                            if setThisAsRight:
                                #print(f"debug: found rightCol in top line={xinRow}")
                                rightCol=xinRow
                                break
            #02-bottom
            if nonZeroCntBottom > rowValThreshold*0.2:
                bottomRow=irow

        while (bottomRow < bottom_one_third):
            rowValThreshold=0.75*rowValThreshold
            for irow in range(rowCnt-1, bottomRow, -1):
                iRowVals=grad_y[irow, :]
                nonZeroCnt=np.sum(np.abs(iRowVals)>1)
                if nonZeroCnt > rowValThreshold:
                    bottomRow=irow
                    
        print(f"topRow={topRow}, Bottom={bottomRow}, ", end='')
        print(f"leftCol={leftCol}, rightCol={rightCol}")

        # history: [topRow, bottomRow, leftCol,rightCol]    
        return [(leftCol,topRow), (rightCol,bottomRow)]


    def getUsImgRectAreaInfo(self, origin_img:np.ndarray):
        self.m_oriImage=origin_img
        imgBgr=origin_img.copy()
        img_removedColors=tryRmAllNonGrayscalePixels(imgBgr)

        gray_image=cv2.cvtColor(img_removedColors, cv2.COLOR_BGR2GRAY)
        openingImg = getOpeningImg(gray_image)

        roiInfo=self.getUSimgRectByGradientPhase(openingImg)
        for icoord in roiInfo:
            if icoord[0] <0 or icoord[1] <0:
                print(f"Error: {self.m_imgname}:ROI Coordinate Invalid:{roiInfo}")
                return
        if False:#debug
            drawedInmg=self.drawCropRectOnImage(self.m_oriImage, roiInfo)
            self.showCropedImg(drawedInmg, roiInfo)
        self.m_usimgRect=roiInfo
        return roiInfo

    def getCropedImg(self, image, cropInfo):
        print(f"debug: cropInfo={cropInfo}, image shape={image.shape}")
        if image.shape[-1]>1:
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        topleft, rightBottom = cropInfo
        x_start, y_start=topleft
        x_end, y_end = rightBottom
        # Crop the image
        cropped_image = image[x_start:x_end, y_start:y_end]

        return cropped_image

    
    def getRoiAndSaveCropedImgToFile(self, origin_img:np.ndarray):
        roiInfo=self.getUsImgRectAreaInfo(origin_img)
        cropped_image=self.getCropedImg(origin_img, roiInfo)

        imgPath=Path(self.m_imgname)
        pathstem=imgPath.stem
        newName=pathstem+"_crop"
        newimgname=self.m_imgname.replace(pathstem, newName)
        print(f"debug:cropped_image.shape={cropped_image.shape}, newName={newimgname}")
        cv2.imwrite(newimgname, cropped_image)
        
    def getCropInfoAndSaveToLmstartJson(self, origin_img:np.ndarray):
        roiInfo=self.getUsImgRectAreaInfo(origin_img)

        imgPath=Path(self.m_imgname)
        lmstartBaseName=r'lmstart.json'
        lmstartFile=imgPath.parent.joinpath( lmstartBaseName)

        if not lmstartFile.exists():
            print(f"Error: json file Not Exist:{lmstartFile}")
            return -1
        with open(lmstartFile, 'r') as f:
            jims=json.load(f)

        jims["USImgRectLRTD"]=roiInfo
        if os.path.exists(lmstartFile):
            os.remove(lmstartFile)
            print(f"\t>>>>>delete file{lmstartFile}")
        with open(lmstartFile, 'w') as jfp:
            json.dump(jims, jfp)

        return 0

    def main_processDicomFolder(self, dcmfolder:pathlib):
        imgfiles=[i for i in sorted(dcmfolder.glob('*.png'))]
        if len(imgfiles) < 1:
            print(f"{dcmfolder} contains no png!")
            return -1
        firstImgfile=imgfiles[0]

        self.m_imgname = firstImgfile
        imageBgr=cv2.imread(firstImgfile)

        return self.getCropInfoAndSaveToLmstartJson(imageBgr)

def processJsonInCases(casesFolder):
    working_dir=pathlib.Path(casesFolder)
    casefolders = working_dir.iterdir()
    for icase in casefolders:
        icasedir=os.path.join(working_dir, icase)
        
        dcmfolder=icasedir
        print(f"\n\nProcess:{dcmfolder}")
        cropimg=CropUsImageClass(dcmfolder)
        failed =cropimg.main_processDicomFolder(Path(dcmfolder))
        
        if 0 != failed:
            print(f"\tconvert failed!!!")
        else:
            print("success,,,")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print(f"App Image")
    else:
        #fp = '/media/eton/hdd931g/42-workspace4debian/10-ExtSrcs/ITKPOCUS/itkpocus/tests/data/83CasesFirstImg/thyroidNodules_axp-042_frm-0001.png'
        alldcmfolder=sys.argv[1]
        print(f"\n\nProcess:{alldcmfolder}")
        processJsonInCases(alldcmfolder)
"""
python /mnt/d/000-srcs/210822-thyroid_train/annotationFormatConverter/cropScreenshotAndAdjustJson.py `pwd`
"""