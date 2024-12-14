import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path 

class CropUsImageClass:
    def __init__(self, imgfile:str):
        self.m_imgname = imgfile
        self.m_percentage=0.3
        self.m_oriImage=None

    def getUSimgRectByGradientPhase(self, gray_image): 
        # Load the image in grayscale
        kernelSize=3
        outputImgDepth=cv2.CV_32F #-1
        # Compute the gradient in the x direction
        ##grad_x = cv2.Sobel(gray_image, outputImgDepth, 1, 0, ksize=kernelSize)
        grad_x = cv2.Scharr(gray_image, outputImgDepth, 1, 0)
        np.savetxt('output-gradX.csv', grad_x, delimiter=',' ,fmt='%.1f')    
        # Compute the gradient in the y direction
        ##grad_y = cv2.Sobel(gray_image, outputImgDepth, 0, 1, ksize=kernelSize)
        grad_y = cv2.Scharr(gray_image, outputImgDepth, 0, 1)
        np.savetxt('output-gradY.csv', grad_y, delimiter=',', fmt='%.1f')    
               
        topRow=-1
        bottomRow=-1
        leftCol=-1
        rightCol=-1

        rowCnt=grad_y.shape[0]
        colCnt=grad_y.shape[1]
        left_one_third=0.3*colCnt
        bottom_one_third=0.66*rowCnt

        rowValThreshold=colCnt*self.m_percentage
        for irow in range(rowCnt):
            irowval=grad_y[irow, :]
            #print(f"debug:irowval={irowval}") if irow ==-1 else None
            nonZeroCntTop=np.sum(irowval>0)
            nonZeroCntBottom=np.sum(irowval<0)
            #01-topline and left, right.
            if nonZeroCntTop > rowValThreshold:
                #print(f"debug:row[{irow}]: has non zero item is: {nonZeroCnt}")
                if topRow<3:
                    topRow=irow
                    thisRowVal=irowval
                    for xinRow, xval in enumerate(thisRowVal):
                        if xval >0 :
                            setThisAsLeft=True
                            for iextend in range(xinRow, xinRow+10, 1):
                                if thisRowVal[iextend] <=0:
                                    setThisAsLeft=False
                                    break;
                            if setThisAsLeft:
                                print(f"debug: found xleft in top line={xinRow}")
                                leftCol=xinRow
                                break;
                    for xinRow in range(len(thisRowVal)-1, 0, -1):#right to left
                        xval=thisRowVal[xinRow]
                        if xval >0 :
                            setThisAsRight=True
                            for iextend in range(xinRow, xinRow+10, 1):
                                if thisRowVal[iextend] <=0:
                                    setThisAsRight=False
                                    break;
                            if setThisAsRight:
                                print(f"debug: found rightCol in top line={xinRow}")
                                rightCol=xinRow
                                break;
            #02-bottom
            if nonZeroCntBottom > rowValThreshold*0.2:
                bottomRow=irow

        while (bottomRow < bottom_one_third):
            rowValThreshold=0.75*rowValThreshold
            for irow in range(rowCnt-1, bottomRow, -1):
                irowval=grad_y[irow, :]
                nonZeroCnt=np.sum(np.abs(irowval)>1)
                if nonZeroCnt > rowValThreshold:
                    bottomRow=irow
                    
                
        print(f"topRow={topRow}, Bottom={bottomRow}")

        print(f"leftCol={leftCol}, rightCol={rightCol}")

        return [topRow, bottomRow, leftCol,rightCol]

    def getUSimgRectByGradientPhaseV1(self, gray_image): 
        # Load the image in grayscale

        # Compute the gradient in the x direction
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        np.savetxt('output-gradX.csv', grad_x, delimiter=',' ,fmt='%.1f')    
        # Compute the gradient in the y direction
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        np.savetxt('output-gradY.csv', grad_y, delimiter=',', fmt='%.1f')    
        
        # Compute the gradient magnitude
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        np.savetxt('output-magnitude.csv', grad_magnitude, delimiter=',',  fmt='%.1f')    
        print(f"grad_magnitude {grad_magnitude.shape}, {grad_magnitude[300:304,300:305]}")
        # Compute the phase angle of the gradient vectors
        grad_phase = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        np.savetxt('output-phase.ecsv', grad_phase, delimiter=',',  fmt='%.1f')    
        grad_phase[grad_phase==90]=0
        grad_phase[grad_phase==180]=0
        grad_phase[grad_phase==270]=0
        #print(f"grad_phase {grad_phase.shape}, {grad_phase[3:4,0:229]}")
        
        topRow=-1
        bottomRow=-1

        rowCnt=grad_phase.shape[0]
        colCnt=grad_phase.shape[1]
        left_one_third=0.3*colCnt
        bottom_one_third=0.66*rowCnt

        rowValThreshold=colCnt*self.m_percentage
        for irow in range(rowCnt):
            irowval=grad_phase[irow, :]
            #print(f"debug:irowval={irowval}") if irow ==-1 else None
            nonZeroCnt=np.sum(irowval>1)
            
            if nonZeroCnt > rowValThreshold:
                #print(f"debug:row[{irow}]: has non zero item is: {nonZeroCnt}")
                if topRow<3:
                    topRow=irow
                else:
                    bottomRow=irow

        while (bottomRow < bottom_one_third):
            rowValThreshold=0.75*rowValThreshold
            for irow in range(rowCnt-1, bottomRow, -1):
                irowval=grad_phase[irow, :]
                nonZeroCnt=np.sum(irowval>1)
                if nonZeroCnt > rowValThreshold:
                    bottomRow=irow
                    
                
        print(f"topRow={topRow}, Bottom={bottomRow}")

        leftCol=-1
        rightCol=-1
        for icol in range(colCnt):
            icolval=grad_phase[:,icol]
            nonZeroCnt=np.sum(icolval>0)
            if nonZeroCnt > rowCnt*self.m_percentage:
                #print(f"debug: col[{icol}]: has non zero item is: {nonZeroCnt}")
                if leftCol<0:
                    leftCol=icol
                else:
                    rightCol=icol
        print(f"leftCol={leftCol}, rightCol={rightCol}")

        return [topRow, bottomRow, leftCol,rightCol]


    def showCropedImg(self, image, cropInfo):
        print(f"debug: cropInfo={cropInfo}, image shape={image.shape}")
        x_start, x_end, y_start,  y_end = cropInfo
        # Crop the image
        cropped_image = image[x_start:x_end, y_start:y_end]
        print(f"debug:cropped_image.shape={cropped_image.shape}")
        # Convert the cropped image from BGR to RGB (for displaying with matplotlib)
        
        # Display the original and cropped images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Cropped Image')
        plt.imshow(cropped_image)
        plt.axis('off')

    def drawCropRectOnImage(self, image, cropInfo):
        if image.shape[-1]<3:
            image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        topleft=(cropInfo[2], cropInfo[0])
        bottomRight=(cropInfo[3], cropInfo[1])
        color=(0,255,0)
        thickness=2
        drawedImg = cv2.rectangle(image,topleft,bottomRight, color,thickness)
        return drawedImg
        
    def saveCropedImg(self, image, cropInfo):
        print(f"debug: cropInfo={cropInfo}, image shape={image.shape}")
        if image.shape[-1]>1:
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_start, x_end, y_start,  y_end = cropInfo
        # Crop the image
        cropped_image = image[x_start:x_end, y_start:y_end]
        imgPath=Path(self.m_imgname)
        pathstem=imgPath.stem
        newName=pathstem+"_crop"
        newimgname=self.m_imgname.replace(pathstem, newName)
        print(f"debug:cropped_image.shape={cropped_image.shape}, newName={newimgname}")
        cv2.imwrite(newimgname, cropped_image)

    def cropImageV3(self, origin_img:np.ndarray):
        self.m_oriImage=origin_img.copy()
        imgBgr=origin_img
        img_removedColors=removeAllNonGrayscalePixels(imgBgr)

        gray_image=cv2.cvtColor(img_removedColors, cv2.COLOR_BGR2GRAY)
        kernel=np.ones((5,5), np.uint8)
        openingImg = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

        roiInfo=self.getUSimgRectByGradientPhase(openingImg)
        self.saveCropedImg(self.m_oriImage, roiInfo)
        drawedInmg=self.drawCropRectOnImage(self.m_oriImage, roiInfo)
        #self.showCropedImg(drawedInmg, roiInfo)

        
    def cropImageV2(self, gray_image:np.ndarray):
        #self.m_imgname = fp
        #imgfile=fp
        #gray_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        
        kernel=np.ones((5,5), np.uint8)
        openingImg = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

        roiInfo=self.getUSimgRectByGradientPhase(openingImg)
        #self.saveCropedImg(gray_image, roiInfo)
        drawedInmg=self.drawCropRectOnImage(gray_image, roiInfo)
        self.showCropedImg(drawedInmg, roiInfo)
        
    def cropImageV1(self, fp:str):
        self.m_imgname = fp
        imageBgr=cv2.imread(fp)
        gray_image = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)

        roiInfo=self.getUSimgRectByGradientPhase(gray_image)
        #self.saveCropedImg(gray_image, roiInfo)
        self.showCropedImg(gray_image, roiInfo)

    def cropImage(self, fp:str):
        self.m_imgname = fp
        imgfile=fp
        gray_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        
        kernel=np.ones((5,5), np.uint8)
        openingImg = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

        roiInfo=self.getUSimgRectByGradientPhase(openingImg)
        #self.saveCropedImg(gray_image, roiInfo)
        self.showCropedImg(openingImg, roiInfo)

def removeAllNonGrayscalePixels(img:np.ndarray):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where the grayscale image is equal to the original image
    mask = np.all(img == gray_image[:, :, np.newaxis], axis=-1)
    
    # Set non-grayscale pixels to zero
    img[~mask] = 0
    #plt.imshow(img)
    return img

def getOpeningImg(fp:str):
    gray_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    kernel=np.ones((5,5), np.uint8)
    openingImg = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    return openingImg

def getOpeningImg(gray_image:np.ndarray):
    #gray_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    kernel=np.ones((5,5), np.uint8)
    openingImg = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    return openingImg

def test():
    fp = '/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/83CasesFirstImg/thyroidNodules_axp-012_frm-0001.png'
    #fp=sys.argv[1]
    cropimg=CropUsImageClass()
    cropimg.cropImageV1(fp)

def testOnV3Crop():
    fp = '/mnt/f/241129-zhipu-thyroid-datas/01-mini-batch/83CasesFirstImg/thyroidNodules_axp-014_frm-0001.png'
    imageBgr=cv2.imread(fp)
    cropimg=CropUsImageClass()
    cropimg.cropImageV3(imageBgr)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print(f"App Image")
    else:
        #fp = '/media/eton/hdd931g/42-workspace4debian/10-ExtSrcs/ITKPOCUS/itkpocus/tests/data/83CasesFirstImg/thyroidNodules_axp-042_frm-0001.png'
        fp=sys.argv[1]
        imageBgr=cv2.imread(fp)
        cropimg=CropUsImageClass(fp)
        cropimg.cropImageV3(imageBgr)

# test cli: 
"""
ls -1 *.png|xargs -I 'var' python /mnt/d/000-srcs/210822-thyroid_train/processImgs/cropUSimgRectByGradientPhase.py `realpath 'var'`
"""