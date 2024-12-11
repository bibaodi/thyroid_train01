import cv2
import numpy as np
import matplotlib.pyplot as plt

def getUSimgRectByGradientPhase(gray_image): 
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
    for irow in range(rowCnt):
        irowval=grad_phase[irow, :]
        #print(f"debug:irowval={irowval}") if irow ==-1 else None
        nonZeroCnt=np.sum(irowval>1)
        if nonZeroCnt > colCnt*0.6:
            #print(f"debug:row[{irow}]: has non zero item is: {nonZeroCnt}")
            if topRow<3:
                topRow=irow
            else:
                bottomRow=irow
    print(f"topRow={topRow}, Bottom={bottomRow}")

    leftCol=-1
    rightCol=-1
    for icol in range(colCnt):
        icolval=grad_phase[:,icol]
        nonZeroCnt=np.sum(icolval>0)
        if nonZeroCnt > rowCnt*0.61:
            #print(f"debug: col[{icol}]: has non zero item is: {nonZeroCnt}")
            if leftCol<0:
                leftCol=icol
            else:
                rightCol=icol
    print(f"leftCol={leftCol}, rightCol={rightCol}")

    return [topRow, bottomRow, leftCol,rightCol]


def showCropedImg(image, cropInfo):
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


def cropImage(fp:str):
    imageBgr=cv2.imread(fp)
    gray_image = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)

    roiInfo=getUSimgRectByGradientPhase(gray_image)
    showCropedImg(gray_image, roiInfo)

fp = '/media/eton/hdd931g/42-workspace4debian/10-ExtSrcs/ITKPOCUS/itkpocus/tests/data/83CasesFirstImg/thyroidNodules_axp-042_frm-0001.png'

cropImage(fp)

