# file format convert from PACS format to labelme format;

#the files struct as follow
# eton@241221 update for one case as 3d; oneline is ellipse which take axis from two lines image;
# eton@241222 make it available from PACS format to labelme foramt;
# eton@250112 refine logs and add more comments, bugfix: Path.glob donot support mulitiple wildcard`casepath.glob('*.{jpg,png,jpeg,bmp}') `, use Pathlib instead;
# eton@250225 add support for multiple nodules in one case;

"""
./02.202401010411.01$ ls -l
total 208
-rwxrwxrwx 1 bibao bibao  98435 Dec 10 11:58 02.202401010411.01.20173.0001.14273100603_crop.jpg
-rwxrwxrwx 1 bibao bibao 102899 Dec 10 11:58 02.202401010411.01.20173.0002.14275200432_crop.jpg
-rwxrwxrwx 1 bibao bibao    741 Dec 13 09:34 02.202401010411.01_pre.json
"""
from enum import Enum
import pathlib
import shutil
import math    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

from typing import List

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
    log_file_name = f"convert301PACS_2labelme_format_{formatted_date_time}.log"
    _ver = sys.version_info
    kwargs = {
        'filename': log_file_name,
        'level': logging.DEBUG,
        'format': '%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s',
        'datefmt': '%Y%m%d %H:%M:%S'
    }
    
    if _ver.minor >= 10:
        kwargs['encoding'] = 'utf-8'
    else:
        print(f"WARNING: this Program develop in Python3.10.12, Current Version May has Problem in `pathlib.Path` to `str` convert.")
    logging.basicConfig(**kwargs)
    


def get_files_with_suffixes(directory: pathlib.Path, suffixes: List[str]) -> List[pathlib.Path]:
    files = []
    for suffix in suffixes:
        files.extend(directory.glob(f'*.{suffix}'))
    return sorted(files)

def getImageFilesBySuffixes(casepath:pathlib.Path):
    image_suffixes = ['jpg', 'png', 'jpeg', 'bmp']
    image_files = get_files_with_suffixes(casepath, image_suffixes)
    return image_files

def getImageFileByJsonFile(json_file:pathlib.Path):
    imgSuffixs=['.jpg', '.png', '.bmp']
    bindImgPath=None
    for imgSuffix in imgSuffixs:
        bindImgPath = json_file.with_suffix(imgSuffix)
        if bindImgPath.is_file():
            break
    return bindImgPath

# Function to find the intersection of two line segments
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def fit_ellipse_to_lines(line1, line2):
    logger.info(f"debug: line1={line1}")
    intersection = line_intersection(line1, line2)
    if intersection is None:
        raise ValueError("Lines do not intersect")

    points = np.array([
        [line1[0][0], line1[0][1]],
        [line1[1][0], line1[1][1]],
        [line2[0][0], line2[0][1]],
        [line2[1][0], line2[1][1]],
        intersection
    ], dtype=np.float32)

    # Fit an ellipse to the points
    ellipse = cv2.fitEllipse(points)
    return ellipse

target_json_str="""
{
	"shapes": [
		{
			"label": "ThyNodu",
			"line_color": null,
			"fill_color": null,
			"points": [
				[
					234.07785113437782,
					325.2085282340851
				]
			]
		}
	],
	"lineColor": [
		0,
		255,
		0,
		128
	],
	"fillColor": [
		255,
		0,
		0,
		128
	],
	"imagePath": "frm-0149.png",
	"imageData": null
}
"""

def rotatedRectangleDefinedEllipseToPolygon(rrEllipse:tuple):
    rotatedRect_Center, rotatedRect_size, rotatedRect_angle = rrEllipse
    
    # Define a lambda function to convert float to int
    # Use the lambda function with map to convert the tuple
    convert_to_int = lambda float_tuple: tuple(map(int, float_tuple))
    #logger.info(f"rotatedRect_Center type={type(rotatedRect_Center)}, {rotatedRect_Center}, rotatedRect_size={rotatedRect_size}")
    
    rotatedRect_Center = convert_to_int( rotatedRect_Center)
    rotatedRect_size = convert_to_int( rotatedRect_size)
    rotatedRect_angle = int(rotatedRect_angle)

    # Calculate the axes of the ellipse
    axes = (int(rotatedRect_size[0] / 2), int(rotatedRect_size[1] / 2))

    #logger.info(f"rotatedRect_Center type={type(rotatedRect_Center)}, {rotatedRect_Center}, rotatedRect_size={rotatedRect_size}")

    delta4Angle=30
    ellipse_points = cv2.ellipse2Poly(rotatedRect_Center, axes, rotatedRect_angle, 0, 360, delta4Angle)
    # Convert the points to a polygon
    polygon = np.array(ellipse_points, dtype=np.int32) #shape=(n, 2)
    list_of_tuples = [tuple(row) for row in polygon.tolist()]
    logger.info(f"debug: polygon.shape={polygon.shape},list_of_tuples={list_of_tuples} ")
    #polygon = polygon.reshape((-1, 1, 2))

    return list_of_tuples

class Converter_301PACS2Labelme:
    def __init__(self, casepath:pathlib.Path, imgfile:str=''):
        self.m_casePath=casepath
        self.m_imgname = imgfile
        self.m_imgCount = 0
        self.m_percentage=0.3
        self.m_oriImage=None

    def getImgAndJsonfileName(self):
        self.m_imgCount +=1
        imgNum = self.m_imgCount
        newImgName=f"frm-{imgNum:04}.png"
        newJsonName=f"frm-{imgNum:04}.json"
        return (newImgName, newJsonName)
        
    def applyImageResult2fileOrVisual(self, img:np.ndarray, useOriginImgName=False, newimgname:str=''):
        if len(newimgname) < 2:
            logger.info(f"Err: newimgName not legal!")
            return -1
        if False == useOriginImgName:
            imgPath=pathlib.Path(newimgname)
            pathstem=imgPath.stem
            newName=pathstem+"_crop"
            newimgname=self.m_imgname.replace(pathstem, newName)
        #logger.info(f"debug:cropped_image.shape={cropped_image.shape}, newName={newimgname}")
        cv2.imwrite(newimgname, img)

    def saveConvertedPairToFile(self, imagepath:pathlib.Path, polygonShape:list, targetFolder:pathlib.Path):
        """
        save the image and json to  target. image copy from origin folder, json will assembled here.
        """
        if not imagepath.is_file():
            logger.info(f"Err: image not exist:{imagepath}")
            return -1
            
        newImgName,newJsonName =self.getImgAndJsonfileName()
        newImagePath = targetFolder.joinpath(newImgName)
        newJsonPath = targetFolder.joinpath(newJsonName)
        
        targetFolder.mkdir(parents=True, exist_ok=True)
        shutil.copy(imagepath, newImagePath)
    
        logger.info(f"debug: newImg path={newImagePath}, {newJsonPath}, polygonShape[0]={type(polygonShape[0])}")
        
        j_target_o = json.loads(target_json_str)
        j_shape_item = j_target_o["shapes"][0].copy()
        j_shape_item["points"].clear()


         #--03 create target labelme format json
        target_json=j_target_o.copy()
        target_json['imagePath']=newImgName
        target_json['shapes'].clear()
        
        j_shape_item["points"]= polygonShape# [(33,44)]#polygonShape
        target_json['shapes'].append(j_shape_item)
        
        #--05 write to disk file
        if newJsonPath.is_file():
            newJsonPath.unlink()
        with open(newJsonPath, 'w') as jfp:
            json.dump(target_json, jfp)
        #logger.info(f"write case json and iamge success:{newJsonPath}")
        return 0

class CaseInfoStruct:
    CaseFolderPrefix='301PX'
    def __init__(self):
        self.point=tuple()
        self.line=[tuple(). tuple()]
        
def getPath4AfterConverted(casepath:pathlib.Path):
    pathParent = casepath.parent
    caseName=casepath.name
    substring=".20" ##02.202410281498.01-->301PACS02-2410281498.01
    if substring in caseName:
        caseName=caseName.replace(substring, "-")

    substring=".000000" ##22.0000001629044-->301PACS22-1629044
    if substring in caseName:
        caseName=caseName.replace(substring, "-")
    caseName=f"{CaseInfoStruct.CaseFolderPrefix}{caseName}"
    labelmeFmtfolder=pathParent.joinpath(caseName)

    return labelmeFmtfolder


@multimethod
def line_length2(point1, point2):
    """
    Calculate the length of a line segment defined by two points.

    Parameters:
    point1 (tuple): The coordinates of the first point (x1, y1).
    point2 (tuple): The coordinates of the second point (x2, y2).

    Returns:
    float: The length of the line segment.
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)
    
@multimethod
def line_length(lineSeg):
    pt1, pt2 = lineSeg
    logger.info(f"lineSeg:{lineSeg}, pt1={pt1}")
    return line_length2(pt1, pt2)

def islineInVertical(measLine):
    oneLineVertical = True
    pt1 = measLine[0]
    pt2 = measLine[1]

    deltaX=abs(pt1[0]-pt2[0])
    deltaY=abs(pt1[1]-pt2[1])
    if deltaX > deltaY:
        oneLineVertical = False #horizon

    return oneLineVertical

def getCaseRelatedPathFromAbsolutePath(casepath:pathlib.Path)->str:
    """
    get the case related path from absolute path;
    input: absolute path , eg:"'/a/b/c/301PACS02-2201041312.01/frm-0002.png'"
    output: case related path, eg:"301PACS02-2201041312.01/frm-0002.png"
    """
    allparts=casepath.parts
    casepath=pathlib.Path(*allparts[-2:])
    return str(casepath)


from enum import Flag
class AnnoFormatVersion(Flag):
    """
    Enum class to represent the format version of the JSON data.
    """
    V1 = 1  # Original list-based format
    V2 = 2  # New dict-based format with horizontal/vertical keys
    UNKNOWN = 99920250225  # Unknown format

class JsonParserFor301PX:
    """
    parse the json file in case folder, assume there is only one json, which include all images and measures information;
    input: json path
    output: tupe(return code, parsed info)

    Version 01: [{
        "name": "02.202201030327.02.20200.0006.11005700392_crop.jpg",
        "points": [{
                "x": 235,    "y": 84
            },{
                "x": 305,   "y": 89
            }]}]
    Version 02: {"area": {
        "height": 656,
        "left": 570,
        "top": 190,
        "width": 626
    },"horizontal": {
        "distance": 0,
        "x1": 235,
        "x2": 305,
        "y1": 84,
        "y2": 89
    },"pixel-per-cm": 0,"runler-length": 0,"width": 1920,"height": 1080 }
    - eton@241225 create this class for support multiple json format from Daniel for 301 PACS;
    """
       
    def __init__(self, jsonsPath:List[pathlib.Path], imagefileslist:list):
        """
        do basic check for input json file;
        find the verison then parse the json file;
        """
        self.jsonsPathlist=jsonsPath
        if not (isinstance(jsonsPath, list) and
               all(isinstance(p, pathlib.Path) for p in jsonsPath)):
            raise TypeError(f"jsonsPath must be List[pathlib.Path], got {type(jsonsPath)}")
            
        self.m_jsonPath=jsonsPath[0] if len(jsonsPath) > 0 else None
        self.m_imagefileslist=imagefileslist
        self.m_formattVer=AnnoFormatVersion.UNKNOWN
        self.m_result={0, "OK"}
        self.m_jsonObj=None

        if self.isJsonfileValid():
            self.m_formattVer = self.getDataFormatVersion()

    def parseIt(self):
        if self.isFormatV1():
           self.m_result= self.parseJsonInPACSfolderV1(self.m_jsonPath, self.m_imagefileslist)
        elif self.isFormatV2():
           self.m_result= self.parseAllJsonsInCasefolderV2(self.jsonsPathlist)
        else:
           logger.error(f"Err: json file[{self.m_jsonPath}] format is not supported.")
           self.m_result=(-1, None)

        return self.m_result
        
    def __call__(self):
        return self.parseIt()

    def isJsonfileValid(self):
        jsonfile = self.m_jsonPath
        if not isinstance(jsonfile, pathlib.Path) or not jsonfile.is_file():
            logger.info(f"Err: json file[{self.m_jsonPath}] not exist.")
            return False
        try:
            with open(jsonfile, 'r') as fp:
                self.m_jsonObj=json.load(fp)
                return True
        except Exception as e:
            logger.error(f"File read error: {str(e)}")
            return False
    def getDataFormatVersion(self):
        """
        Check the version of the JSON data."
        """
        jsonfile = self.m_jsonPath
        try:
            if self.m_jsonObj is None:
                with open(jsonfile, 'r') as fp:
                    jsobj = json.load(fp)
            else:
                jsobj=self.m_jsonObj

            if isinstance(jsobj, list) and all("name" in item and "points" in item for item in jsobj):
                return AnnoFormatVersion.V1
            elif isinstance(jsobj, dict) and ("horizontal" in jsobj or "vertical" in jsobj):
                return AnnoFormatVersion.V2
            else:
                return AnnoFormatVersion.UNKNOWN
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            logger.error(f"File read error: {str(e)}")
        return AnnoFormatVersion.UNKNOWN
    

    def isFormatV1(self):
        """Check if JSON matches V1 format structure"""
        return self.m_formattVer == AnnoFormatVersion.V1

    def isFormatV2(self):
        """Check if JSON matches V2 format structure"""
        return self.m_formattVer == AnnoFormatVersion.V2

    def parseJsonInPACSfolderV1(self, jsonPath:pathlib.Path, imagefileslist:list):
        """
        parse the json file in case folder, assume there is only one json, which include all images and measures information;
        input: json path
        output: tupe(return code, parsed info)
        - eton@241222;
        """
        if not jsonPath.is_file():
            logger.info(f"Error: json file[{jsonPath}] not exist.")
            return (-1, None)
        
        if self.m_jsonObj is None:
            with open(jsonPath, 'r') as fp:
                jsobj = json.load(fp)
        else:
            jsobj=self.m_jsonObj

        itemCntInJson=len(jsobj)
        imgsCnt=len(imagefileslist)
        if itemCntInJson != imgsCnt:
            logger.info(f"WARNING: json has {itemCntInJson} items, but images count is {imgsCnt}.!!!")
        
        twoIntersectionLines=[]
        oneLineSegments=[]
        allimgMeasPairs=[]  # parse json from files;
        for iitem in jsobj: ##01- get info from origin files;
            #logger.info(f"iitem={iitem}")
            bindImgName=iitem["name"]
            measurePoints=iitem["points"]
            logger.info(f"debug: bindImgName={bindImgName} ,  measurePoints={measurePoints}")
            pointTupleList=[]
            for measPt in measurePoints:
                pointTupleList.append((measPt['x'], measPt['y']))
            bindImgPath=jsonPath.parent.joinpath(bindImgName)
            imgMeasPair = None

            if 2 == len(pointTupleList):
                point1=pointTupleList[-1]
                point2=pointTupleList[-2]
                oneLineSegments=[point1, point2]
                imgMeasPair=(bindImgPath, oneLineSegments) #(image path, image measure points);
            elif 4 == len(pointTupleList):
                twolines=[]
                for idx in range(0, 4, 2):
                    point1=pointTupleList[idx]
                    point2=pointTupleList[idx+1]
                    twolines.append(point1)
                    twolines.append(point2)

                line1=twolines[0:2]
                line2=twolines[2:]
                twoIntersectionLines=[line1, line2]
                #imgMeasPair=(bindImgPath, twoIntersectionLines) # lineSegment as item unit
                #make all item is points
                imgMeasPair=(bindImgPath, twolines)  # points as item unit
                
            if type(imgMeasPair) is tuple:
                allimgMeasPairs.append(imgMeasPair)
        return [0, allimgMeasPairs]

    def parseOneJsonInCasefolderV2(self, jsonPath:pathlib.Path):
        """
        parse the json file in case folder, assume there is only one json match same name image, which include same name image's measures information;
        input: json path
        output: tupe(return code, parsed info) if failed else the (image path, measure points) tuple;
        - eton@250225 first version;
        """
        errResult = (-1, None)

        if not jsonPath.is_file():
            logger.error(f"Error: json file[{jsonPath}] not exist.")
            return errResult
        try:
            with open(jsonPath, 'r') as fp:
                jsobj = json.load(fp)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return errResult
        except Exception as e:
            logger.error(f"File read error: {str(e)}")
            return errResult
        
        points = []
        measureKeyNames=['horizontal', 'vertical'] # Extract horizontal,vertical line points (x1,y1) and (x2,y2)
        for keyName in measureKeyNames:
            if keyName in jsobj:
                lineSeg = jsobj[keyName] if keyName in jsobj else None
                if lineSeg is not None:
                    try:
                        points.extend([(lineSeg['x1'], lineSeg['y1']), (lineSeg['x2'], lineSeg['y2'])])
                    except Exception as e:
                        logger.error(f"Error: json file[{getCaseRelatedPathFromAbsolutePath(jsonPath)}] has invalid points. lineSeg={lineSeg}")
                        return errResult
                
        if len(points) < 4:
            logger.error(f"Error: json file[{getCaseRelatedPathFromAbsolutePath(jsonPath)}] has {len(points)} points < 4.")
            return errResult

        bindImgPath=getImageFileByJsonFile(jsonPath)

        if not isinstance(bindImgPath, pathlib.Path) or not bindImgPath.is_file():
            logger.error(f"Error: json file[{getCaseRelatedPathFromAbsolutePath(jsonPath)}] has no image file.")
            return errResult            
        imgMeasPair=(bindImgPath, points)

        return imgMeasPair
    def parseAllJsonsInCasefolderV2(self, jsonsPathlist:List[pathlib.Path]):
        """
        parse the json file in case folder, assume there is only one json match same name image, which include same name image's measures information;
        input: json path list
        output: tupe(return code, parsed info)
        - eton@250225 first version;
        """
        allimgMeasPairs=[]
        errResult = (-1, None)
        for jsonPath in jsonsPathlist:
            resultTuple = self.parseOneJsonInCasefolderV2(jsonPath)
            expectImgPath, expectPoints = resultTuple
            if not isinstance(expectImgPath, pathlib.Path) or not expectImgPath.is_file() or not isinstance(expectPoints, list):
                logger.error(f"Error: json file[{getCaseRelatedPathFromAbsolutePath(jsonPath)}] parse failed.")
                continue
            imgMeasPair=(expectImgPath, expectPoints)
            allimgMeasPairs.append(imgMeasPair)

        return [0, allimgMeasPairs]

def find_pointsOnOrthogonalLineByDistance(point1, point2, outLen:float):
    """
    Calculate a line orthogonal to the given line, with a specified length, intersecting at the center.

    Parameters:
    point1 (tuple): The coordinates of the first point of lineA (x1, y1).
    point2 (tuple): The coordinates of the second point of lineA (x2, y2).
    outLen (float): The desired length of lineB.

    Returns:
    tuple: The coordinates of the two points of lineB (new_point1, new_point2).
    """
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the midpoint of lineA
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Calculate the direction vector of lineA
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)

    # Calculate the orthogonal direction vector
    ortho_dx = -dy / length
    ortho_dy = dx / length

    # Calculate the endpoints of lineB
    half_length = outLen / 2
    new_point1 = (mid_x + half_length * ortho_dx, mid_y + half_length * ortho_dy)
    new_point2 = (mid_x - half_length * ortho_dx, mid_y - half_length * ortho_dy)

    int_tuple1 = tuple(int(item) for item in new_point1)
    int_tuple2 = tuple(int(item) for item in new_point2)

    return (int_tuple1, int_tuple2)

def processOneLineAndALength(img:np.ndarray, point1:tuple, point2:tuple, anotherAxisLen:float):
    logger.info(f"debug: point1-2={point1}, {point2}, AxisLen={anotherAxisLen}")
    #cv2.line(img,point1, point2, colorRed, 1)
    
    orthoPts = find_pointsOnOrthogonalLineByDistance(point1, point2, anotherAxisLen)
    logger.info(f"debug: orthoPts={orthoPts}")
    #cv2.line(img,orthoPts[0], orthoPts[1], colorRed, 1)
    line1=[point1, point2]
    line2=[orthoPts[0], orthoPts[1]]
    ellipse = fit_ellipse_to_lines(line1, line2)
    #cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    polygon_list_of_tuples = rotatedRectangleDefinedEllipseToPolygon(ellipse)

    
    polygon = polygon_list_of_tuples
    logger.info(f"debug:polygon type={type(polygon)}, len= {len(polygon)}, values[0]={polygon[0]}")
    if type(polygon_list_of_tuples) is not np.ndarray:
         polygon = np.array(polygon_list_of_tuples) #(list_of_tuples)to ndarray
    logger.info(f"debug:polygon type={type(polygon)}, shape={polygon.shape}, len= {len(polygon)}, values[0]={polygon[0]}")

    polygon = polygon.astype(np.int32)

    return (img, polygon_list_of_tuples)

@multimethod
def processOnePACSfolder(casepath:pathlib.Path):
    if isinstance(casepath, str):
        casepath=pathlib.Path(casepath)
    if  not casepath.is_dir():
        logger.error(f"not exist dir:{casepath}")
        return -1
        
    labelmefolderpath=getPath4AfterConverted(casepath)
    
    imgs=getImageFilesBySuffixes(casepath)
    jsons=[ijson for ijson in sorted(casepath.glob("*.json"))]

    if len(jsons) < 1:
        logger.error("Err: json file not found in casefolder:{casepath}")
        return -1
    
    jsonParser=JsonParserFor301PX(jsons, imgs)
    ret, measInfo = jsonParser.parseIt()
    logger.info(f"debug: ret={ret}, meas={measInfo}")
    if ret <0:
        logger.error(f"Err: parse json failed")
        return -1
    allimgMeasPairs=measInfo
    
    idx4OnlyOneLine=-1
    axisLenForOneLine=None
    for imgIdx, imgMeasPair in enumerate(allimgMeasPairs): ##02- get another line length as ellipise another axis;
        bindImgPath = imgMeasPair[0]
        measItem =  imgMeasPair[1]
        oneLineVertical = True
        if  type(measItem) is not list:
            logger.error(f"Err:Num{imgIdx} meas points type Not list.")
            continue
        NofPoints = len(measItem)
        if 2 == NofPoints:
            idx4OnlyOneLine = imgIdx
            logger.info(f"debug: One Line segment:{measItem}")
            pt1 = measItem[0]
            pt2 = measItem[1]

            deltaX=abs(pt1[0]-pt2[0])
            deltaY=abs(pt1[1]-pt2[1])
            if deltaX > deltaY:
                oneLineVertical = False #horizon
            logger.info(f"debug:oneLineVertical ={oneLineVertical}")
            
        elif 4 == NofPoints: #the two line item
            for ptidx in range(0, NofPoints, 2):
                oneitem =measItem[ptidx: ptidx+2]
                oneLine = oneitem
                curisVertical =  islineInVertical(oneLine)
                if True == (oneLineVertical ^ curisVertical):
                    axisLenForOneLine = line_length(oneLine)
                logger.info(f"debug:line{oneLine} vertical={curisVertical}, axisLenForOneLine={axisLenForOneLine}")
        
            logger.info(f"debug:two Line segments:{measItem}")
        if idx4OnlyOneLine >=0:
            onlyOneLinePair = imgMeasPair[idx4OnlyOneLine]
                        
    cvter = Converter_301PACS2Labelme(casepath)
    for imgMeasPair in allimgMeasPairs: ##03 --draw or save to image 
        bindImgPath = imgMeasPair[0]
        measItem =  imgMeasPair[1]
        if  type(measItem) is not list:
            logger.error(f"Err:Num{imgIdx} meas points type Not list.")
            continue
        NofPoints = len(measItem)
    
        img=cv2.imread(bindImgPath)
        drawedImg = None

        if 2 == NofPoints:
            if axisLenForOneLine is None:
                logger.error(f"canot found another length for this only one line image!!! ignore this.")
                continue
            pt1 = measItem[0]
            pt2 = measItem[1]
            drawedImg, shapePolygon = processOneLineAndALength(img, pt1, pt2, axisLenForOneLine)

        elif 4 == NofPoints: #the two line item
            logger.info(f"two Line segments:{measItem}")
            line1=measItem[0:2]
            line2=measItem[2:4]
            ellipse = fit_ellipse_to_lines(line1, line2)
            shapePolygon = rotatedRectangleDefinedEllipseToPolygon(ellipse)
            logger.info(f"debug: ellipse type={type(ellipse)}, ellipse={ellipse}, shapePolygon={type(shapePolygon[0])}")

            drawedImg = img
        if False and drawedImg is not None:
            cvter.applyImageResult2fileOrVisual(drawedImg, True)
        cvter.saveConvertedPairToFile(bindImgPath, shapePolygon, labelmefolderpath)
    return 0
    

def processOnePACSfolder4V2MultiNodulesOneCase(casepath:pathlib.Path):
    if isinstance(casepath, str):
        casepath=pathlib.Path(casepath)
    if  not casepath.is_dir():
        logger.error(f"not exist dir:{casepath}")
        return -1
        
    labelmefolderpath=getPath4AfterConverted(casepath)
    
    imgs=getImageFilesBySuffixes(casepath)
    annojsons=[ijson for ijson in sorted(casepath.glob("*.json"))]

    if len(annojsons) < 1:
        logger.error("Err: json file not found in casefolder:{casepath}")
        return -1
    
    jsonpath = jsons[0]
    jsonParser=JsonParserFor301PX(jsonpath, imgs)
    ret, measInfo = jsonParser.parseIt()
    logger.info(f"debug: ret={ret}, meas={measInfo}")
    if ret <0:
        logger.error(f"Err: parse json failed")
        return -1
    allimgMeasPairs=measInfo
    
    idx4OnlyOneLine=-1
    axisLenForOneLine=None
    for imgIdx, imgMeasPair in enumerate(allimgMeasPairs): ##02- get another line length as ellipise another axis;
        bindImgPath = imgMeasPair[0]
        measItem =  imgMeasPair[1]
        oneLineVertical = True
        if  type(measItem) is not list:
            logger.error(f"Err:Num{imgIdx} meas points type Not list.")
            continue
        NofPoints = len(measItem)
        if 2 == NofPoints:
            idx4OnlyOneLine = imgIdx
            logger.info(f"debug: One Line segment:{measItem}")
            pt1 = measItem[0]
            pt2 = measItem[1]

            deltaX=abs(pt1[0]-pt2[0])
            deltaY=abs(pt1[1]-pt2[1])
            if deltaX > deltaY:
                oneLineVertical = False #horizon
            logger.info(f"debug:oneLineVertical ={oneLineVertical}")
            
        elif 4 == NofPoints: #the two line item
            for ptidx in range(0, NofPoints, 2):
                oneitem =measItem[ptidx: ptidx+2]
                oneLine = oneitem
                curisVertical =  islineInVertical(oneLine)
                if True == (oneLineVertical ^ curisVertical):
                    axisLenForOneLine = line_length(oneLine)
                logger.info(f"debug:line{oneLine} vertical={curisVertical}, axisLenForOneLine={axisLenForOneLine}")
        
            logger.info(f"debug:two Line segments:{measItem}")
        if idx4OnlyOneLine >=0:
            onlyOneLinePair = imgMeasPair[idx4OnlyOneLine]
                        
    cvter = Converter_301PACS2Labelme(casepath)
    for imgMeasPair in allimgMeasPairs: ##03 --draw or save to image 
        bindImgPath = imgMeasPair[0]
        measItem =  imgMeasPair[1]
        if  type(measItem) is not list:
            logger.error(f"Err:Num{imgIdx} meas points type Not list.")
            continue
        NofPoints = len(measItem)
    
        img=cv2.imread(bindImgPath)
        drawedImg = None

        if 2 == NofPoints:
            if axisLenForOneLine is None:
                logger.error(f"canot found another length for this only one line image!!! ignore this.")
                continue
            pt1 = measItem[0]
            pt2 = measItem[1]
            drawedImg, shapePolygon = processOneLineAndALength(img, pt1, pt2, axisLenForOneLine)

        elif 4 == NofPoints: #the two line item
            logger.info(f"two Line segments:{measItem}")
            line1=measItem[0:2]
            line2=measItem[2:4]
            ellipse = fit_ellipse_to_lines(line1, line2)
            shapePolygon = rotatedRectangleDefinedEllipseToPolygon(ellipse)
            logger.info(f"debug: ellipse type={type(ellipse)}, ellipse={ellipse}, shapePolygon={type(shapePolygon[0])}")

            drawedImg = img
        if False and drawedImg is not None:
            cvter.applyImageResult2fileOrVisual(drawedImg, True)
        cvter.saveConvertedPairToFile(bindImgPath, shapePolygon, labelmefolderpath)
    return 0
    

def process_multiPACScases(casesFolder):
    working_dir=pathlib.Path(casesFolder)
    casefolders = working_dir.iterdir()

    for icase in tqdm(casefolders, desc="PACS_exportedData2Labelmeformat Converting:"):
        icasepath=icase
        caseName=icasepath.name
        if caseName.startswith(CaseInfoStruct.CaseFolderPrefix):
            continue
        #if "02.2024" in caseName:
        #    continue
    
        logger.info(f"^^^Process:{icase.name}")
        failed = processOnePACSfolder(icasepath)

        if 0 != failed:
            logger.error(f"process pacs folder:[{icase.name}] failed!!!")
            continue ##break
        else:
            logger.info("process pacs folder success,,,")


if __name__ == "__main__":
    if len(sys.argv)<2:
        logger.info(f"App Image")
    else:
        initLogger()
        casesFolder=sys.argv[1]
        #casesFolder=r"/mnt/f/241129-zhipu-thyroid-datas/10-received-datas/241216-staticPACS_censoredOut/censor_out_pre/02.202401031860.01"
        logger.info(f"ProcessHome:{casesFolder}")
        process_multiPACScases(casesFolder)

        
        















