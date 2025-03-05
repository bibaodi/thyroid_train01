"""
abstract the json format of labelme
- eton@250304 V0.1 
"""
import json
import os

jtargetStr="""
{
  "shapes": [
    {
      "label": "\u9888\u52a8\u8109:CA",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          296,
          253
        ],
        [
          295,
          373
        ],
        [
          430,
          363
        ],
        [
          427,
          255
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
  "imagePath": "frm-0001.png",
  "imageData": null
}
"""

jtargetObj = json.loads(jtargetStr)
jShapeObj = jtargetObj["shapes"][0].copy()
jShapeObj["points"].clear()
jShapeObj["label"]=str()

def getOneShapeObj():
    return jShapeObj.copy()

def getOneTargetObj():
    return jtargetObj.copy()

import pathlib

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