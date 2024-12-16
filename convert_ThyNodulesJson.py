##working at 22/01/29
json_file=r"/mnt/d/workspace/220128-thyroid-nodules/220128-nodules/1.3.6.1.4.1.52026.66519049.5308.126.1.1.20220128001204810.json"

import json
import os
import sys
import numpy

j_target_s="""
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

j_target_o = json.loads(j_target_s)
j_shape_item = j_target_o["shapes"][0].copy()
j_shape_item["points"].clear()
j_shape_item["label"]=str()

#print(j_shape_item,j_shape_item)

def convert_nodule_json_v2(json_file, target_json_dir):
    """
        deal with one xiaobai's json which have multi-labelme json
        - if points less than 4, ignore it. --eton@241202
    """
    
    if not os.path.exists(target_json_dir):
        print(f"json target directory not exist.")
        return -1
    # 01-read from nodule-json-file
    with open(json_file, 'r') as f:
        json_ = json.load(f)
        print("json has item number=", len(json_))
        for i in json_:
            print(f"\titem[{i}]")
        all_imgs_datas = json_['data']
        print("total has data number=", len(all_imgs_datas), type(all_imgs_datas))
        print(type(all_imgs_datas[0]))
        one_data_item = all_imgs_datas[0]
        print(f"\titems in \\data\\")
        for i in one_data_item:
            print(f"\t\t{i}")
#        print(f"\t\t lesions in data.")
        print("\t\t one_data_item['frameNumber']:", one_data_item['frameNumber'])
        
        
        ## extract infomation --all images.
        for one_data_item in all_imgs_datas:
            frameNumber = one_data_item['frameNumber']
            #--02 extract json per image 
            polygonPoints=[]
            filename = f"frm-{(frameNumber+1):04d}"
            #--02.1 remove the exist json file
            target_json_name = os.path.join(target_json_dir, f"{filename}.json")
            if os.path.exists(target_json_name):
                os.remove(target_json_name)
                print(f">>>>>delete file{target_json_name}")

            #--03 create target labelme format json
            target_json=j_target_o.copy()
            target_json['imagePath']=f"{filename}.png"
            target_json['shapes'].clear()
        
            lesions = one_data_item["lesions"]
            isCurrentImageHasLabelmeJsonFile=False
            if(len(lesions)>0):
                for lesion_idx in range(len(lesions)):
                    polygonPoints=lesions[lesion_idx]["polygonPoint"]
            
                    points_len = int((len(polygonPoints)))
                    if points_len < 4:
                        print(f"points not enough 4:[{target_json_dir}/{filename}], ignore it.")
                        continue

                    all_points = numpy.zeros((points_len, 2))
                    all_points = polygonPoints
                    #--03.2 delete zeros
                    available_pts = []
                    for pt in range(points_len):
                        #print(f"point [{pt}]={all_points[pt]}")
                        if all_points[pt][0] is None or all_points[pt][0]<10:
                            #print(f"point[{pt}] is {all_points[pt]} delete it...")
                            continue
                        else:
                            available_pts.append(all_points[pt])
                    
                    if len(available_pts) < 4:
                        print(f"available points not enough 4:[{target_json_dir}/{filename}], ignore it.")
                        continue
                    #--04 create target labelme format json
                    #if lesion_idx>0:
                    target_json["shapes"].append(j_shape_item.copy())
                    
                    target_json['shapes'][lesion_idx]['label']='ThyNodu'
                    target_json['shapes'][lesion_idx]['points']=available_pts
                    isCurrentImageHasLabelmeJsonFile=True
            else:
                target_json['shapes'].clear()
                
            if False == isCurrentImageHasLabelmeJsonFile:
                continue
                    
            #--05 write to disk
            target_json_name = os.path.join(target_json_dir, f"{filename}.json")
            
            if os.path.exists(target_json_name):
                os.remove(target_json_name)
                print(f">>>>>delete file{target_json_name}")
            with open(target_json_name, 'w') as jfp:
                json.dump(target_json, jfp)
            
        
    return 0
  
## convert_nodule_json(json_file, ".")

#241130-process thyroid nodules from aixplorer

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
    
def list_folders(directory):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out only the directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders
    except Exception as e:
        print(f"Error: {e}")
        return []

def list_File_withSuffix(directory, suffix:str="_MARK.json"):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out only the directories
        folders = [entry for entry in entries if entry.endswith(suffix)]
        return folders
    except Exception as e:
        print(f"Error: {e}")
        return []


def processJsonInCases(casesFolder):
    working_dir=casesFolder
    casefolders = list_folders(working_dir)
    for icase in casefolders:
        icasedir=os.path.join(working_dir, icase)
        foldersInCase=list_folders(icasedir)
        jsonInCase=list_File_withSuffix(icasedir)
        print(foldersInCase, jsonInCase)
        
        if(len(jsonInCase)<1 or len(foldersInCase)<1):
            print(f"caseDir:[{icasedir}] missing json/dicom-img-folder, ignore....")
            continue
        json_file_name = os.path.join(icasedir, jsonInCase[0])
        dcm_folder_name = os.path.join(icasedir, foldersInCase[0])
        
        failed = convert_nodule_json_v2(json_file_name, dcm_folder_name)
        if 0 != failed:
            print(f"\tconvert failed!!!")
        else:
            print("success,,,")
#os.listdir(working_dir)


########
if __name__ == "__main__":
    dcm_root_dir=r"/mnt/f/241129-zhipu-thyroid-datas/12-received_data-updates/241208-increments/002"
    if len(sys.argv)<2:
        print(f"Usage:\n\t App DcmRootDir")
    else:
        dcm_root_dir=sys.argv[1]
        processJsonInCases(dcm_root_dir)
        #convert_nodule_json_v2('/mnt/f/240926-RayShap/241129-thyroid-datas/52-debug/thyroidNodules_axp087/1.2.250.1.204.5.8373724313.20210416143848345628.2.0.50.80.2.20241111095056666_MARK.json', r'/mnt/f/240926-RayShap/241129-thyroid-datas/52-debug/thyroidNodules_axp087/targetJsonLabelformatDir')
