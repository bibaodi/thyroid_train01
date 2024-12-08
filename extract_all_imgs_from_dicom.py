import pydicom
import math
import os
import sys
import time
import json


def generateImagesFromDicom(dataset, filename="extracted_folder", extractfps=0):
    """
        Go through all items in the dataset and print them with custom format

        Modelled after Dataset._pretty_str()
    """
    cine_rate = 12    
    for data_element in dataset:
        if data_element.name == 'Cine Rate':
            print(f"Cine Rate:{data_element.value}, {type(data_element.value)}")
            cine_rate = data_element.value.real
        if data_element.name == 'Pixel Data':
            pixel_datas = data_element.value
       
    # check frame number and tranfersyntex here
    frmpath = filename + '_frms'
    os.makedirs(filename + '_frms', exist_ok=True)
    
    print('Extract frames from [Pixel Data]:')
    if 0 == extractfps or extractfps > cine_rate:
        delta_step = cine_rate / cine_rate
    else:
        delta_step = cine_rate / extractfps
    frmindex = 0
    for sequence_index, frm in enumerate(pydicom.encaps.generate_pixel_data_frame(pixel_datas)):
        delta = sequence_index - (frmindex * delta_step)
        if delta >= 1 or delta < 0:
            continue
        frmname = os.path.join(frmpath, f"frm-{(frmindex+1):04d}.png")
        #print("generateImagesFromDicom: {0} {1} : {2} bytes".format(">>", frmname, len(frm)))
        with open(frmname, 'wb+') as f:
            f.write(frm)
        frmindex += 1
    print(f"Total generate {frmindex} image...")

    return frmpath if frmindex > 1 else None

def record2File(imagedir, fps, raw_file_type='mp4'):
    """when finish the extract action, make a file to record the status"""
    jsonfile = "lmstart.json"
    now = time.localtime(time.time())
    datatime = "{0}-{1}-{2} {3}:{4}:{5}".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour,now.tm_min, now.tm_sec)
    recorder = {"datetime":datatime, "fps":fps, "imagepath":imagedir, "raw_file_type": raw_file_type}
    print(recorder)
    
    with open(os.path.join(imagedir, jsonfile), "w") as fp:
        json.dump(recorder, fp, ensure_ascii=True)
    return 0
    
def extractDcmToImages(dicom_file, extractfps):
    print("dcmfile:", dicom_file)
    if os.path.isfile(dicom_file) and  os.path.splitext(dicom_file)[1] ==".dcm":
        print("dcmfile is file? :", dicom_file)
        ds = pydicom.dcmread(dicom_file)
        try:
            image_dir = generateImagesFromDicom(ds, filename=dicom_file, extractfps=extractfps)
        except Exception as e:
            print('Err:dcm library error')
            image_dir = None
        if image_dir:
            record2File(imagedir=image_dir, fps=extractfps, raw_file_type="dcm")
        return image_dir
    else:
        return None


def extract_all_dcm_imgs(dcm_root_dir, count=0):
    if not os.path.exists(dcm_root_dir):
        print(f"Dir not exist:{dcm_root_dir}")
        return -1
    subdirs = []
    dirs = sorted(os.listdir(dcm_root_dir))
    print(f"dirs={dirs}")
    for item in dirs:
        path = os.path.join(dcm_root_dir, item)
        if os.path.isdir(path):
            subdirs.append(path)
    if len(subdirs) <1:
        subdirs.append(dcm_root_dir)
    print(f"subdirs={subdirs}")
    
    folder_details = []
    dcm_f_count = 0
    for ct, dcm_dir in enumerate(subdirs):
        print("*"*32)
        print('Handle with {0} [{1}/]'.format(dcm_dir, ct + 1))
        files_in_dir = os.listdir(dcm_dir)
        if len(files_in_dir) < 2:
            getLogger().info(f"Empty dirctory : {dcm_dir}")
            continue
        else:
            for dcmf in files_in_dir:
                fullfilename = os.path.join(dcm_dir, dcmf)
                if not os.path.isfile(fullfilename):
                    continue
                print("file in dir:", fullfilename)
                extractDcmToImages(fullfilename, 100)
                dcm_f_count = dcm_f_count+1
                if count != 0 and count == dcm_f_count:
                    print(f"\n<<<already extract {count} directory dcm file~")
                    return
    return 0


if __name__ == "__main__":
    dcm_root_dir=r"/mnt/f/241129-zhipu-thyroid-datas/12-received_data-updates/241208-increments/002"
    if len(sys.argv)<2:
        print(f"Usage:\n\t App DcmRootDir")
    else:
        dcm_root_dir=sys.argv[1]
        extract_all_dcm_imgs(dcm_root_dir, 0)
