from ultralytics import YOLO
import pathlib


# - eton@250215 change dataset home from absolute to relative;

# Create a new YOLO model from scratch
#model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("thynoduClsBoM.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11-cls-resnet18.yaml")  # load modeli yolo11-cls-resnet18.yaml from yaml 
#model = YOLO("yolo11m-cls.pt")  # load maxs pretrained model

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# datasetf=r'../datasets/301Pacs978Items4ObjectDetectBenignMalignV1_250109/301Pacs978Items4ObjectDetectBenignMalign.yaml'
datasetHome=r'../datasets/'
datasetName=r"301pacsDataInLbmfmtRangeY22-24.clsBoM_extend2times"
datasetName=r"clsBoM_v01_extend2times"
datasetName=r"clsBoM_v05"
datasetName=r"clsGlandPos_v01"
trainMessage="2cd train to classify Benign Malign model, 8/2 for train/val, delete too small images"
trainMessage="3rd test on dong5k images only"
trainMessage="4th with new corrected BoM data singleNodule Part:thyroidNodule4BenMalSingle250322"
trainMessage="5th with new corrected BoM data multiNodule Part:thyroidNodule4BenMalMulti250322"
trainMessage="6th with new corrected BoM data single+multi Nodules Part:thyroidNodule4BenMalSingle/Multi250322"
trainMessage="7th filter tirads 4/5 and use bethesda 2 as benign 6 as malign"
trainMessage="8th make train data 50-50 by delete 15k malign items , based on filter tirads 4/5 and use bethesda 2 as benign 6 as malign"
trainMessage="9th make train data 50-50 by add 15k benign items from SingleNodule, based on filter tirads 4/5 and use bethesda 2 as benign 6 as malign"
trainMessage="10th make train&val data 50-50 by add 15k/2.5k benign items from SingleNodule, based on filter tirads 4/5 and use bethesda 2 as benign 6 as malign"
trainMessage='1st train echo composition 0SOLID  1CYSOL, other classes samples not enough'
trainMessage='1st train echo Nodule Margins'
trainMessage='1st train echo foci'
trainMessage='1st train gland position'
datasetf=datasetHome+datasetName

# Train the model
#results = model.train(data="mnist", epochs=10, imgsz=32)
results = model.train(data=datasetf, epochs=30, imgsz=96, erasing=0.4)#96

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1  # top1 accuracy

# Export the model to ONNX format
success = model.export(format="onnx")
# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")
tobeTestImg=r'../datasets/42-minibatch/thynodu-t03.jpg'
testImg=pathlib.Path(tobeTestImg)
if testImg.is_file():
    results = model.predict(tobeTestImg)
else:
    print(f"test img not exist:{testImg}")
print(f"export to ONNX: success={success}")

