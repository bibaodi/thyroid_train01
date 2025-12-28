from ultralytics import YOLO
# - eton@250212 test for add albumentations
# - eton@250215 change dataset home from absolute to relative;

# Create a new YOLO model from scratch
#model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# datasetf=r'../datasets/301Pacs978Items4ObjectDetectBenignMalignV1_250109/301Pacs978Items4ObjectDetectBenignMalign.yaml'
datasetHome=r'/mnt/xin1data/yolo-ultralytics-250101/datasets/'
datasetName=r'ThyNodu_250215.yaml'
datasetName=r'detectTrachea_v01T250525.yaml'
trainMessage="only detect the thyroid nodule, add another4393+11036 items"

datasetf=datasetHome+datasetName
results = model.train(data=datasetf, epochs=31, imgsz=96)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")
tobeTestImg=r'../datasets/42-minibatch/thynodu-t01.jpg'
results = model.predict(tobeTestImg)
# Export the model to ONNX format
success = model.export(format="onnx")
print(f"export to ONNX: success={success}")
