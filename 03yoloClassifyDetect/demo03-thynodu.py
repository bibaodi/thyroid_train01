from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
datasetf=r'/home/eton/00-src/250101-YOLO-ultralytics/datasets/yoloDataset01.yaml'
results = model.train(data=datasetf, epochs=30)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")
tobeTestImg=r'/home/eton/00-src/250101-YOLO-ultralytics/bus.jpg'
results = model.predict(tobeTestImg)
# Export the model to ONNX format
#success = model.export(format="onnx")
