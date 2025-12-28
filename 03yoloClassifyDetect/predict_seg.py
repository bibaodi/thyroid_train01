from ultralytics import YOLO
import cv2
import numpy as np

modelfile=r'runs/segment/train9/weights/best.onnx'
# Load a pretrained YOLO model (recommended for training)
model = YOLO(modelfile)
print("model.args:", model)



# Validate the model
if 0:
    metrics = model.val()
    print("Mean Average Precision for boxes:", metrics.box.map)
    print("Mean Average Precision for masks:", metrics.seg.map)

tobeTestImg=r'../datasets/42-minibatch/thynodu-t01.jpg'

results = model.predict(tobeTestImg)
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    #result.save(filename="/tmp/result.jpg")  # save to disk

    img = np.copy(result.orig_img)
    b_mask = np.zeros(img.shape[:2], np.uint8)
    contour = masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    isolated = np.dstack([img, b_mask])
    cv2.imwrite("/tmp/res-isolated.png", isolated)
    lineThickness=1
    cv2.drawContours(img, [contour], -1, (55, 255, 25),thickness=lineThickness,lineType=cv2.LINE_AA )#color:BGR, 
    cv2.imwrite("/tmp/res-contour.png", img)
