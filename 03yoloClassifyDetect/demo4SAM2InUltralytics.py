import os
import sys
from ultralytics import SAM
from ultralytics import YOLO
from ultralytics import settings

# View all settings
#print(settings)

def useYolo11Seg_1(tobeTestImg):

    # Load a pretrained model
    model = YOLO("yolo11n-seg.pt")
    # Validate the model
    #metrics = model.val()
    #print("Mean Average Precision for boxes:", metrics.box.map)
    #print("Mean Average Precision for masks:", metrics.seg.map)

    # Display model information (optional)
    model.info()

    # Run inference with bboxes prompt
    print(f"test image:[{tobeTestImg}]")

    results = model(tobeTestImg)
    #print(results)
    for r in results:
        print(f"Detected {len(r)} objects in image")   
        # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
 
def useSAM2_1(tobeTestImg, prompts):
    # Load a model
    model = SAM("sam2.1_b.pt")
    # Display model information (optional)
    model.info()

    print(f"test image:[{tobeTestImg}]")

    x,y,w,h=prompts
    centerX=x+0.5*w
    centerY=y+0.5*h
    # Run inference with single point
    #results = model(tobeTestImg, points=[centerX, centerY], labels=[1])

    # Run inference with bboxes prompt
    results = model(tobeTestImg, bboxes=[x,y, x+w, y+h])
    #print(results)
    for r in results:
        print(f"Detected {len(r)} objects in image")   
        r.show()  # display to screen
        
    return 0

    # Run inference with multiple points
    results = model(points=[[473, 183], [514, 226]], labels=[1, 1])

    # Run inference with multiple points prompt per object
    results = model(points=[[[473, 183], [514, 226]]], labels=[[1, 1]])

    # Run inference with negative points prompt
    results = model(points=[[[501, 195], [523, 265]]], labels=[[1, 0]])

def runSegmentModelPredict(tobeTestImg, prompt):
    #imgHome=r'/home/eton/00-src/250101-YOLO-ultralytics/'
    #tobeTestImg=imgHome+r'datasets/yoloDataset01.V0/images/val/301PACS02-2401010320_frm-0001.png'
    useSAM2_1(tobeTestImg, prompt)
    #useYolo11Seg_1(tobeTestImg)

def predict_TIRADS_01(modelfile, imagefile):
    model = YOLO(modelfile)
    # Run inference on 'bus.jpg'
    #results = model([imagefile])  # results list
    results = model.predict([imagefile])  # results list
    NofRets=len(results)
    print(f".pt {'='*64}results type={type(results)}, len={len(results)}")
    if NofRets>0:
        ret0=results[0]
        print(f"result type={type(ret0)}")
        #print(f"result={ret0}")
        boxes=ret0.boxes
        xywh=boxes.xywh[0]
        print(f"boxes:{boxes}, xywh={xywh.tolist()}")

    return xywh
    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Show results to screen (in supported environments)
        r.show()

        # Save results to disk
        #r.save(filename=f"results{i}.jpg")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print(f"Usage: {sys.argv[0]} <model> <image>")
        sys.exit(1)
    m=sys.argv[1]
    m=r'/home/eton/00-src/250101-YOLO-ultralytics/42-250107-train_BenignMalign/runs/detect/train16/weights/best.pt'
    mx=m.replace('.pt', '.onnx')
    img=sys.argv[2]
    img=r'/home/eton/00-src/250101-YOLO-ultralytics/datasets/yoloDataset01/images/train/301PACS02-2401010411.01_frm-0002.png'
    predbox = predict_TIRADS_01(m, img)

    runSegmentModelPredict(img, predbox)

