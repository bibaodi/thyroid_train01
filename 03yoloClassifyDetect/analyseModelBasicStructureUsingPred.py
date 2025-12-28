#!/bin/python
import cv2
import numpy as np
import sys
from PIL import Image
from ultralytics import YOLO

import onnxruntime as onnx

s_score_threshold=0.24


def process_mode_output(pred):
    print(f"prediction shape={pred.shape}, type={type(pred)}")  #prediction shape=(1,5, 8400)
    squeezed = np.squeeze(pred)#shape=(1, 5, 8400)
    pred = squeezed
    print(f"prediction shape={pred.shape}, type={type(pred)}")  #prediction shape=(5, 8400)
    predictions = pred.transpose((-1, -2))
    pred=predictions
    print(f"prediction shape={pred.shape}, type={type(pred)}")  #prediction shape=(5, 8400)
    xc = pred[:, 4:5].max(1) > s_score_threshold  # candidates
    print(f"xc shape={xc.shape}, indexshape={pred[:, 4:5].shape}, {xc}")

    indices = np.where(xc.squeeze())
    print(f"indices.type={type(indices)}, indices={indices}")
    for i in indices:
        print(f"true value shape={i.shape},coordinate ={i}")
    
    
    print(f"prediction shape={predictions.shape}")  #prediction shape=(8400, 5)
    predictions=predictions[xc.squeeze()]
    print(f"prediction shape={predictions.shape}")  #prediction shape=(108, 5)

    if predictions.shape[0] <1:
        print(f"WARN: no predictions")
        return []
    debugCnt = 0

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []
    gain=1.0
    rows = predictions.shape[0]
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = predictions[i][4:]
        if classes_scores<s_score_threshold:
            continue
        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)
        #print(f"{i:03}:max_score={max_score}, classes_scores={classes_scores}")

        # Get the class ID with the highest score
        class_id = np.argmax(classes_scores)

        # Extract the bounding box coordinates from the current row
        x, y, w, h = predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3]

        # Calculate the scaled coordinates of the bounding box
        left = int((x - w / 2) / gain)
        top = int((y - h / 2) / gain)
        width = int(w / gain)
        height = int(h / gain)

        # Add the class ID, score, and box coordinates to the respective lists
        class_ids.append(class_id)
        scores.append(max_score)
        boxes.append([left, top, width, height])
        debugCnt+=1
        if debugCnt >100:
            break
    
    print(f"nof boxes: {len(boxes)}")
    iou_threshold = 0.01
    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, s_score_threshold, iou_threshold)
    print(f"indices: {len(indices)}={indices}")


    debugCnt=0
    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        debugCnt+=1
    if debugCnt<10:
       	print(f"{i:04d}: box={box}; scores={score}, class_ids={class_id}")

    scores = predictions[:, 4]
    print(f"prediction score={scores}")
    high_conf_mask = scores > s_score_threshold
    high_conf_preds = predictions[high_conf_mask]

    if len(high_conf_preds) == 0:
        max_conf_idx = scores.argmax()
        high_conf_preds = predictions[max_conf_idx:max_conf_idx + 1]
        return [[x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf]
                for x, y, w, h, conf in high_conf_preds]

    sorted_indices = np.argsort(-high_conf_preds[:, 4])
    high_conf_preds = high_conf_preds[sorted_indices]

    max_conf_box = high_conf_preds[0]
    selected_boxes = [max_conf_box]


#preprocess
import torch
def preprocessImg( im):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(im.transpose(2, 0, 1))  # HWC to CHW, (n, h, w, 3) to (n, 3, h, w)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.float()  
    #im = im.to(self.device)
    #im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
    
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

def predict_with_onnx(modelfile, imagefile):
    model_sess = onnx.InferenceSession(modelfile)
    inputs=model_sess.get_inputs()
    outputs=model_sess.get_outputs()

    print(f"onnx {'='*64}inputs-type={type(inputs)}")
    print(f"model: {model_sess}")
    for input in inputs:
        print(f"input-type={type(input)}, {input}, dir(input)")
    #print(f"outputs-type={type(outputs)}")
    for output in outputs:
        print(f"output-type={type(output)}, {output}")
#input-type=<class 'onnxruntime.capi.onnxruntime_pybind11_state.NodeArg'>,
#NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
#output-type=<class 'onnxruntime.capi.onnxruntime_pybind11_state.NodeArg'>,
#NodeArg(name='output0', type='tensor(float)', shape=[1, 5, 8400])
    inp=inputs[0]
    inp_name=inp.name
    inp_shape = inp.shape #[-2:]

    oup=outputs[0]
    oup_name=oup.name
    imgData=cv2.imread(imagefile)
    imgData.resize(inp_shape)
    imgData = imgData.astype(np.float32) / 255.0
    #image actual size=(768, 1024, 3)
    #imgData.transpose(2, 0, 1)
    print(f"image actual size={imgData.shape}, modelInputShape={inp_shape}")
    results = model_sess.run([oup_name], {inp_name: imgData})
    NofRets=len(results)
    print(f"onnx: results type={type(results)}, len={len(results)}")
    if NofRets>0:
        ret0=results[0]
        process_mode_output(ret0)

        print(f"result type={type(ret0)}, shape={ret0.shape}")
        predNd=ret0[0]#5,8400;
        # Condition: M0[4, :] > 0.1
        condition = predNd[4, :] > 0.1
        # Extract the submatrix using the conditiongain
        subPred = predNd[:, condition]

        for irow in range( min(subPred.shape[1], 5)):
            ipred=subPred[:, irow]
            print(f"{irow}: shape={ipred.shape}, top10={ipred.tolist()}")
        #print(f"result={ret0}")
        #print(f"boxes:{ret0.boxes}")
    return

def predict_TIRADS_01(modelfile, imagefile):
    model = YOLO(modelfile, task='detect')  # Initialize model
    print(f"device={model.device}")
    # Set the device to CPU
    #model = model.to("cpu")
    print(f"device={model.device}")
    
    # Run inference
    #results = model([imagefile])  # results list
    if modelfile.endswith('.onnx'):
        results = model.predict([imagefile], device='cpu')  # results list
    else:
        results = model.predict([imagefile], device=0)  # results list
    NofRets=len(results)
    print(f".pt {'='*64}results type={type(results)}, len={len(results)}")
    if NofRets>0:
        ret0=results[0]
        print(f"result type={type(ret0)}")
        #print(f"result={ret0}")

        print(f"boxes:{ret0.boxes}")

    return
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
    m=r'../250301error_multiBoxes/best.pt'
    mx=m.replace('.pt', '.onnx')
    img=sys.argv[2]
    img=r'/home/eton/00-src/250101-YOLO-ultralytics/datasets/yoloDataset01/images/train/301PACS02-2401010411.01_frm-0002.png'
    img=r'../250301error_multiBoxes/687.jpg'
    #img=sys.argv[2]
    predict_TIRADS_01(m, img)
    predict_with_onnx(mx, img)
#
