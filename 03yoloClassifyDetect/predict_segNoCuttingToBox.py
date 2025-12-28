# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# eton@250510 v1.0 make output of yolo-segmentation model not cut by box and model both support work with onnxruntime and torch.
import argparse

import cv2, os, sys
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as torchnnF

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS
from ultralytics.engine.predictor import BasePredictor as YOLO_Predictor
from ultralytics import YOLO as YOLO_PT
import matplotlib.pyplot as plt

debug_code = False
def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = torchnnF.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks
def filter_mask_in_boxes(masks, boxes):
    """
    filter masks to bounding boxes, remove others.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks.
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form.

    Returns:
        (torch.Tensor): masks in box.//currently only support 1 mask in 1 box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
    box_mask = (r >= x1) * (r < x2) * (c >= y1) * (c < y2)  # box_mask shape(n,h,w)
    box_mask0InNumpy = box_mask[0].cpu().numpy().astype(np.uint8)
    if debug_code:
        print(f"debug: box_mask.shape={box_mask.shape}")
        cv2.imwrite(f"/tmp/boxmask.png", (box_mask[0].detach().cpu().numpy() * 255).astype(np.uint8)) 
        print(f"debug: masks.shape={masks.shape}, boxes.shape={boxes.shape}, boxes={boxes}, x1={x1}, y1={y1}, x2={x2}, y2={y2}\n w={w}, h={h}")
    
    for ii, imask in enumerate(masks):
        maskInnumpy = imask.cpu().numpy()
        if debug_code:
            boxxyxy = boxes[ii]
            print(f"debug: imask.shape={imask.shape}, boxxyxy={boxxyxy}")
            max_value = np.max(maskInnumpy)
            min_value = np.min(maskInnumpy)
            print(f"Maximum value: {max_value},Minimum value: {min_value}")
        thresh = 0.0 #99*(max_value + min_value)
        maskInnumpy = (maskInnumpy > thresh).astype(np.uint8)
        
        contours, hierarchy = cv2.findContours(
            maskInnumpy, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
        for idx4contour, iContour in enumerate(contours):
            newmaskimg = np.zeros(maskInnumpy.shape, dtype=np.uint8)
            cv2.drawContours(newmaskimg, [iContour], -1, 1, -1)
            if np.sum(box_mask0InNumpy * newmaskimg) > 0:
                if debug_code:
                    print(f"debug: found a contour that is inside the box, sum={np.sum(box_mask0InNumpy * newmaskimg)}")
                    cv2.imwrite(f"/tmp/maskInBox{idx4contour}.png", newmaskimg*245)
                return torch.from_numpy(newmaskimg)
    
        if debug_code:
            print(f"debug: contours={type(contours)}, len(contours)={len(contours)}, contours[0].type={type(contours[0])}, contours[0].shape={contours[0].shape}")
            cv2.imwrite(f"/tmp/resmask{ii}.png", maskInnumpy*255 )
            outimg = cv2.cvtColor(maskInnumpy, cv2.COLOR_GRAY2BGR)
            outimg = cv2.drawContours(outimg, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(f"/tmp/resmask{ii}-contours.png", outimg)
            # Flatten the array
            flattened_array = maskInnumpy.flatten().clip(0, 1)
            # Plot the histogram
            plt.hist(flattened_array, bins=10, alpha=0.7, color='blue')
            plt.title('Value Distribution in 2D Array')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # Save the plot as an image
            plt.savefig('/tmp/value_distribution.png')
            plt.close()

    return masks 

class YOLOSeg:
    """
    YOLOv8 segmentation model for performing instance segmentation using ONNX Runtime.

    This class implements a YOLOv8 instance segmentation model using ONNX Runtime for inference. It handles
    preprocessing of input images, running inference with the ONNX model, and postprocessing the results to
    generate bounding boxes and segmentation masks.

    Attributes:
        session (ort.InferenceSession): ONNX Runtime inference session for model execution.
        imgsz (Tuple[int, int]): Input image size as (height, width) for the model.
        classes (dict): Dictionary mapping class indices to class names from the dataset.
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold used by non-maximum suppression.

    Examples:
        >>> model = YOLOv8Seg("yolov8n-seg.onnx", conf=0.25, iou=0.7)
        >>> img = cv2.imread("image.jpg")
        >>> results = model(img)
        >>> cv2.imshow("Segmentation", results[0].plot())
    """

    def __init__(self, onnx_model, conf=0.25, iou=0.7, imgsz=640):
        """
        Initialize the instance segmentation model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            imgsz (int | Tuple[int, int]): Input image size of the model. Can be an integer for square input or a tuple
                for rectangular input.
        """
        self.argtask = "segment"
        if onnx_model.endswith(".onnx"):
            print("debug onnx provider:", ort.__version__, ort.get_device())
            self.session = ort.InferenceSession(
                onnx_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"],
            )
            inputs = self.session.get_inputs()
            for iIn in inputs:
                print(f"inputs:{iIn}")
            outputs = self.session.get_outputs()
            for iIn in outputs:
                print(f"outputs:{iIn}")
        else:
            # Initialize PyTorch model and predictor
            cuda = torch.cuda.is_available()
            ckpt = torch.load(onnx_model,map_location="cpu") # YOLO_PT(onnx_model)
            device = torch.device("cuda:0" if cuda else "cpu")
            
            self.model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model 
            self.model.eval()
            self.model.device = device
            self.session =None

        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.conf = conf
        self.iou = iou

    def __call__(self, img):
        return self.predict(img)

    def predict(self, img):
        """
        Run inference on the input image using the ONNX model.

        Args:
            img (np.ndarray): The original input image in BGR format.

        Returns:
            (List[Results]): Processed detection results after post-processing, containing bounding boxes and
                segmentation masks.
        """
        prep_img = self.preprocess(img, self.imgsz)
        if self.session is None:
            # For YOLO PyTorch model
            with torch.no_grad():
                inputImgs = torch.from_numpy(prep_img).to(self.model.device)
                preds = self.model(inputImgs)
                outs = preds
                if debug_code:
                    print(f"debug: torch preds: {type(preds)}, len={len(preds)},{preds[0].shape},")

        elif isinstance(self.session, ort.InferenceSession):
            # For ONNX Runtime model
            outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        else:
            raise ValueError("Unsupported session type.")
        return self.postprocess(img, prep_img, outs)

    def letterbox(self, img, new_shape=(640, 640)):
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image in BGR format.
            new_shape (Tuple[int, int]): Target shape as (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img

    def preprocess(self, img, new_shape):
        """
        Preprocess the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (Tuple[int, int]): The target shape for resizing as (height, width).

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized.
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255  # Normalize to [0, 1]
        return img

    def process_mask(self, protos, masks_in, bboxes, shape):
        """
        Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

        Args:
            protos (torch.Tensor): Prototype masks with shape (mask_dim, mask_h, mask_w).
            masks_in (torch.Tensor): Predicted mask coefficients with shape (n, mask_dim), where n is number of detections.
            bboxes (torch.Tensor): Bounding boxes with shape (n, 4), where n is number of detections.
            shape (Tuple[int, int]): The size of the input image as (height, width).

        Returns:
            (torch.Tensor): Binary segmentation masks with shape (n, height, width).
        """
        c, mh, mw = protos.shape  # CHW
        #print(f"debug: protos.shape={protos.shape}, masks_in.shape={masks_in.shape}")
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # Matrix multiplication
        masks = scale_masks(masks[None], shape)[0]  # Scale masks to original image size
        #print(f"debug: after scale mask: masks.shape={masks.shape}, bboxes={bboxes}")
        masks = filter_mask_in_boxes(masks, bboxes)  # Crop masks to bounding boxes
        return masks.gt_(0.0)  # Convert to binary masks
    
    def postprocess(self, img, prep_img, outs):
        """
        Post-process model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (list): Model outputs containing predictions and prototype masks.

        Returns:
            (List[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        # Extract protos - tuple if PyTorch model or array if exported
        protos = outs[1][-1] if isinstance(outs[1], tuple) else outs[1]
        preds = outs[0] 
        preds, protos = [torch.from_numpy(p) if isinstance(p, np.ndarray) else p for p in (preds, protos)]
        if debug_code:
            print(f"debug: preds+protos:{type(preds)}preds.shape={preds.shape}, protos.type={type(protos)}")
        model_names={0: 'thyNodu'} 
        preds = ops.non_max_suppression(preds, self.conf, self.iou, None, False, max_det=3000, nc=len(model_names), end2end=False, return_idxs=False)
        #print(f"debug: after apply nms: {type(preds)},{len(preds)},preds0.shape={preds[0].shape}, protos.shape={protos.shape}")
        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)#0-3:wxywh,4:conf,5:cls;
            #print(f"debug: pred.shape={pred.shape},pred[:5]={pred[:, :5]},0-3:wxywh,4:conf,5:cls")
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names={'nodu':"Nodule"}, boxes=pred[:, :6], masks=masks), )

        return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.1, help="NMS IoU threshold")
    args = parser.parse_args()

    model = YOLOSeg(args.model, args.conf, args.iou)
    img = cv2.imread(args.source)
    results = model.predict(img)

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        #result.save(filename="/tmp/result-nocut.jpg")  # save to disk

        img = np.copy(result.orig_img)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        #print(f"debug: {type(masks.xy)}, len={len(masks.xy)}, masks={masks}") if len(masks.xy) <= 0 else print("a lot masks found")
        contour = masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        isolated = np.dstack([img, b_mask])
        cv2.imwrite("/tmp/resnocut-isolated.png", isolated)
        lineThickness=1
        cv2.drawContours(img, [contour], -1, (55, 255, 25),thickness=lineThickness,lineType=cv2.LINE_AA )#color:BGR, 
        cv2.imwrite("/tmp/resnocut-contour.png", img)
        print(f"inference done, saved to /tmp/resnocut-isolated.png and /tmp/resnocut-contour.png")
