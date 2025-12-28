"""
Run object detection on images, Press ESC to exit the program
For Raspberry PI, please use `import tflite_runtime.interpreter as tflite` instead
"""
import re
import cv2
import numpy as np
import os
import sys
import glob
import time
import tensorflow.lite as tflite
import xml.etree.ElementTree as ET
# import tflite_runtime.interpreter as tflite

from PIL import Image

Time = []
Time_Image_Pre = 0.0
Time_predict = 0.0
Time_Image_Post = 0.0
OverThreshold = 0.6

def parse_gt_xml(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    comment: an image may contains one more objects, one object constructs an obj_dict,
             and obj_dicts construct objects_list
    """
    if not os.path.exists(filename):
        print(f"xml file not exist:{filename}")
        return None
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()

        if obj.find('name').text is None:
            raise Exception("None class label")

        obj_dict['name'] = obj.find('name').text    # class name, like coat, pants, bus. etc
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_dict)
        
    return objects


def ap_calculate(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # look for recall value changessihu
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    #print(f"@@@debug: process_image: type of image{type(image)}")
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim
    #print(f"@@@debug: process_image: image.shape{input_data.shape}")
    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    #print(f"@@@debug: outputdetails: {output_details}")
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count
    
    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
    #for i in range(4):
    #    output = interpreter.get_tensor(output_details[i]['index'])
    #    print(f"@@@debug: index[{i}].shape{output.shape}")
    #print(f"@@@debug:\n\tscores={scores}")

    result = []

    for idx, score in enumerate(scores):
        if score > 0.9 or idx == 0:  #[0]下标元素一定是对大概率的.
            result.append({'pos': positions[idx], '_id': classes[idx], 'score': scores[idx] })
        if len(result) > 0:
            break

    return result


def generate_result(result, frame, savepath, relative_size):
    r"""Display Detected Objects"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1.0
    colors = [(0, 0, 255), (0, 255,0)]  # Blue color
    thickness = 1

    # position = [ymin, xmin, ymax, xmax]
    # x * IMAGE_WIDTH
    # y * IMAGE_HEIGHT
    width = relative_size[3] - relative_size[2]
    height = relative_size[1] - relative_size[0]
    width = frame.shape[1] if width == 0 else width
    height = frame.shape[0] if height == 0 else height

    for i,obj in enumerate(result):
        pos = obj['pos']
        _id = obj['_id']
        score = obj['score']
        f = lambda x:(x if x>=0 else 0) 
        x1 = int(f(pos[1]) * width) + relative_size[2]
        x2 = int(f(pos[3]) * width) + relative_size[2]
        y1 = int(f(pos[0]) * height) + relative_size[0]
        y2 = int(f(pos[2]) * height) + relative_size[0]

        cv2.putText(frame, f"P:{score:0.3}", (int(x1*1.1), int(y1*1.1)), font, size, colors[i], thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], thickness)

    #cv2.imshow('Object Detection', frame)
    cv2.imwrite(savepath, frame)


def process_predict(model, input_type, input_ctx, extension):
    """
    model: 模型文件
    input_type: ['0, 1, 2']区分图片, 图片文件夹, 多个图片文件夹三类情形
    input_ctx: 输入的内容图片路径, 图片的文件夹路径, 多个文件夹路径
    """
    if not os.path.isfile(model):
        print(f"model must exist: {model}")
        return -1
    model_name = '-'
    model_name += os.path.basename(model).split('.')[0]
    if not os.path.exists(input_ctx):
        print(f"input ctx must exist")
        return -1
    if os.path.isfile(input_ctx) and 0 == input_type:
        imgs = [input_ctx]
    elif os.path.isdir(input_ctx) and 1 == input_type:
        if input_ctx[-1] == os.sep:
            preffix = ""
        else:
            preffix = f"{os.sep}"
        pattern = f"{input_ctx}{preffix}*{extension}"
        imgs = sorted(glob.glob(pattern, recursive=False))
    elif os.path.isdir(input_ctx) and 2 == input_type:
        if input_ctx[-1] == os.sep:
            preffix = "**{os.sep}"
        else:
            preffix = f"{os.sep}**{os.sep}"
        pattern = f"{input_ctx}{preffix}*{extension}"
        imgs = sorted(glob.glob(pattern, recursive=True))
    if input_type in [1,2]:
        print(f"@@@debug: pattern: {pattern}")
    interpreter = load_model(model)
    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    # Get input index
    input_index = input_details[0]['index']
    #--
    gt_dict = {}
    for ind, image_filename in enumerate(imgs):
        _xml_filename = image_filename.replace(extension,'.xml')
        #print(f"@debug: xml is :{_xml_filename}")
        gt_dict[image_filename] = parse_gt_xml(_xml_filename)
    # -----gt-----extract objects in :param classname:
    bbox_gt_dict = {}
    npos = 0    # all gt bboxes, for recall calculation
    for image_filename in imgs:
        if gt_dict[image_filename] is None:
            continue
        objects = [obj for obj in gt_dict[image_filename]]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        bbox_gt_dict[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}
    global Time, Time_Image_Post, Time_predict
    Time.append(time.time())
    nd = len(imgs)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for indx, img in enumerate(imgs):
        time0 = time.time()
        frame = cv2.imread(img, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #image = image.resize((width, height))
        top = 180
        buttom = 740#frame.shape[0]  #440*0
        left = 327#184#368#*0
        right = 1036#684#994#frame.shape[1]  #582*0
        sub_frame = frame[top:buttom, left:right, :] #rows. colums, channels
        image = cv2.resize(sub_frame, (width, height))
        global Time_Image_Pre
        time1 = time.time()
        Time_Image_Pre +=  (time1 - time0)
        #actual_size = get_image_actual_width_index_range_mem(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        #print(f"@@@debug: actual size={actual_size}")
        #print(f"@@@debug: image.type={type(image)}")
        top_result = process_image(interpreter, image, input_index)
        detect_bbox = top_result[0]['pos']
        f = lambda x:(x if x>=0 else 0) 
        #print(f"@debug: detec={detect_bbox}, frame.shape={frame.shape}")
        x1 = int(f(detect_bbox[1]) * frame.shape[1]) 
        x2 = int(f(detect_bbox[3]) * frame.shape[1])
        y1 = int(f(detect_bbox[0]) * frame.shape[0])
        y2 = int(f(detect_bbox[2]) * frame.shape[0])
        bb = np.array([x1, y1, x2, y2]).astype(float)
        if img in bbox_gt_dict and 'bbox' in bbox_gt_dict[img]:
            bbgt = bbox_gt_dict[img]['bbox'].astype(float)
        else:
            bbgt = None
        #print(f"@debug: bb={bb}, bbgt={bbgt}")
        if bbgt is not None and bbgt.size > 0 and bbgt is not None:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps) #maybe more than one exist
            print(f"@debug: o={overlaps}, ovmax={ovmax}")
        if bbgt is not None and ovmax > OverThreshold:
            tp[indx] = 1
        else:
            fp[indx] = 1

        time2 = time.time()
        Time_predict +=  (time2 - time1)

        bs = os.path.basename(img)
        ds = os.path.dirname(img)
        ds = os.path.join(os.path.dirname(ds), os.path.basename(ds)+model_name)
        if not os.path.exists(ds):
            os.mkdir(ds, mode=511)
        savepath = os.path.join(ds, bs)
        relative_size = (top, buttom, left, right)
        generate_result(top_result, frame, savepath, relative_size)
        time3 = time.time()
        Time_Image_Post +=  (time3 - time2)
        #key = cv2.waitKey(0)
        #if key == 27:  # esc
        #    cv2.destroyAllWindows()
        print(f"@@@debug: img is {img}")
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = ap_calculate(rec, prec)
    print(f"fp={fp}, tp={tp}, rec={rec}, prec={prec}, ap={ap}")
    Time.append(time.time())
    print(f'{"="*20}\n load model time is:{Time[1]-Time[0]}\nall {len(imgs)} images processed time is:{Time[2]-Time[1]}')
    print(f"pre={Time_Image_Pre}\npredict={Time_predict}\npost={Time_Image_Post}")
    return 0

if __name__ == "__main__":
    str_1=r"ffmpeg -threads 8 -i thyroid_zjm_02.wmv -q:v 2 ./thyroid_zjm_02/frm-%04d.png"
    str_2=r"ffmpeg -threads 8 -y -r 30 -i eva_frm-%4d.png  thyroid_zjm_01_result.mp4"
    print(f"usage:\n\t{str_1}\n\t{str_2}\n\tpython app model image type(0|1|2|) png")
    print(f"debug: argv={sys.argv}")
    #输入: 'shape': array([  1, 300, 300,   3]
    #输出: 4个
    #global Time
    Time.append(time.time())
    suffix = '.png'
    if len(sys.argv) > 4:
        suffix = sys.argv[4]
    process_predict(sys.argv[1], int(sys.argv[3]), sys.argv[2], suffix)
