'''
estimate mil on multi object format data

'''
import os
import csv
import datetime

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from uisbp.preprocess import adjust_hist
from uisbp.transform_utils import crop, center_crop, resize
from uisbp.runner import ClassificationRunner, preprocess_img

from uisbp.preprocess import MultiLabelData

def milPred(X, para):
    mil_model = ClassificationRunner()
    mil_model.load_model(model='mil', dset=para[0], img_size=para[1])
    mil_preds = mil_model.predict(X, post_process=False, channel_first_model=False)
    return mil_preds

def calOneCol(npredvalues, thresold, nexpections):
    napred = npredvalues>thresold
    naCorrect = napred == nexpections
    metrics = {}
    metrics['positive_pred_cnt'] = np.count_nonzero(napred)
    metrics['negative_pred_cnt'] = np.count_nonzero(np.logical_not(napred))

    metrics['positive_cmp_cnt'] = np.count_nonzero(naCorrect) # TP + FN
    metrics['negative_cmp_cnt'] = np.count_nonzero(np.logical_not(naCorrect))

    metrics['accuracy'] = accuracy_score(nexpections, napred)
    metrics['precision'] = precision_score(nexpections, napred)
    metrics['recall'] = recall_score(nexpections, napred)
    metrics['f1'] = f1_score(nexpections, napred)

    print(metrics)

    colpred = napred.tolist() + [ metrics['positive_pred_cnt'], metrics['negative_pred_cnt']] + [''] * 4

    keys = ['positive_cmp_cnt', 'negative_cmp_cnt', 'accuracy', 'precision', 'recall', 'f1']
    colCorrect = naCorrect.tolist() +  [metrics[k] for k in keys]

    return  colpred, colCorrect

KEY_LABEL='Plaque' 
KEY_LABEL='BP' 

def milRun(image_dir, out_dir, modelparas):

    print("1. Create np data from input.")
    ml = MultiLabelData(image_dir, outdir)
    imgs, masks, masks_label_names, label_dict, ids = ml.get_imgs_masks_labels(size=448, keyLabel=KEY_LABEL, use_uncertain=False)

    print('imgs.shape, masks.shape, ids.shape', imgs.shape, masks.shape, ids.shape)
    print("model:", modelparas )
    print('label in estimation data:', masks_label_names)

    # MIL Classification
    print("2. MIL predication.")
    modelpara = modelparas[0]
    milpred = milPred(imgs, modelpara)

    print("3. Ouput result")
    metricsname = ['positive_cnt', 'negative_cnt','accuracy', 'precision', 'recall', 'f1']
    metricLines = len(metricsname)
    head = ['imagename', 'expection', 'mil']
    tblTResCols = []

    colFname = [os.path.basename(id) for id in ids]

    tblTResCols.append(colFname.copy() + metricsname)
    
    indexkey= masks_label_names.index(KEY_LABEL)
    keymasklayer = masks[...,indexkey:indexkey+1]
    print('keymasklayer shape', keymasklayer.shape)
    colPixelCount = [np.count_nonzero(x) for x in keymasklayer]
    #print(len(colPixelCount), colPixelCount)
    nexpections = np.array(colPixelCount) > 0 
    #print(len(nexpections), nexpections)
    cnts = [np.count_nonzero(nexpections), np.count_nonzero(np.logical_not(nexpections))]
    print("result count[P/N]:{}/{}".format(cnts[0], cnts[1]))
    tblTResCols.append(nexpections.tolist() + cnts + [""]*(metricLines-2))

    npredvalues = milpred
#    print(milpred)
    tblTResCols.append(npredvalues.tolist() + [""]*metricLines) 

    for x in range(1,10):
        thresold = 0.1 * x
        print("thresold", thresold)
        head.append('>%0.2f' % thresold)
        head.append('correct%0.2f' % thresold)

        colpred, colCorrect = calOneCol(npredvalues, thresold, nexpections)
        tblTResCols.append(colpred)
        tblTResCols.append(colCorrect)

    tblResult = [list(x) for x in zip(*tblTResCols)]

    tblResult.insert(0, head)
    tblResult.append(head)

    finaloutputdir = os.path.join(out_dir, 'run_mil_only')    
    os.makedirs(finaloutputdir, exist_ok=True)
    print(modelpara[0], str(modelpara[1]))
    csvpath = os.path.join(finaloutputdir,
        "run.mil_{0}_{1}.{2}.csv".format(modelpara[0], modelpara[1], str(datetime.date.today())) )

    print(csvpath)

    with open(csvpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tblResult)    

if __name__ == '__main__':

    # change this to point to "images" directory from test data
    # imgs_fname_test.txt imgs_fname_test.txt during train should be here also
    img_root_dir = r'/home/ubuser/Train/data/aprmiltest/apr-test-datasmall/images'
    img_root_dir = r'/home/ubuser/Train/data/miltest/sample1'
    img_root_dir = r'/home/ubuser/Train/data/miltest/PlaqueV2.0.20181210_plaque.precrop'
    img_root_dir = r'/home/ubuser/Train/data/miltest/bpsample_mul'

    # result directory
    outdir = '/home/ubuser/Train/data/miltest/milbin_2019_run/sample1'
    outdir = '/home/ubuser/Train/data/miltest/milbin_2019_run/bpsample_mul'


    # keep same format with some other tools
    modelparas = [('sep', 96)]
   # modelparas = [('bin-plaque-190109', 96)]
   

    milRun(img_root_dir, outdir, modelparas)
