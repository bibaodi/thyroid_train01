"""
    Post-training quantization: optimize tflite model using Keras model.h5
        Float16 quantization	2x smaller, GPU acceleration
        Dynamic range quantization	4x smaller, 2x-3x speedup
        Full integer quantization	4x smaller, 3x+ speedup
History:
    -200703: first support and 8bit only is not support on tf2.2.0 for op:REDUCE_MAX
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from uisbp.train.segmentation.metrics import (CrossEntropyJaccardLoss, MeanDice)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
model_input_size=96

def save_tflite(model, optimize=0, outdir=r'/tmp/'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    suffix=''
    if 0==optimize:
        suffix='f16'
        converter.target_spec.supported_types = [tf.float16]
    elif 1==optimize:
        suffix='dyn' # dynamic range quantization
    elif 1 < optimize:
        sample_dataf=r'/data/mul_imt7.4above28_96/np_data/imgs_test.npy'
        sample_data = np.load(sample_dataf)
        images = tf.cast(sample_data, tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
        def representative_dataset_gen():
            for input_value in ds.take(1000):
                #if input_value.shape[-1] != model_input_size:
                    #input_value = cv2.resize(input_value, (model_input_size, model_input_size), cv2.INTER_AREA)
                yield [input_value]
        converter.representative_dataset = representative_dataset_gen
        if 2==optimize:
            suffix='8bit'
        else:
            print('RuntimeError: Quantization not yet supported for op: REDUCE_MAX.[Test in Tensorflow 2.2.0]')
            os._exit(0)
            suffix='8bitonly' # full 8-bit interge
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    tflite_file = f'{outdir}{os.sep}convert-{suffix}.tflite'
    open(tflite_file, "wb").write(tflite_model)
    return 0

def load_saved_model(fmodel):
    fmodel=r"/train/history_train/res_xiaobai_mul_Plaque.Total.V5.0_0507_20200217T1139_sz96/linknet18_32_0.5_multi_96/linknet_model.h5"

    saved_model=""

    object_weights={'CA': 10266354, 'JV': 343386, 'Plaque': 245248, '_background_': 48519389}
    labels=object_weights.keys()
    weights=[0.01370424,0.40972134, 0.5736747,  0.00289972]

    cejloss=CrossEntropyJaccardLoss(jaccard_weight=1.0, 
                                class_weights=weights,
                                num_classes=len(weights),
                                special_indexs=[], special_weights=[])
    cos={
        "CrossEntropyJaccardLoss":cejloss,
        "mean_dice_ca":MeanDice(list(labels), 'CA'),
        "mean_dice_jv":MeanDice(list(labels), 'JV'),
        "mean_dice_plaque":MeanDice(list(labels), 'Plaque'),

        }
    loaded_model = tf.keras.models.load_model(fmodel, custom_objects=cos)

    return loaded_model

if __name__ == '__main__':
    model=load_saved_model('')
    save_tflite(model=model, optimize=2)
    print("exit...")
