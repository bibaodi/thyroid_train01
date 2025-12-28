# Training Scripts

At Infinia ML we have been running most analyses on a Titan V GPU with CUDA Version 9.0.176 and 
cuDNN Version 7.0.5

## Data Processing

The file `process_data.py` will create data files (stored as `numpy` arrays and text files) 
from the raw frame data (in the format used in the april (jpg images and bbox files) or
may data (png image files and `labelme` json files)) to be used in the MIL and segmentation 
training codes.

The preprocessing is as follows:

* **Binary Label Data**: Crop all images to square 448 x 448 by cropping from the bottom for
portrait images and cropping first from the sides (to remove black) and then cropping from
bottom for landscape images. The masks are cropped in the same way and stored as binary masks 
(i.e. single channel array). The output `numpy` arrays in this case are

    * `imgs_train.npy`: n_image x 448 x 448 x 1 `numpy` array of ultrasound images
    * `imgs_mask_train.npy`: n_image x 448 x 448 x 1 `numpy` array of binary masks
    * `imgs_fname_train.txt`: Single column data file with image filenames (extension excluded)
    
    The folder also contains the same files for test data using  `_test` instead of `_train`.

* **Multi-label data**: Same cropping as above. Now masks are stored as 448 x 448 x n_labels
binary data. Each channel is a binary mask for a given label (including the background). 
The output `numpy` arrays in this case are

    * `imgs_train.npy`: n_image x 448 x 448 x 1 `numpy` array of ultrasound images
    * `imgs_mask_train.npy`: n_image x 448 x 448 x n_labels `numpy` array of multi-class 
    binary masks
    * `imgs_fname_train.txt`: Single column data file with image filenames (extension excluded)
    * `labels.txt`: single column text files with label names corresponding to channels in
    mask array.
    
    The folder also contains the same files for test data using  `_test` instead of `_train`.

The script takes 4 command line arguments

* `rootdir`: This is the full path to the root data directory. For example if the raw images
for a single video are stored in `data/images/03_05_XX_BP_L_01/` then the root directory
is `data/images`.

* `dataset`: This is a filter that lest the code know which preprocessing steps to use. The
available options are `binary` and `multi` for binary data (only BP mask) or multi-class 
data (multiple label masks), respectively.

* `outdir`: Output directory for data. For example if you specify `data_output` as the `outdir`
argument then the numpy data will be stored in `data_output/np_data`

* `merge_bp`: Option to merge separate BP segments into one contiguious region.

This file only needs to be run once for each data set.

## Segmentation Training

The file `train_segmentation.py` will perform the training process for the 
[LinkNet](https://arxiv.org/pdf/1707.03718.pdf) or [Unet](https://arxiv.org/pdf/1505.04597.pdf) 
model given input images, masks and IDs which were created by the `process_data` script. 
The network configuration can be changed with several command line arguments below. The
defaults are set to what we have found most effective at Infinia ML but they may need to be
modified for future data.

We have used a custom loss function found in `segmentation.metrics`. This loss function is a
combination of the loss function used in the Unet and LinkNet papers above. It is a 
weighted (by pixel percentage in training set) cross entropy (binary or categorical 
depending on training data) loss plus a loss based on the Jaccard coefficient 
(intersection over union). The crossentropy loss gives good per-pixel classification while
the Jaccard loss rewards the network for predicting solid masks.

The training is done using the Adam optimizer with momentum of 0.99 (as recommended in
the Unet paper). The learning rate is varied using Stochastic Gradient Descent with Warm
Restarts ([SGDR](https://arxiv.org/pdf/1608.03983.pdf)). This method decreased the learning
rate over a number of epochs and then increases it back to the starting point and decreases
again in a cycle. This allows the algorithm to explore many modes finding the most general
solution. This method, also reduces the need for carefully tuning the learning rate manually.

Data augmentation is also performed to help with generalization of the model and to reduce
overfitting. The following augmentations are performed at training time
        
* Elastic Transformation
* Random left-right flips
* Random zoom-in and zoom-out
* Random rotation 
* Random shear
* Random left-right / up-down shift
* Random gain adjustments

After training is performed, the script will output a frozen protobuf graph for use 
in applications. The script will also output numpy arrays of the predicted probability
masks on the test set as `ypred.npy`. 

The arguments to the script are as follows.

* `datapath`: Full path to numpy arrays that contain data. The data files will have to be 
made beforehand with the `process_data` script.

* `dataset`: String indicating which dataset we are using. Options are `apr`, `may`,
`binary`, or `multi`. To keep backwards compatibility `apr` == `binary` and `may` == `multi`
but for future datasets you can just use `binary`, or `multi` depending on the dat type. 

* `model`: Name of the model to use. The three options are `unet`, `linknet18`, and
`linknet34` for the U-net and LinkNet models, respectively. `linknet18` and `linknet43`
use [Residual networks](https://arxiv.org/pdf/1512.03385.pdf) "resnet18" and "resnet34" 
as encoders, respectively. **Note**: `linknet18` is what we have been using at Infinia ML
and will give better results on the current data. `linknet18` is set as the default.

* `start_channel_depth`: The starting channel depth of the encoder. Larger values should
be used for complicated data. We have found that 32 works best for LinkNet models and 16 
works best for Unet models. The default is 32.

* `dropout`: Dropout value to use for the encoder part of the network. We have found that
a value of 0.5 balances model accuracy with model generalizability. The default is 0.5.

* `size`: Image size to use for training. The data preprocessing will already crop the
data to square images. This just tells the code to resize to a new size.

* `outdir`: Full path to output directory where all results and model products will be stored.

* `seed`: Random number seed to use in train/validation split.

* `batch_size`: Batch size to use in model.

* `learning_rate`: The maximum learning rate to use in training. We have found that 5e-3
works best for LinkNet models and 1e-2 for Unet models. The default is 5e-3.

* `checkpoint`: Full path to a hdf5 file containing model weights if picking up from 
another run.

* `use_binary_output`: Output model that only segments BP, even though model was trained with multi-class data

* `flip_data`: Flip all data to be right oriented. Helps with stability across frames.

* `channels_first_output`: Output model file that uses a channels-first format (i.e, output segmentation will be n_channels x size x size)

* `combine_train_valid`: Combine training and validation data to make larger training set. Useful with limited data samples.

* `gpuid`: Integer specifying which GPU device to use

* `keylabel`: Key class label of segmentation.

* `use_augmentation`: Use augmentation during training. Default is false.

 
**Training tips**: 
1. If you find the algorithm diverging (i.e. the loss increases with epoch) then you
may need to reduce the initial learning rate from the default value.
2. If the model is not getting good performance on the training or validation set then 
you can try increasing the `start_channel_depth` parameter (i.e. 16 -> 32) or use a larger
LinkNet network (i.e. `linknet18` -> `linknet34`).
3. If the model is overfitting, you can either make the network smaller (by reversing
the changes mentioned above) or by increasing dropout.

**Note**:
To reproduce results from September 2018 run the following command:

`python train_segmentation.py --datapath /path/to/data --dataset multi --outdir /path/to/output/directory/ --flip_data --combine_train_valid`

plus any additional output option flags.


## Detection (MIL) Training

The python script `train_mil.py`, which can be run from the shell script `run_train_mil.sh`, will perform the training process for the 
MIL model given input images, masks and IDs. The model architecture and hyper-parameters (learning rates, etc) are fixed to what we found
to be best during our training at Infinia ML. 

After training is performed, the python script `validate_mil.py`, which can be run from the shell script `run_validate_mil.sh`, will perform the validation process to show (in a print file) which epoch is the "best" and the corresponding detection threshold.

Once you've identified the best epoch, the shell script `run_freeze_and_optimize_tf_graph_mil.sh`, which uses the python script `freeze_and_optimize_tf_graph_mil.py` can be run to output an optimized frozen protobuf graph for use in applications. 

The arguments to the `train()` function used in `train_mil.py` and the `validate()` function used in `validate_mil.py` can be found documented in the configuration yaml file located in `mil/config.yml`. The datapath must be specified under the key `processed-data-path` in the configuration yaml file.

### NOTE
In most cases, bp means key class label.
