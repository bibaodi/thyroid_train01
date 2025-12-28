#!/usr/bin/env python

import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import kld
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers
from tensorflow.python.tools import freeze_graph

from uisbp.data import DataLoader, DataSet, get_video_dict, get_split_ids
from uisbp.train.segmentation.model import Unet, linknet18, linknet34
from uisbp.train.segmentation.learner import Learner
from uisbp.train.segmentation.metrics import (mean_dice_coef, mean_jaccard_coef, CrossEntropyJaccardLoss, MeanDice)
from uisbp.augmentations import (Augmentation, elastic_transform, random_flip_lr, random_zoom, random_rotation,
                                 random_shear, random_shift, random_gain)
from uisbp.transform_utils import adjust_hist


def flip_data(X, Y, ids):
    flip_bool, Xf, Yf = [], [], []
    for ct, (x, y, ix) in enumerate(zip(X, Y, ids)):
        if 'BP_L' in ix:
            Xf.append(np.fliplr(x).copy())
            Yf.append(np.fliplr(y).copy())
            flip_bool.append(1)
        else:
            Xf.append(x)
            Yf.append(y)
            flip_bool.append(0)
    return np.array(Xf), np.array(Yf), np.array(flip_bool, dtype=np.bool)


def init_parser():
    parser = argparse.ArgumentParser(description='Train Nerve segmentation algorithm')

    parser.add_argument('--datapath',
                        dest='datapath',
                        type=str,
                        required=True,
                        help='Path to folder where numpy data is stored.')
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default='apr',
                        help='Dataset name. Valid options are [apr, may, multi, binary]')
    parser.add_argument('--model',
                        dest='model',
                        type=str,
                        default='linknet18',
                        help='Model architecture to use [unet, linknet18, linknet34]')
    parser.add_argument('--start_channel_depth',
                        dest='start_channel_depth',
                        type=int,
                        default=32,
                        help='Starting channel depth for encoder [default 32]')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5, help='Dropout in encoder [default 0.5]')
    parser.add_argument('--size', dest='size', type=int, default=96, help='Square image size to train on.')
    parser.add_argument('--outdir',
                        dest='outdir',
                        type=str,
                        default=f'.{os.sep}results{os.sep}',
                        help='Output directory for saved models and any other outputs.')
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=898,
                        help='random number seed for train/validation split.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Training Batch Size.')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        type=float,
                        default=5e-3,
                        help='Max Learning rate for use in SGDR')
    parser.add_argument('--checkpoint',
                        dest='checkpoint',
                        type=str,
                        default=None,
                        help='Full path to checkpoint file if initializing from another run.')
    parser.add_argument('--use_binary_output',
                        dest='use_binary_output',
                        action='store_true',
                        help='Use binary output for BP even though model was trained with multi-class input')
    parser.add_argument('--flip_data',
                        dest='flip_data',
                        action='store_true',
                        help='Option to orient all images so that they are right-oriented during training')
    parser.add_argument('--channels_first_output',
                        dest='channels_first_output',
                        action='store_true',
                        help='Option to output model in channels-first format')
    parser.add_argument('--combine_train_valid',
                        dest='combine_train_valid',
                        action='store_true',
                        help='Option to combine the training and validation data to create a larger training set.')
    parser.add_argument('--gpuid', dest='gpuid', type=int, default=0, help='Which GPU device to use [default 0]')
    parser.add_argument('--keylabel', dest='keylabel', type=str, required=True, help='Key class label of segmentation.')
    parser.add_argument('--use_augmentation',
                        dest='use_augmentation',
                        action='store_true',
                        help='Option to use augmentation during training')
    return parser


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    if args.size not in [96, 128, 160, 224, 416, 448]:
        raise ValueError('Input size must be in [96, 128, 160, 224, 416] for segmentation')
    # get data
    dl = DataLoader(args.datapath, dataset=args.dataset)
    filter_images = True if args.dataset in ['apr', 'binary'] else False
    X_train, Y_train, ids_train, X_valid, Y_valid, ids_valid, X_test, Y_test, ids_test = dl.get_train_valid_split(
        size=args.size, seed=args.seed, filter_nerve_images=filter_images)
    labels = dl.labels

    # if only have a few videos, combine all data and split again
    ids = np.concatenate((ids_train, ids_valid, ids_test))
    vdict = get_video_dict(ids)
    if len(vdict) < 10:
        print(f'Only have {len(vdict)} total videos. Combining all data for new train/test split')
        ids = np.concatenate((ids_train, ids_valid, ids_test))
        x = np.concatenate((X_train, X_valid, X_test))
        y = np.concatenate((Y_train, Y_valid, Y_test))

        train_ids, test_ids = get_split_ids(ids, 0.2, seed=args.seed)
        valid_ids = test_ids

        X_train, Y_train, ids_train = x[train_ids], y[train_ids], ids[train_ids]
        X_valid, Y_valid, ids_valid = x[valid_ids], y[valid_ids], ids[valid_ids]
        X_test, Y_test, ids_test = x[test_ids], y[test_ids], ids[test_ids]

    if args.combine_train_valid:
        print('Combining traning and validation set')
        X_train = np.concatenate((X_train, X_valid))
        Y_train = np.concatenate((Y_train, Y_valid))
        ids_train = np.concatenate((ids_train, ids_valid))

        X_valid, Y_valid, ids_valid = X_test, Y_test, ids_test

    train_vdict = get_video_dict(ids_train)
    test_vdict = get_video_dict(ids_test)
    valid_vdict = get_video_dict(ids_valid)

    print(f'Training video names: {", ".join(list(train_vdict.keys()))}')
    print(f'Testing video names: {", ".join(list(test_vdict.keys()))}')
    print(f'Validation video names: {", ".join(list(valid_vdict.keys()))}')

    print(f'Training data size: {X_train.shape}')
    print(f'Validation data size: {X_valid.shape}')
    print(f'Testing data size: {X_test.shape}')
    if labels is not None:
        print(f'Class labels: {labels}')

    # set up data augmentation
    if args.use_augmentation:
        augs = tuple([
            #Augmentation(fn=elastic_transform, kwargs={'sigma':15, 'alpha':50}),  # comment for BP by duke, because have duplicate movement with below
            Augmentation(fn=random_flip_lr, kwargs={'p': 0.0 if args.flip_data else 0.5}),  # comment for BP for BP数据不够多，而且左右不均衡。简单粗暴的处理方式，就是把BP强扭成一个方向，然后训练。现在BP的模型左右效果是有明显差距的。BP的预处理有一步做了L2R，如果augmentation 再做flip ，就矛盾了。
            Augmentation(fn=random_zoom, kwargs={
                'zoom_range': (0.7, 1.3),
                'fill_mode': 'nearest'
            }),
            Augmentation(fn=random_rotation, kwargs={
                'rg': 15.0,
                'fill_mode': 'nearest'
            }),
            Augmentation(fn=random_shear, kwargs={
                'intensity': 0.5,
                'fill_mode': 'nearest'
            }),
            Augmentation(fn=random_shift, kwargs={
                'wrg': 0.2,
                'hrg': 0.2,
                'fill_mode': 'nearest'
            }),
            Augmentation(fn=random_gain, kwargs={'gain_range': (0.3, 2.5)})
        ])
    else:
        augs = None

    # set up data iterators and data set
    preprocess_fn = lambda x, y: (adjust_hist(x), y)
    #def preprocess_fn(x, y):  # this lead to the train can not be converged, so must define the function as a lambda ? need to test
    #    return (adjust_hist(x), y)

    if args.flip_data:
        print('Flipping all images to be right-oriented')  # due to there is not enough BP data of both side
        assert args.keylabel == 'BP'
        X_train_flip, Y_train_flip, train_flip_bool = flip_data(X_train, Y_train, ids_train)
        X_valid_flip, Y_valid_flip, valid_flip_bool = flip_data(X_valid, Y_valid, ids_valid)
        X_test_flip, Y_test_flip, test_flip_bool = flip_data(X_test, Y_test, ids_test)

        ds = DataSet((X_train_flip, Y_train_flip), (X_valid_flip, Y_valid_flip),
                     batch_size=args.batch_size,
                     test=(X_test_flip, Y_test_flip),
                     preprocess_fn=preprocess_fn,
                     augs=augs)
    else:
        ds = DataSet((X_train, Y_train), (X_valid, Y_valid),
                     batch_size=args.batch_size,
                     test=(X_test, Y_test),
                     preprocess_fn=preprocess_fn,
                     augs=augs)

    # set up model and learner
    if args.model in ['linknet18', 'linknet34']:

        if args.model == 'linknet18':
            arch = linknet18((args.size, args.size, 1),
                             start_ch=args.start_channel_depth,
                             dropout=args.dropout,
                             out_ch=Y_train.shape[-1])
        else:
            arch = linknet34((args.size, args.size, 1),
                             start_ch=args.start_channel_depth,
                             dropout=args.dropout,
                             out_ch=Y_train.shape[-1])

        outdir = f'{args.outdir}{os.sep}{args.model}_{args.start_channel_depth}_{args.dropout}_{args.dataset}_{args.size}'

    elif args.model == 'unet':
        dropout = [args.dropout] * 5 + [0] * 4
        arch = Unet((args.size, args.size, 1),
                    start_ch=args.start_channel_depth,
                    inc_rate=2,
                    dropouts=dropout,
                    residual=True,
                    maxpool=False,
                    use_shortcut=True,
                    out_ch=Y_train.shape[-1])
        outdir = f'{args.outdir}{os.sep}{args.model}_{args.start_channel_depth}_{args.dropout}_{args.dataset}_{args.size}'

    else:
        raise ValueError(f'Unknown model {args.model}.')
    print(f'Using {args.model} model')

    # get class weights
    if args.dataset in ['apr', 'binary']:
        Y_train[np.nonzero(Y_train)] = 1
        weights = np.array([np.sum(Y_train == 1), np.sum(Y_train == 0)])
        weights = 1 / weights
        weights = weights / np.sum(weights)
    elif args.dataset in ['may', 'multi']:
        class_weights = {label: np.sum(Y_train[:, :, :, ct], axis=(0, 1, 2)) for ct, label in enumerate(labels)}
        total = np.sum(list(class_weights.values()))
        for key, val in class_weights.items():
            if val == 0:
                print(f'Warning: Class {key} is not in training set!')
            class_weights[key] = max(val, 1) / total
        weights = 1 / np.array(list(class_weights.values()))
        weights /= weights.sum()
    else:
        raise ValueError(f'Unknown dataset type {args.dataset}')

    # setup loss
    loss = CrossEntropyJaccardLoss(jaccard_weight=1.0,
                                   class_weights=weights,
                                   num_classes=Y_train.shape[-1],
                                   bp_index=list(labels).index(args.keylabel),
                                   bp_weight=2)

    # setup learner
    print(f'Will save results to {outdir}')
    learn = Learner(arch, ds, loss=loss, outdir=outdir, opt_fn=Adam(lr=1e-4, beta_1=0.95))

    if args.checkpoint is not None:
        print(f'Loading weights from {args.checkpoint}')
        learn.load(args.checkpoint, weights_only=True)

    if args.dataset in ['apr', 'binary']:
        metrics = (mean_dice_coef, mean_jaccard_coef, kld)
        monitor = 'val_mean_dice_coef'
    else:
        metrics = tuple([MeanDice(labels, lab) for lab in labels[:-1]])
        monitor = 'val_mean_dice_' + args.keylabel.lower()

    # warmup
    learn.fit(1e-5, n_cycle=1, cycle_len=1, metrics=metrics, monitor=(monitor, 'max'))

    # Initial training default n_cycle=20, cycle_len=2
    learn.fit(args.learning_rate, n_cycle=20, cycle_len=2, metrics=metrics, monitor=(monitor, 'max'))

    # fine tuning default n_cycle=4, cycle_len=1, cycle_mult=3
    learn.fit(args.learning_rate/5, n_cycle=4, cycle_len=1, cycle_mult=3, metrics=metrics, monitor=(monitor, 'max'))

    # load in best weights and write to pb file
    learn.load(f'{outdir}{os.sep}weights.h5', weights_only=True)
    model = learn.model

    #ignore test data procedure --200630
    if 0:
        # save predictions
        ypred = model.predict_generator(ds.test_dl, steps=len(ds.test_dl))
        if args.flip_data:
            ypred[test_flip_bool] = ypred[test_flip_bool][:, :, ::-1, :]
        np.save(f'{outdir}{os.sep}ypred.npy', ypred)
        np.save(f'{outdir}{os.sep}ytest.npy', Y_test)

    output_extras = ''
    # option to output binary model for BP even though multi-class model was trained
    if args.use_binary_output and args.dataset in ['may', 'multi']:
        output_extras += 'binary'
        print('Modifying model from multi-label to binary output')
        slice_layer = keras.layers.Lambda(lambda x: tf.expand_dims(x[..., list(labels).index(args.keylabel)], axis=-1))
        tmp_model = Model(inputs=model.input, outputs=slice_layer(model.output))
        tmp_model.set_weights(model.get_weights())
        model = tmp_model

    # make output channels-first
    if args.channels_first_output:
        output_extras += 'channels_first'
        print('Modifying model from channels-last to channels-first')
        reshape_layer = keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))
        tmp_model = Model(inputs=model.input, outputs=reshape_layer(model.output))
        tmp_model.set_weights(model.get_weights())
        model = tmp_model

    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)

    # Serialize the model and get its weights, for quick re-building.
    config = model.get_config()
    weights = model.get_weights()

    # Re-build a model where the learning phase is now hard-coded to 0.
    new_model = Model.from_config(config, custom_objects={'tf': tf, 'labels': labels})
    new_model.set_weights(weights)

    temp_dir = outdir
    print(f'Saving output graphs to {temp_dir}')
    model_file_h5 = learn.save('linknet_model')
    print(f"saved hdf5 file is: {model_file_h5}")

    print("finish....................")

    #checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
    #ckpt = tf.train.Checkpoint(optimizer=learn.opt_fn, model=learn.model)
    #checkpoint_path = ckpt.save(file_prefix=checkpoint_prefix)
    #print("ckpt_save_ret", checkpoint_path)
