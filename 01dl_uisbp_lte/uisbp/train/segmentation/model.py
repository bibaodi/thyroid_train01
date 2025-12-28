#from keras.models import Model
#from keras.layers import Input, Dense, Flatten, BatchNormalization, Concatenate, add, ZeroPadding2D
#from keras.layers.core import Dropout, Lambda, Activation
#from keras.layers.convolutional import Convolution2D, Conv2DTranspose
#from keras.layers.pooling import MaxPooling2D
#----------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import concatenate, add, ZeroPadding2D, Dropout, Lambda, Activation
from tensorflow.keras.layers import Convolution2D, Convolution2DTranspose
from tensorflow.keras.layers import MaxPooling2D

from .metrics import *


def shortcut(_input, residual):

    input_shape = K.int_shape(_input)
    residual_shape = K.int_shape(residual)
    stride_width = input_shape[1] // residual_shape[1]
    stride_height = input_shape[2] // residual_shape[2]
    equal_channels = residual_shape[3] == input_shape[3]

    shortcut = _input

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(residual_shape[3],
                          1,
                          strides=(stride_width, stride_height))(_input)

    return add([shortcut, residual])


##### U-net model #####


def conv_block(m, dim, activation, batchnorm, residual, dropout=0):
    n = Convolution2D(dim, 3, padding='same')(m)
    n = BatchNormalization()(n) if batchnorm else n
    n = Activation(activation)(n)
    n = Dropout(dropout)(n) if dropout else n
    n = Convolution2D(dim, 3, padding='same')(n)
    n = BatchNormalization()(n) if batchnorm else n
    return Activation(activation)(shortcut(
        n, m)) if residual else Activation(activation)(n)


def down_sample(m, dim, activation, batchnorm, maxpool=False):
    if maxpool:
        out = MaxPooling2D((2, 2))(m)
    else:
        n = Convolution2D(dim, 2, strides=(2, 2), padding='same')(m)
        n = BatchNormalization()(n) if batchnorm else n
        out = Activation(activation)(n)
    return out


def deconv_block(m,
                 c,
                 dim,
                 activation,
                 batchnorm,
                 residual,
                 use_shortcut=True,
                 dropout=0):
    n = Convolution2DTranspose(dim,
                        2,
                        strides=2,
                        activation=activation,
                        padding='same')(m)
    if use_shortcut:
        n = shortcut(n, c)
    else:
        n = concatenate()([n, c])
    n = conv_block(n,
                   dim,
                   activation,
                   batchnorm,
                   residual=residual,
                   dropout=dropout)
    return n


def Unet(img_shape,
         out_ch=1,
         start_ch=64,
         inc_rate=2,
         activation='elu',
         dropouts=0.5,
         batchnorm=True,
         residual=False,
         aux_output=False,
         maxpool=True,
         use_shortcut=False):

    if isinstance(dropouts, float):
        dropouts = [dropouts] * 9

    inputs = Input(shape=img_shape)
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = conv_block(s, start_ch, activation, batchnorm, residual, dropouts[0])
    p1 = down_sample(c1, start_ch, activation, batchnorm, maxpool)

    c2 = conv_block(p1, start_ch * inc_rate, activation, batchnorm, residual,
                    dropouts[1])
    p2 = down_sample(c2, start_ch * inc_rate, activation, batchnorm, maxpool)

    c3 = conv_block(p2, start_ch * inc_rate**2, activation, batchnorm,
                    residual, dropouts[2])
    p3 = down_sample(c3, start_ch * inc_rate**2, activation, batchnorm,
                     maxpool)

    c4 = conv_block(p3, start_ch * inc_rate**3, activation, batchnorm,
                    residual, dropouts[3])
    p4 = down_sample(c4, start_ch * inc_rate**3, activation, batchnorm,
                     maxpool)

    bottom = conv_block(p4, start_ch * inc_rate**4, activation, batchnorm,
                        residual, dropouts[4])

    if aux_output:
        out = Convolution2D(1, (1, 1), activation=activation)(bottom)
        auxiliary_output = Dense(
            1,
            activation='sigmoid',
            name='aux_output',
        )(Flatten()(out))

    u6 = deconv_block(bottom,
                      c4,
                      start_ch * inc_rate**3,
                      activation,
                      batchnorm,
                      residual,
                      dropout=dropouts[5],
                      use_shortcut=use_shortcut)

    u7 = deconv_block(u6,
                      c3,
                      start_ch * inc_rate**2,
                      activation,
                      batchnorm,
                      residual,
                      dropout=dropouts[6],
                      use_shortcut=use_shortcut)

    u8 = deconv_block(u7,
                      c2,
                      start_ch * inc_rate,
                      activation,
                      batchnorm,
                      residual,
                      dropout=dropouts[7],
                      use_shortcut=use_shortcut)

    u9 = deconv_block(u8,
                      c1,
                      start_ch,
                      activation,
                      batchnorm,
                      residual,
                      dropout=dropouts[8],
                      use_shortcut=use_shortcut)

    if out_ch == 1:
        output = Convolution2D(out_ch, (1, 1),
                        activation='sigmoid',
                        name='main_output')(u9)
    elif out_ch > 1:
        output = Convolution2D(out_ch, (1, 1),
                        activation='softmax',
                        name='main_output')(u9)
    else:
        raise ValueError(f'out_ch = {out_ch} value not allowed.')

    outputs = [output, auxiliary_output] if aux_output else [output]
    model = Model(inputs=[inputs], outputs=outputs)

    return model


##### LinkNet #####


def conv_bn(input_tensor, kernel_size, filters, strides=(1, 1),
            padding='same'):
    x = Convolution2D(filters, kernel_size, strides=strides,
               padding=padding)(input_tensor)
    x = BatchNormalization()(x)
    return x


def residual_block(input_tensor,
                   kernel_size,
                   filters,
                   init_strides=(1, 1),
                   dropout=0):

    x = conv_bn(input_tensor, kernel_size, filters, strides=init_strides)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x
    x = conv_bn(x, kernel_size, filters, strides=(1, 1))

    return Activation('relu')(shortcut(input_tensor, x))


def encoder_block(input_tensor,
                  filters,
                  layers=2,
                  dropout=0,
                  init_strides=(1, 1)):

    for layer in range(layers):
        if layer == 0:
            strides = init_strides
            x = residual_block(input_tensor, (3, 3),
                               filters,
                               init_strides=strides,
                               dropout=dropout)
        else:
            strides = (1, 1)
            x = residual_block(x, (3, 3),
                               filters,
                               init_strides=strides,
                               dropout=dropout)

    return x


def decoder_block(input_tensor, filters):

    x = conv_bn(input_tensor, (1, 1), filters // 4, padding='valid')
    x = Activation('relu')(x)

    #x = ZeroPadding2D(padding=(1,1))(x)
    x = Convolution2DTranspose(filters // 4, (4, 4), strides=(2, 2),
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = conv_bn(x, (1, 1), filters, padding='valid')
    x = Activation('relu')(x)

    return x


def LinkNet(img_shape, start_ch=64, layers=(2, 2, 2, 2), dropout=0, out_ch=1):

    inputs = Input(shape=img_shape)
    x = Lambda(lambda x: x / 255)(inputs)

    # input layer
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Convolution2D(start_ch, (7, 7), strides=(2, 2), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # encoder layers
    e1 = encoder_block(x, start_ch, layers=layers[0], dropout=dropout)
    e2 = encoder_block(e1,
                       start_ch * 2,
                       layers=layers[1],
                       dropout=dropout,
                       init_strides=(2, 2))
    e3 = encoder_block(e2,
                       start_ch * 4,
                       layers=layers[2],
                       dropout=dropout,
                       init_strides=(2, 2))
    e4 = encoder_block(e3,
                       start_ch * 8,
                       layers=layers[3],
                       dropout=dropout,
                       init_strides=(2, 2))

    # decoder layers
    d4 = add([decoder_block(e4, start_ch * 4), e3])
    d3 = add([decoder_block(d4, start_ch * 2), e2])
    d2 = add([decoder_block(d3, start_ch), e1])
    d1 = decoder_block(d2, start_ch)

    # final decoding layers
    f1 = Convolution2DTranspose(start_ch // 2, (3, 3),
                         strides=(2, 2),
                         activation='relu')(d1)
    f2 = Convolution2D(start_ch // 2, (3, 3), activation='relu')(f1)
    f2 = ZeroPadding2D(padding=(1, 1))(f2)

    if out_ch == 1:
        f3 = Convolution2D(out_ch, (2, 2), activation='sigmoid')(f2)
    elif out_ch > 1:
        f3 = Convolution2D(out_ch, (2, 2), activation='softmax')(f2)
    else:
        raise ValueError(f'out_ch = {out_ch} value not allowed.')

    model = Model(inputs=[inputs], outputs=[f3])

    return model


def linknet18(img_shape, start_ch=32, dropout=0, out_ch=1):
    """Linknet model using modified Resnet 18 as encoder"""
    return LinkNet(img_shape,
                   start_ch=start_ch,
                   layers=(2, 2, 2, 2),
                   dropout=dropout,
                   out_ch=out_ch)


def linknet34(img_shape, start_ch=32, dropout=0, out_ch=1):
    """Linknet model using modified Resnet 34 as encoder"""
    return LinkNet(img_shape,
                   start_ch=start_ch,
                   layers=(3, 4, 6, 3),
                   dropout=dropout,
                   out_ch=out_ch)
