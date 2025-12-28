import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Input, Lambda, MaxPooling2D
from tensorflow.python.keras.models import Model


from .keras_layers import NoisyAndLayer


# Universe of optimization methods
OPT_TYPES = ("sgd_momentum", "adam")


#**********************************************************************************************************************************
#                                               Whole image based models
#**********************************************************************************************************************************
# Binary Task
def mil_binary_sz_96(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling            
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)

    c4 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c4)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c4, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_128(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling            
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)

    c4 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)

    c5 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (c4)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c5)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c5, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_160(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling            
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)

    c4 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)

    c5 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (c4)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c5)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c5, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_192(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling  
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)
    
    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (c4)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)
    
    c6 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_224(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling            
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)

    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)

    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)

    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)

    c6 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_256(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling   

    Same model architecture for sz = 224, 256, 288         
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)

    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)

    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)

    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)

    c6 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_288(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling     

    Same model architecture for sz = 224, 256, 288       
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)

    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)

    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)

    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)

    c6 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_320(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling      

    Same model architecture for sz = 320, 352      
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)
        
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)
    
    c7 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7)
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_352(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling 

    Same model architecture for sz = 320, 352               
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)
        
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)
    
    c7 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7)
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_384(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling  

    Same model architecture for sz = 384, 416          
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)
    
    p4 = MaxPooling2D((2, 2), name="maxpool4") (c5)
    
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (p4)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)
    
    c7 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7)
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


def mil_binary_sz_416(input_placeholder, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing binary classification MIL with noisy-and global pooling 

    Same model architecture for sz = 384, 416           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1)
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2)
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3)
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4)
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5)
    
    p4 = MaxPooling2D((2, 2), name="maxpool4") (c5)
    
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (p4)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6)
    
    c7 = Conv2D(1, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7)
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1])     

    # Display summary    
    if give_summary:
        model.summary()

    return model


## Multi-label Task
def mil_multi_label_sz_96(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling    
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())

    c4 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c4)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c4, m1, f1]) 

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_128(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling    
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())

    c4 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())

    c5 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (c4)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c5)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c5, m1, f1]) 

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_160(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling    
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())

    c4 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())

    c5 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (c4)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c5)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c5, m1, f1])   

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_192(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling    
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())
    
    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (c4)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())
    
    c6 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1, f1])  

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_224(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 224, 256, 288           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())

    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())

    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)

    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())

    c6 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1, f1])    

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_256(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 224, 256, 288           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())

    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())

    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)

    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())

    c6 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1, f1]) 

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_288(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 224, 256, 288           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())

    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())

    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)

    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())

    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())

    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)

    c5 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())

    c6 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())

    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c6)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c6, m1, f1])   

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_320(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 320, 352           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())
        
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())
    
    c7 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7, training=K.learning_phase())
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1, f1])     

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_352(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 320, 352         
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())
        
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (c5)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())
    
    c7 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7, training=K.learning_phase())
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1, f1])     

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_384(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 384, 416           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())
    
    p4 = MaxPooling2D((2, 2), name="maxpool4") (c5)
    
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (p4)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())
    
    c7 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7, training=K.learning_phase())
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1, f1])     

    # Display summary     
    if give_summary:
        model.summary()

    return model


def mil_multi_label_sz_416(input_placeholder, num_classes=5, is_training=True, give_summary=True, input_scale=255):
    """
    Args:
        input_placeholder: tf.placeholder
            A placeholder for the input image
        num_classes: int
            The number of classification categories
        is_training: bool, optional
            Whether we are in training or validation mode
        give_summary: bool, optional
            Whether to display keras model summary
        input_scale: float, optional
            Value to use in normalizing input to model
    Returns:
        : Model
            A keras tensorflow model implementing multi-label classification MIL with noisy-and global pooling 

    Same model architecture for sz = 384, 416           
    """
    inputs = Input(tensor=input_placeholder, name="input")

    s = Lambda(lambda x: x/input_scale, name="input_scale")(inputs)

    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv1") (s)
    c1 = Activation('relu', name="act1")(c1)
    c1 = BatchNormalization(axis=-1, name="bn1")(c1, training=K.learning_phase())
    
    p1 = MaxPooling2D((2, 2), name="maxpool1") (c1)
    
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv2") (p1)
    c2 = Activation('relu', name="act2")(c2)
    c2 = BatchNormalization(axis=-1, name="bn2")(c2, training=K.learning_phase())
    
    p2 = MaxPooling2D((2, 2), name="maxpool2") (c2)
    
    c3 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv3") (p2)
    c3 = Activation('relu', name="act3")(c3)
    c3 = BatchNormalization(axis=-1, name="bn3")(c3, training=K.learning_phase())
    
    c4 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv4") (c3)
    c4 = Activation('relu', name="act4")(c4)
    c4 = BatchNormalization(axis=-1, name="bn4")(c4, training=K.learning_phase())
    
    p3 = MaxPooling2D((2, 2), name="maxpool3") (c4)
    
    c5 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='valid', name="conv5") (p3)
    c5 = Activation('relu', name="act5")(c5)
    c5 = BatchNormalization(axis=-1, name="bn5")(c5, training=K.learning_phase())
    
    p4 = MaxPooling2D((2, 2), name="maxpool4") (c5)
    
    c6 = Conv2D(256, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv6") (p4)
    c6 = Activation('relu', name="act6")(c6)
    c6 = BatchNormalization(axis=-1, name="bn6")(c6, training=K.learning_phase())
    
    c7 = Conv2D(num_classes, (1, 1), activation=None, kernel_initializer='he_normal', padding='valid', name="conv7") (c6)
    c7 = BatchNormalization(axis=-1, name="bn7")(c7, training=K.learning_phase())
    
    # MIL global pooling output
    # Assumes input is logit, hence, no activation is applied to its input
    m1 = NoisyAndLayer(True, name='mil_output')(c7)

    # Fully-connected output
    f1 = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros', name='fc_output')(m1)

    # Create Model
    if is_training:
        model = Model(inputs=[inputs], outputs=[m1, f1]) 
    else:
        model = Model(inputs=[inputs], outputs=[c7, m1, f1])

    # Display summary     
    if give_summary:
        model.summary()

    return model


"""
Summaries and training ops
"""
def add_loss_summaries(total_loss, decay=0.9999):
    """
    Args:
        total_loss: float
            Total loss from network
        decay: float
            The moving average decay constant. Lies in [0, 1]
    Returns:
        loss_averages_op: Tensorflow Op
            The op for generating moving averages of losses.

    Generates moving average for all losses and associated summaries for visualizing the performance of the network on tensorboard
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(decay=decay, name='avg')
    named_losses = tf.get_collection("named_losses")
    losses = named_losses
    # print('losses: ', losses)

    # Create shadow copies (moving average) of the losses
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # Same model architecture for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '_raw' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, update_ops, data_size, batch_size, learning_rate, num_epochs_per_decay, learning_rate_decay_factor, 
    opt_method="sgd_momentum", momentum=0.9, moving_average_decay=0.9999, var_list=tf.trainable_variables()):
    """
    Args:
        total_loss: float
            The overall loss function to be optimized
        global_step: int
            The optimization step
        update_ops: list
            The list of internal updates (from Keras model) to be run as part of each training step
        data_size: int
            The total number of training examples
        batch_size: int
            The number of training examples in each batch
        learning_rate: float
            The learning rate
        num_epochs_per_decay: int
            The number of epochs after which to scale the learning rate by learning_rate_decay_factor
        learning_rate_decay_factor: float
            The factor to multiply learning by after a given number of iterations
        opt_method: string
            The optimization algorithm to use. Options are "adam" and "sgd_momentum"
        momentum: float
            The fraction of the update vector of the past time step to be added to the current update vector. Only used when opt_method is "sgd_momentum"
        moving_average_decay: float
            The moving average decay constant. Lies in [0, 1]
        var_list: list
            The list of variables for which a shadow (moving average) copy should be created

    Returns:
        train_op: Tensorflow Op         
            A tensorflow training op
    """
    num_batches_per_epoch = data_size // batch_size
    decay_steps = num_batches_per_epoch * num_epochs_per_decay

    # Decay learning rate based on optimization step (global_step)
    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)

    # Monitor learning rate
    tf.summary.scalar('learning_rate', lr)   
    
    # Create summary (to monitor) for each loss and its moving average.
    # Return average op to ensure the loss averages are computed during optimization
    loss_averages_op = add_loss_summaries(total_loss, decay=moving_average_decay)

    # Compute loss averages before computing gradients
    with tf.control_dependencies([loss_averages_op]):
        if opt_method.lower() not in OPT_TYPES:
            raise ValueError("%s is not a supported optimization method. Options are %s" % (opt_method, OPT_TYPES))
        elif opt_method.lower() == "sgd_momentum":
            opt = tf.train.MomentumOptimizer(lr, momentum)
        elif opt_method.lower() == "adam":
            opt = tf.train.AdamOptimizer(lr)
       
        grads_and_vars = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
    
    # Apply gradients
    apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step = global_step)
    
    # Get histogram of optimization variables and their gradients
    for grad, var in grads_and_vars:
        # Get histogram summary of variables
        tf.summary.histogram(var.op.name, var)
        if grad is not None:
            # Get histogram summary of variable gradients
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)

    # Create shadow variables, and add ops to maintain moving averages of the variables in var_list
    variable_averages_op = ema.apply(var_list)

    with tf.control_dependencies([apply_gradient_op, variable_averages_op] + update_ops):
        train_op = tf.no_op(name = 'train')

    return train_op