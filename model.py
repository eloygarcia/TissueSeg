# import numpy as np 

# import os
# import skimage.io as io
# import skimage.transform as trans

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

"""
Here is a dice loss for keras which is smoothed to approximate a linear (L1) loss.
It ranges from 1 to 0 (no error), and returns results similar to binary crossentropy
"""
# define custom loss and metric functions 
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
#    
## Test
#y_true = np.array([[0,0,1,0],[0,0,1,0],[0,0,1.,0.]])
#y_pred = np.array([[0,0,0.9,0],[0,0,0.1,0],[1,1,0.1,1.]])
#
#r = dice_coef_loss(
#    K.theano.shared(y_true),
#    K.theano.shared(y_pred),
#).eval()
#print('dice_coef_loss',r)
#
#
#r = keras.objectives.binary_crossentropy(
#    K.theano.shared(y_true),
#    K.theano.shared(y_pred),
#).eval()
#print('binary_crossentropy',r)
#print('binary_crossentropy_scaled',r/r.max())
## TYPE                 |Almost_right |half right |all_wrong
## dice_coef_loss      [ 0.00355872    0.40298507  0.76047904]
## binary_crossentropy [ 0.0263402     0.57564635  12.53243514]

def lr_scheduler(epoch, mode='power_decay'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    print('lr: %f' % lr)
    return lr
    
def unet(pretrained_weights = None,input_size = (256,256,1), numClasses=1, initialFilter = 64):
    
    scheduler = LearningRateScheduler(lr_scheduler)
    inputs = Input(input_size)
 
    #conv0 = Conv2D(64, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv0 = Conv2D(64, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    #pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = Conv2D(initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(2*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(4*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(8*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(16*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(16*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.1)(conv5)

    up6 = Conv2D(8*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(8*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(8*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.1)(conv6)

    up7 = Conv2D(8*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(4*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(4*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.1)(conv7)

    up8 = Conv2D(4*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(2*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(2*initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.1)(conv8)

    up9 = Conv2D(initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(initialFilter, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = BatchNormalization()(conv9)
    # conv9 = Conv2D(2, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    #up10 = Conv2D(64, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
   
    #merge10 = concatenate([conv0,up10], axis = 3)
    #conv10 = Conv2D(64, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    #conv10 = Conv2D(64, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    #conv10 = Conv2D(2, (5,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    
    # conv10 = Conv2D(1, (1,1), activation = None)(conv9)
    # conv10 = Conv2D(1, (1,1), activation = 'sigmoid')(conv9)
    conv10 = Conv2D(numClasses, (1,1), activation = 'softmax', padding='same')(conv9) ## cUIDADO QUE HAS CAMBIADO ESTO

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1.e-4, beta_1 = 0.9, beta_2 = 0.99, amsgrad=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # OK-2 model.compile(optimizer = Adam(lr = 5e-5, beta_1 = 0.9, beta_2 = 0.5, amsgrad=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # OK-1 model.compile(optimizer = Adam(lr = 1e-4, beta_1 = 0.8, beta_2 = 0.5, amsgrad=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    
    # model.compile(optimizer = Adam(lr = 1e-4, beta_1 = 0.99, beta_2 = 0.99, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    # model.compile(optimizer = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True), loss ='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # model.compile(optimizer = Adam(lr = 1e-4, beta_1 = 0.9, beta_2 = 0.9, amsgrad=True), loss = dice_coef_loss, metrics = ['accuracy'])
    ############## keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


