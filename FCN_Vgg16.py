# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:55:06 2019

@author: egarciamarcos
"""

from LossFunctions import *

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# from get_weights_path import *
# from resnet_helpers import *

def FCN_Vgg16(pretrained_weights = None,input_size = (512,512,1),numClasses=1):
    img_input = Input(shape=input_size)
    """ Encoder Block """
    # Block 1
    conv1a = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    conv1b = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1a)
    add1 = Add()([conv1a,conv1b])
    drop1 = Dropout(0.05,name='drop1')(add1)
    # pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(drop1)

    # Block 2
    conv2a = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv2b = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2a)
    add2 = Add()([conv2a,conv2b])
    drop2 = Dropout(0.05, name = 'drop2')(add2)
    # pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(drop2)

    # Block 3
    conv3a = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv3b = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3a)
    add3 = Add()([conv3a,conv3b])
    #conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
    drop3 = Dropout(0.05,name = 'drop3')(add3)
    # pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)
    pool3 = MaxPooling2D((2, 2), name='block3_pool')(drop3)

    # Block 4
    conv4a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv4b = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4a)
    add4 = Add()([conv4a,conv4b])
    #conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
    drop4 = Dropout(0.05,name='drop4')(add4)
    # pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)
    pool4 = MaxPooling2D((2, 2), name='block4_pool')(drop4)
    
    """ minimum point """
    # Block 5
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5)
    #conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5)
    drop5 = Dropout(0.05, name='drop5')(conv5)
    #pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    # pool5 = MaxPooling2D((2, 2), name='block5_pool')(conv5)

    """ Decoder Block """
    # Block 6
    up6 = UpSampling2D(size = (2,2), name='upsampling6')(drop5)
    # conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(conv6)
    drop6 = Dropout(0.05, name='drop6')(conv6)
    
    # Block 7
    up7 = UpSampling2D(size = (2,2), name='upsampling7')(drop6)
    # conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv1')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv2')(conv7)
    drop7 = Dropout(0.05, name='drop7')(conv7)
    
    # Block 8
    up8 = UpSampling2D(size = (2,2), name='upsampling8')(drop7)
    # conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv1')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv2')(conv8)
    drop8 = Dropout(0.05, name='drop8')(conv8)
    
    # Block 9
    up9 = UpSampling2D(size = (2,2), name='upsampling9')(drop8)
    # conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv2')(conv9)
    drop9 = Dropout(0.05, name='drop9')(conv9)
    
    # Convolutional layers transfered from fully-connected layers
    # x = Conv2D(256, (7, 7), activation='relu', padding='same', name='fc1')(x)
    # x = Conv2D(256, (1, 1), activation='relu', padding='same', name='fc2')(x)
    # x = Conv2D(1, (1, 1), activation='linear', name='predictions_1000')(x)
    #x = Reshape((7,7))(x)

    """ Final Layer """
    # conv10 = Conv2D(1, (1,1), activation = None)(conv9)
    # conv10 = Conv2D(1, (1,1), activation = 'sigmoid')(conv9) # con binary_crossentropy
    conv10 = Conv2D(numClasses, (3,3), activation = 'softmax', padding='same')(conv9) # con categorical_crossentropy
    
    # Create model
    model = Model(img_input, conv10)

    #model.compile(optimizer = Adam(lr = 1e-4, beta_1 = 0.9, beta_2 = 0.99, amsgrad=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 5e-5, beta_1 = 0.9, beta_2 = 0.99, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    # model.compile(optimizer = Adam(amsgrad=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # model.compile(optimizer = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True), loss ='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True), loss ='categorical_crossentropy', metrics=['categorical_accuracy'])

    
    # model_loss = discriminative_loss(delta_v=5e3, delta_d=20e3, gamma=100)
    # model.compile(optimizer = Adam(amsgrad=True), loss = [model_loss] , metrics = ['accuracy'])
    
    # model.compile(optimizer = Adam(lr = 1e-4, beta_1 = 0.9, beta_2 = 0.5), loss = dice_coef_loss, metrics = ['accuracy'])
    ############## keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
