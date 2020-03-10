# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:07:32 2020

@author: egarciamarcos
"""
import os
import glob
from skimage import io
from skimage.color import rgb2hed
import numpy as np

from skimage.io import imshow

"""
from keras.models import *
# later...
# load json and create model
json_file = open('model-MultiClass-2560.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model-MultiClass-2560.hdf5")
print("Loaded model from disk")
"""

imagesDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores'
lista = glob.glob(os.path.join(imagesDir,'Images','*.png'))

imageSize = 2560
subImageSize = 256

offset=32


id2code = {0: (0, 0, 0),
           1: (255, 0, 0),
           2: (0, 255, 0)}
          #  3: (0, 0, 255)}

"""
def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image
"""

def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) , dtype=np.uint8 )
    for k in colormap.keys():
        output[single_layer==k,:] = colormap[k]
    return output

for i in lista:
    zerosMat = np.zeros((imageSize,imageSize, len(id2code.keys())))
    
    img = io.imread(os.path.join(imagesDir,'Images',i),as_gray = False)
    img = img.astype('float32')
    
    img = rgb2hed(img)
    #img = (img - np.min(img))/(np.max(img)-np.min(img)) *255
    for l in range(int(np.ceil(imageSize/(subImageSize-offset)) +1)):
        for m in range(int(np.ceil(imageSize/(subImageSize-offset)) +1)):
            # subImage = img[l*subImageSize:(l+1)*subImageSize, m*subImageSize:(m+1)*subImageSize,:]
            index_1 = l*(subImageSize-offset)
            index_2 = m*(subImageSize-offset)
            if index_1 + subImageSize < imageSize and index_2 + subImageSize < imageSize:
                subImage = img[index_1:index_1+subImageSize, index_2:index_2+subImageSize,:]
            elif index_1 + subImageSize < imageSize and index_2 + subImageSize > imageSize:
                subImage = img[index_1:index_1+subImageSize, imageSize-subImageSize:imageSize,:]
            elif index_1 + subImageSize > imageSize and index_2 + subImageSize < imageSize:
                subImage = img[imageSize-subImageSize:imageSize,index_2:index_2+subImageSize,:]
            else:
                subImage = img[imageSize-subImageSize:imageSize,imageSize-subImageSize:imageSize,:]
            #img = (img - np.min(img))/(np.max(img)-np.min(img))
    
    
            subImage = np.reshape(subImage,(1,)+subImage.shape)
            res = model.predict(subImage,verbose=0)
            # plt.imshow(res[0,:,:,0],  cmap='gray')
            # plt.show()
            
            
            if index_1 + subImageSize < imageSize and index_2 + subImageSize < imageSize:
                zerosMat[index_1:index_1+subImageSize, index_2:index_2+subImageSize,:] = np.fmax( zerosMat[index_1:index_1+subImageSize, index_2:index_2+subImageSize,:], res[0,:,:,:])
            elif index_1 + subImageSize < imageSize and index_2 + subImageSize > imageSize:
                zerosMat[index_1:index_1+subImageSize, imageSize-subImageSize:imageSize,:] = np.fmax( zerosMat[index_1:index_1+subImageSize, imageSize-subImageSize:imageSize,:], res[0,:,:,:])
            elif index_1 + subImageSize > imageSize and index_2 + subImageSize < imageSize:
                zerosMat[imageSize-subImageSize:imageSize,index_2:index_2+subImageSize, :] = np.fmax( zerosMat[imageSize-subImageSize:imageSize,index_2:index_2+subImageSize,:], res[0,:,:,:])
            else:
                zerosMat[imageSize-subImageSize:imageSize,imageSize-subImageSize:imageSize,:] = np.fmax( zerosMat[imageSize-subImageSize:imageSize,imageSize-subImageSize:imageSize,:], res[0,:,:,:])
            
            # zerosMat[l*subImageSize:(l+1)*subImageSize, m*subImageSize:(m+1)*subImageSize] = np.maximum( zerosMat[l*subImageSize:(l+1)*subImageSize, m*subImageSize:(m+1)*subImageSize], res[0,:,:,0])
                
            
    output = onehot_to_rgb(zerosMat, id2code)
    io.imsave(os.path.join(imagesDir,os.path.basename(i)),output)
    # break
    