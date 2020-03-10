# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:08:57 2020

@author: egarciamarcos
"""
import os
import glob
import numpy as np
from skimage.io import imsave, imread

baseDir = 'C:\\Users\\egarciamarcos\\Desktop\\Run3\\HE-CK-645\\Pruebas\\Cores'
lista = glob.glob(os.path.join(baseDir,'Images','*.png'))

tumorDir = os.path.join(baseDir,'Tumor')
necrosisDir = os.path.join(baseDir,'Necrosis')
stromaDir = os.path.join(baseDir,'Stroma')

imageSize = 2560
for nameImage in lista:
    finalImage=np.zeros((imageSize,imageSize), dtype=np.uint8)
    
    name = os.path.basename(nameImage)
    if os.path.exists(os.path.join(tumorDir,name[:-4]+'-Tumor.png')):
        finalImage = np.fmax(finalImage, np.array(imread(os.path.join(tumorDir,name[:-4]+'-Tumor.png'))/255,dtype = np.uint8))
    if os.path.exists(os.path.join(stromaDir,name[:-4]+'-Stroma.png')):
        finalImage = np.fmax(finalImage, 2 * np.array( imread(os.path.join(stromaDir,name[:-4]+'-Stroma.png'))/255, dtype=np.uint8))
    if os.path.exists(os.path.join(necrosisDir,name[:-4]+'-Necrosis.png')):
        finalImage = np.fmax(finalImage, 3 * np.array( imread(os.path.join(necrosisDir,name[:-4]+'-Necrosis.png'))/255, dtype = np.uint8))
    
    #finalImage=[255*stroma, 255*tumor, 255*necrosis]
    imsave(os.path.join(baseDir,'Masks',name[:-4]+'-mask.png'), finalImage)
