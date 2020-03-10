# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:15:02 2020

@author: egarciamarcos
"""

import os
import re
import glob
from skimage import io
from skimage.color import rgb2hed
import numpy as np

import pandas as pd
from scipy.misc import imresize

"""
EVALUACION DE DEEPATHOLOGY
"""
names = []
values = []

imagesDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores'

listaGarazi = glob.glob(os.path.join(imagesDir,'GaraziMasks','*1.png'))
listaMasks = glob.glob(os.path.join(imagesDir,'Masks','*.png')) # Garazi == 1 para tumor

# file = open(os.path.join(imagesDir,'Results.txt'),'w') 
# file.write('Core \t Dice\n') 

tissueValue = 0 # Background
# tissueValue = 1 # Tumor

for i in listaGarazi:
    new_mask = io.imread(i,as_gray = True)
    resized_mask = imresize(new_mask, (2560,2560,3), interp='nearest')
    # new_mask = np.array((new_mask>127), dtype=np.uint8)
    
    name = os.path.basename(i)
    
    match = re.search('_',name)
    core = name[:match.end()]
    # data = name[match.end()+1:-5]
    # data = data.split(',')
    #print(np.unique(new_mask))
    for j in listaMasks:
        name2 = os.path.basename( j )
        match2 = re.search('_', name2)
        core2 = name2[:match2.end()]
        if core == core2:
            #stromaMask = io.imread(os.path.join(imagesDir,'Stroma',name2[:-10]+'-Stroma.png'),as_gray=True)
            #necrosisMask = io.imread(os.path.join(imagesDir,'Necrosis',name2[:-10]+'-Necrosis.png'), as_gray=True)
            #real_mask = (stromaMask + necrosisMask)/255
            
            ####
            real_mask = io.imread(j, as_gray = True)
            """
            Para Background y tumor
            """
            real_mask = np.array(real_mask==tissueValue,dtype=np.float)
            Dice = 2*np.sum(real_mask[resized_mask==tissueValue]) / (np.sum(real_mask) + np.sum(np.array(resized_mask==tissueValue, dtype=np.float)))
            """
            Para el resto
            """
            # real_mask = np.array(real_mask>1,dtype=np.float)
            # Dice = 2*np.sum(real_mask[resized_mask>1]) / (np.sum(real_mask) + np.sum(np.array(resized_mask>1, dtype=np.float)))
            
            # print(Dice)
            if not Dice==0 and not np.isnan(Dice):
                names.append(core)
                values.append(Dice)
                #file.write((name+'\t'+str(Dice)+'\n')) 
                # print(Dice)

dt = {'name':names, 'Dice':values}
data = pd.DataFrame(dt)

#data.boxplot()
#data.plot.bar()

data.sort_values(by='Dice')

trainingSet =[]
trainingList = ['A-1_','A-2_','A-3_', 'B-1_','B-2_','B-3_', 'C-1_','C-2_','C-3_',
                'D-1_','D-2_','D-3_', 'E-1_','E-2_','E-3_', 'F-1_','F-2_','F-3_']
for i in trainingList:
    if not data[data.name.str.contains(i)].Dice.values.size == 0:
        trainingSet.append( data[data.name.str.contains(i)].Dice.values )

validationSet=[]
validationList = ['A-4','A-5','A-6', 'B-4','B-5','B-6', 'C-4','C-5','C-6',
                'D-4','D-5','D-6', 'E-4','E-5','E-6', 'F-4','F-5','F-6']
for i in validationList:
    if not data[data.name.str.contains(i)].Dice.values.size == 0:
        validationSet.append( data[data.name.str.contains(i)].Dice.values )

testSet=[]
testList = ['A-7', 'A-8', 'A-9', 'A-10', 'A-11', 'A-12', 'A-13', 'A-14',
            'B-7', 'B-8', 'B-9', 'B-10', 'B-11', 'B-12', 'B-13', 'B-14'
            'C-7', 'C-8', 'C-9', 'C-10', 'C-11', 'C-12', 'C-13', 'C-14'
            'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12', 'D-13', 'D-14'
            'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'E-14'
            'F-7', 'F-8', 'F-9', 'F-10', 'F-11', 'F-12', 'F-13', 'F-14']
for i in testList:
    if not data[data.name.str.contains(i)].Dice.values.size == 0:
        testSet.append( data[data.name.str.contains(i)].Dice.values )

import matplotlib.pyplot as plt

plt.boxplot([np.array(trainingSet),np.array(validationSet),np.array(testSet)])
plt.xticks([1, 2, 3], ['Training', 'Validation', 'Test'])
plt.ylabel('Dice')
plt.show()

### Para buscar un core
# dt[dt.name.str.contains('A-1_')]
# file.close()


"""
A partir de aquí va aparte
"""

id2code = {0: (0, 0, 0),
           1: (255, 0, 0),
           2: (0, 255, 0),
           3: (0, 0, 255)}

# imagen=io.imread(os.path.join(imagesDir,'Masks','A-1_(1.00,2759,7266,2560,2560)-mask.png'),as_gray=True)

def colormask(image, colormap = id2code):
    output = np.zeros( image.shape[:2]+(3,) , dtype=np.uint8 )
    for k in colormap.keys():
        output[image==k,:] = colormap[k]
    return output

def rgb_to_singleLayer(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:] = encoded_image + i*( np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2]))
    return encoded_image

"""
Guardar Mascaras en color
"""

listaMasks = glob.glob(os.path.join(imagesDir,'Masks','*.png')) 
for l in listaMasks:
    im = io.imread(l)
    color_im = colormask(im)
    io.imsave(os.path.join(imagesDir,'Masks','Color',os.path.basename(l)),color_im)

for i in listaGarazi:
    new_mask = io.imread(i,as_gray = True)
    new_mask[new_mask>2] = 2
    color_im = colormask(new_mask)
    io.imsave(os.path.join(imagesDir,'GaraziMasks','Color',os.path.basename(i)),color_im)

"""
Resultados y Boxplots
"""
    
names = []
valuesBackground = []
valuesTumor = []
valuesStroma = []
valuesNecrosis = []

imagesDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores'

listaUnet = glob.glob(os.path.join(imagesDir,'*.png'))
listaMasks = glob.glob(os.path.join(imagesDir,'Masks','Color','*.png')) 

tissueValue = 1 # Tumor

for i in listaUnet:
    new_mask = io.imread(i,as_gray = False)
    new_mask = rgb_to_singleLayer(new_mask)
    # resized_mask = imresize(new_mask, (2560,2560,3), interp='nearest')
    # new_mask = np.array((new_mask>127), dtype=np.uint8)
    
    name = os.path.basename(i)
    
    match = re.search('_',name)
    core = name[:match.end()]
    # data = name[match.end()+1:-5]
    # data = data.split(',')
    #print(np.unique(new_mask))
    for j in listaMasks:
        name2 = os.path.basename( j )
        match2 = re.search('_', name2)
        core2 = name2[:match2.end()]
        if core == core2:
            real_mask = io.imread(j, as_gray = False)
            real_mask = rgb_to_singleLayer(real_mask)
            
            names.append(core)
            
            for tissueValue in range(4):
                temp_mask = np.array(real_mask==tissueValue,dtype=np.float)
                Dice = 2*np.sum(temp_mask[new_mask==tissueValue]) / (np.sum(temp_mask) + np.sum(np.array(new_mask==tissueValue, dtype=np.float)))
                
                if np.isnan(Dice):
                    Dice = 0
                    
                if tissueValue==0:
                    valuesBackground.append( Dice )
                elif tissueValue==1:
                    valuesTumor.append( Dice )
                elif tissueValue==2:
                    valuesStroma.append( Dice )
                elif tissueValue==3:
                    valuesNecrosis.append( Dice )
                else:
                    print(" Ya la has vuelto a joder! ")
                        

dt = {'name':names, 'DiceBackground':valuesBackground, 'DiceTumor':valuesTumor, 'DiceStroma':valuesStroma, 'DiceNecrosis':valuesNecrosis}
data = pd.DataFrame(dt)

#data.boxplot()
#data.plot.bar()
# data.sort_values(by='Dice')

trainingSetBackground =[]
trainingSetTumor =[]
trainingSetNecrosis =[]
trainingSetStroma =[]

trainingList = ['A-1_','A-2_','A-3_', 'B-1_','B-2_','B-3_', 'C-1_','C-2_','C-3_',
                'D-1_','D-2_','D-3_', 'E-1_','E-2_','E-3_', 'F-1_','F-2_','F-3_']
for i in trainingList:
    if not data[data.name.str.contains(i)].DiceBackground.values == 0 and not data[data.name.str.contains(i)].DiceBackground.values.size == 0:
        trainingSetBackground.append( data[data.name.str.contains(i)].DiceBackground.values )
    if not data[data.name.str.contains(i)].DiceTumor.values == 0 and not data[data.name.str.contains(i)].DiceTumor.values.size == 0:
        trainingSetTumor.append( data[data.name.str.contains(i)].DiceTumor.values )
    if not data[data.name.str.contains(i)].DiceStroma.values == 0 and not data[data.name.str.contains(i)].DiceStroma.values.size == 0:
        trainingSetStroma.append( data[data.name.str.contains(i)].DiceStroma.values )
    if not data[data.name.str.contains(i)].DiceNecrosis.values == 0 and not data[data.name.str.contains(i)].DiceNecrosis.values.size == 0:
        trainingSetNecrosis.append( data[data.name.str.contains(i)].DiceNecrosis.values )

validationSetBackground =[]
validationSetTumor =[]
validationSetNecrosis =[]
validationSetStroma =[]
validationList = ['A-4','A-5','A-6', 'B-4','B-5','B-6', 'C-4','C-5','C-6',
                'D-4','D-5','D-6', 'E-4','E-5','E-6', 'F-4','F-5','F-6']
for i in validationList:
    if not data[data.name.str.contains(i)].DiceBackground.values == 0 and not data[data.name.str.contains(i)].DiceBackground.values.size == 0:
        validationSetBackground.append( data[data.name.str.contains(i)].DiceBackground.values )
    if not data[data.name.str.contains(i)].DiceTumor.values == 0 and not data[data.name.str.contains(i)].DiceTumor.values.size == 0:
        validationSetTumor.append( data[data.name.str.contains(i)].DiceTumor.values )
    if not data[data.name.str.contains(i)].DiceStroma.values == 0 and not data[data.name.str.contains(i)].DiceStroma.values.size == 0:
        validationSetStroma.append( data[data.name.str.contains(i)].DiceStroma.values )
    if not data[data.name.str.contains(i)].DiceNecrosis.values == 0 and not data[data.name.str.contains(i)].DiceNecrosis.values.size == 0:
        validationSetNecrosis.append( data[data.name.str.contains(i)].DiceNecrosis.values )

testSetBackground =[]
testSetTumor =[]
testSetNecrosis =[]
testSetStroma =[]
testList = ['A-7', 'A-8', 'A-9', 'A-10', 'A-11', 'A-12', 'A-13', 'A-14',
            'B-7', 'B-8', 'B-9', 'B-10', 'B-11', 'B-12', 'B-13', 'B-14'
            'C-7', 'C-8', 'C-9', 'C-10', 'C-11', 'C-12', 'C-13', 'C-14'
            'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12', 'D-13', 'D-14'
            'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'E-14'
            'F-7', 'F-8', 'F-9', 'F-10', 'F-11', 'F-12', 'F-13', 'F-14']
for i in testList:
    if not data[data.name.str.contains(i)].DiceBackground.values == 0 and not data[data.name.str.contains(i)].DiceBackground.values.size == 0:
        testSetBackground.append( data[data.name.str.contains(i)].DiceBackground.values )
    if not data[data.name.str.contains(i)].DiceTumor.values == 0 and not data[data.name.str.contains(i)].DiceTumor.values.size == 0:
        testSetTumor.append( data[data.name.str.contains(i)].DiceTumor.values )
    if not data[data.name.str.contains(i)].DiceStroma.values == 0 and not data[data.name.str.contains(i)].DiceStroma.values.size == 0:
        testSetStroma.append( data[data.name.str.contains(i)].DiceStroma.values )
    if not data[data.name.str.contains(i)].DiceNecrosis.values == 0 and not data[data.name.str.contains(i)].DiceNecrosis.values.size == 0:
        testSetNecrosis.append( data[data.name.str.contains(i)].DiceNecrosis.values )

import matplotlib.pyplot as plt

#Background
plt.boxplot([np.array(trainingSetBackground),np.array(validationSetBackground),np.array(testSetBackground)])
plt.xticks([1, 2, 3], ['Training', 'Validation', 'Test'])
plt.ylabel('Dice')
plt.show()

import matplotlib.pyplot as plt

#Tumor
plt.boxplot([np.array(trainingSetTumor),np.array(validationSetTumor),np.array(testSetTumor)])
plt.xticks([1, 2, 3], ['Training', 'Validation', 'Test'])
plt.ylabel('Dice')
plt.show()

plt.boxplot([np.array(trainingSetNecrosis),np.array(validationSetNecrosis),np.array(testSetNecrosis)])
plt.xticks([1, 2, 3], ['Training', 'Validation', 'Test'])
plt.ylabel('Dice')
plt.show()

plt.boxplot([np.array(trainingSetStroma),np.array(validationSetStroma),np.array(testSetStroma)])
plt.xticks([1, 2, 3], ['Training', 'Validation', 'Test'])
plt.ylabel('Dice')
plt.show()


"""
VAMOS A CHECKEAR EL DESVALANCEO DE CLASES
"""
names = []
voxelsBackground = []
voxelsTumor = []
voxelsStroma = []
voxelsNecrosis = []

listaMasks = glob.glob(os.path.join(imagesDir,'Masks','Color','*.png')) 
for j in listaMasks:
    name = os.path.basename(j)
    match = re.search('_', name)
    core = name[:match.end()]
    names.append(core)
    
    mask = io.imread(j, as_gray = False)
    mask = rgb_to_singleLayer(mask)
    for tissueValue in range(4):
        totalTissue = np.sum(np.array(mask==tissueValue, dtype=np.float))
        if tissueValue == 0:
            voxelsBackground.append(totalTissue)
        elif tissueValue == 1:
            voxelsTumor.append(totalTissue)
        elif tissueValue == 2:
            voxelsStroma.append(totalTissue)
        elif tissueValue == 3:
            voxelsNecrosis.append(totalTissue)
            
dtVoxels = {'name':names, 'voxelsBackground':voxelsBackground, 'voxelsTumor':voxelsTumor, 'voxelsStroma':voxelsStroma, 'voxelsNecrosis':voxelsNecrosis}
dataVoxels = pd.DataFrame(dtVoxels)
    
trainVoxelsBackground =[]
trainVoxelsTumor =[]
trainVoxelsNecrosis =[]
trainVoxelsStroma =[]

trainingList = ['A-1_','A-2_','A-3_', 'B-1_','B-2_','B-3_', 'C-1_','C-2_','C-3_',
                'D-1_','D-2_','D-3_', 'E-1_','E-2_','E-3_', 'F-1_','F-2_','F-3_']
for i in trainingList:
    trainVoxelsBackground.append( dataVoxels[dataVoxels.name.str.contains(i)].voxelsBackground.values )
    trainVoxelsTumor.append( dataVoxels[dataVoxels.name.str.contains(i)].voxelsTumor.values )
    trainVoxelsStroma.append( dataVoxels[dataVoxels.name.str.contains(i)].voxelsStroma.values )
    trainVoxelsNecrosis.append( dataVoxels[dataVoxels.name.str.contains(i)].voxelsNecrosis.values )
   
   
Total = np.array( [np.sum(np.array(trainVoxelsBackground, dtype=np.uint32)),
           np.sum(np.array(trainVoxelsTumor, dtype=np.uint32)),
           np.sum(np.array(trainVoxelsStroma, dtype=np.uint32)),
           np.sum(np.array(trainVoxelsNecrosis, dtype=np.uint32))], dtype=np.uint32 )

plt.bar(range(4),Total, tick_label =['Background','Tumor','Stroma','Necrosis'])

Total2 = np.array( [np.sum(np.array(dataVoxels.voxelsBackground.values, dtype=np.float)),
                    np.sum(np.array(dataVoxels.voxelsTumor.values, dtype=np.float)),
                    np.sum(np.array(dataVoxels.voxelsStroma.values, dtype=np.float)),
                    np.sum(np.array(dataVoxels.voxelsNecrosis.values, dtype=np.float))], dtype = np.float)
    