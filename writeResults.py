# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:18:50 2020

@author: egarciamarcos
"""
import os
import glob
from skimage import io
from skimage.color import rgb2hed
import numpy as np

import pandas as pd

names = []
values = []

imagesDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores'

lista = glob.glob(os.path.join(imagesDir,'*.png'))

# file = open(os.path.join(imagesDir,'Results.txt'),'w') 
# file.write('Core \t Dice\n') 

for i in lista:
    name = os.path.basename(i)
    
    old_mask = io.imread(os.path.join(imagesDir,'Tumor',name[:-4]+'-Tumor.png' ))
    old_mask = old_mask/ np.max(old_mask)
    
    new_mask = io.imread(os.path.join(imagesDir,name),as_gray = False)
    # new_mask = np.array((new_mask>127), dtype=np.uint8)
    
    Dice = 2*np.sum(old_mask[new_mask>127] / (np.sum(old_mask) + np.sum(np.array(new_mask>127, dtype=np.float))))
    if not Dice==0 and not np.isnan(Dice):
        names.append(name)
        values.append(Dice)
    #file.write((name+'\t'+str(Dice)+'\n')) 
    print(Dice)
    

dt = {'name':names, 'Dice':values}
data = pd.DataFrame(dt)

data.boxplot()
data.plot.bar()

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

dt[dt.name.str.contains('A-1_')]
# file.close()

