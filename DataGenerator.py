# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:27:17 2020

@author: egarciamarcos
"""

import cv2
import numpy as np

import keras
from keras.applications.nasnet import NASNetMobile, preprocess_input

import imgaug as ia
from imgaug import augmenters as iaa

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, images_paths, labels, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):
		self.labels       = labels              # array of labels
		self.images_paths = images_paths        # array of image paths
		self.dim          = image_dimensions    # image dimensions
		self.batch_size   = batch_size          # batch size
		self.shuffle      = shuffle             # shuffle bool
		self.augment      = augment             # augment data bool
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.images_paths) / self.batch_size))

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		'Generate one batch of data'
		# selects indices of data for next batch
		indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

		# select data and load images
		labels = np.array([self.labels[k] for k in indexes])
		images = [cv2.imread(self.images_paths[k]) for k in indexes]
        
		# preprocess and augment data
		if self.augment == True:
		    images = self.augmentor(images)
		
		images = np.array([preprocess_input(img) for img in images])
		return images, labels
	
	def augmentor(self, images):
		'Apply data augmentation'
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		seq = iaa.Sequential(
				[
				# apply the following augmenters to most images
				iaa.Fliplr(0.5),  # horizontally flip 50% of all images
				iaa.Flipud(0.2),  # vertically flip 20% of all images
				sometimes(iaa.Affine(
					scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
					# scale images to 80-120% of their size, individually per axis
					translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
					# translate by -20 to +20 percent (per axis)
					rotate=(-10, 10),  # rotate by -45 to +45 degrees
					shear=(-5, 5),  # shear by -16 to +16 degrees
					order=[0, 1],
					# use nearest neighbour or bilinear interpolation (fast)
					cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
					mode=ia.ALL
					# use any of scikit-image's warping modes (see 2nd image from the top for examples)
				)),
				# execute 0 to 5 of the following (less important) augmenters per image
				# don't execute all of them, as that would often be way too strong
				iaa.SomeOf((0, 5),
				           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
						                                     n_segments=(20, 200))),
					           # convert images into their superpixel representation
					           iaa.OneOf([
							           iaa.GaussianBlur((0, 1.0)),
							           # blur images with a sigma between 0 and 3.0
							           iaa.AverageBlur(k=(3, 5)),
							           # blur image using local means with kernel sizes between 2 and 7
							           iaa.MedianBlur(k=(3, 5)),
							           # blur image using local medians with kernel sizes between 2 and 7
					           ]),
					           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
					           # sharpen images
					           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
					           # emboss images
					           # search either for all edges or for directed edges,
					           # blend the result with the original image using a blobby mask
					           iaa.SimplexNoiseAlpha(iaa.OneOf([
							           iaa.EdgeDetect(alpha=(0.5, 1.0)),
							           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
							                                  direction=(0.0, 1.0)),
					           ])),
					           iaa.AdditiveGaussianNoise(loc=0,
					                                     scale=(0.0, 0.01 * 255),
					                                     per_channel=0.5),
					           # add gaussian noise to images
					           iaa.OneOf([
							           iaa.Dropout((0.01, 0.05), per_channel=0.5),
							           # randomly remove up to 10% of the pixels
							           iaa.CoarseDropout((0.01, 0.03),
							                             size_percent=(0.01, 0.02),
							                             per_channel=0.2),
					           ]),
					           iaa.Invert(0.01, per_channel=True),
					           # invert color channels
					           iaa.Add((-2, 2), per_channel=0.5),
					           # change brightness of images (by -10 to 10 of original value)
					           iaa.AddToHueAndSaturation((-1, 1)),
					           # change hue and saturation
					           # either change the brightness of the whole image (sometimes
					           # per channel) or change the brightness of subareas
					           iaa.OneOf([
							           iaa.Multiply((0.9, 1.1), per_channel=0.5),
							           iaa.FrequencyNoiseAlpha(
									           exponent=(-1, 0),
									           first=iaa.Multiply((0.9, 1.1),
									                              per_channel=True),
									           second=iaa.ContrastNormalization(
											           (0.9, 1.1))
							           )
					           ]),
					           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
					                                               sigma=0.25)),
					           # move pixels locally around (with random strengths)
					           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
					           # sometimes move parts of the image around
					           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
				           ],
				           random_order=True
				           )
				],
				random_order=True
		)
		return seq.augment_images(images)