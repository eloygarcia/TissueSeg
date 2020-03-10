from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
# import CustomImageDataGenerator
# from CustomImageDataGenerator import ImageDataGenerator

import numpy as np 
from keras.utils import to_categorical

import os
import glob
import skimage.io as io
import skimage.transform as trans

# import matplotlib.pyplot as plt


from skimage.color import rgb2hed
"""
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
"""

id2code = {0: (0, 0, 0),
           1: (255, 0, 0),
           2: (0, 255, 0)}
           #3: (0, 0, 255)}

def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:-1]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:-1])
    return encoded_image

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

"""
"""

from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
import tensorflow as tf

from PIL import Image, ImageEnhance

def random_contrast(im, sd=0.5, min=0, max=10):
    """Creates a new image which randomly adjusts the contrast of `im` by
       randomly sampling a contrast value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling contrast value.
        min:  (int or float) Clip contrast value to be no lower than this.
        max:  (int or float) Clip contrast value to be no higher than this.

    Returns:
        PIL image with contrast randomly adjusted.
    """
    contrast_factor = np.clip(np.random.normal(loc=1, scale=sd), min, max)
    enhancer = ImageEnhance.Contrast(im)
    return enhancer.enhance(contrast_factor)

def random_saturation(img, sd=0.5, min=0, max=10):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    saturation_factor = np.clip(np.random.normal(loc=1, scale=sd), min, max)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(saturation_factor)
    
from keras import backend as K

def color(x: K.tf.Tensor) -> K.tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    K.tf.enable_eager_execution()
    
    x = K.tf.image.random_hue(x, 0.08)
    x = K.tf.image.random_saturation(x, 0.6, 1.6)
    x = K.tf.image.random_brightness(x, 0.05)
    x = K.tf.image.random_contrast(x, 0.7, 1.3)
   # x = Kimage.convert_image_dtype(x,dtype=tf.float32)
    
    x = K.tf.minimum(x, 1.0)
    x = K.tf.maximum(x, 0.0)
    return x

import imgaug as ia
from imgaug import augmenters as iaa

def adjustData(img,mask,flag_multi_class,num_class):
    img = img.astype('float')
    
    #img = np.array(img[0,:,:,:], dtype=np.uint8)
    # print(img.shape)
    
    #img = (img - np.min(img))/(np.max(img)-np.min(img))
    img = img/255
    
    # img = random_noise(img, mode='gaussian')    
    # img = img + np.random.random(img.shape).astype(np.float32)*128
    """
    tf.enable_eager_execution()
    with tf.Session() as sess: # ??
        # with tf.device('/device:CPU:0'):
        img = tf.convert_to_tensor(img)
        # img = tf.image.random_brightness(img,max_delta=0.5) # Brightness esta en el generador    
        img = tf.image.random_hue(img,max_delta=0.05)   # Incluído a ultima hora, no entrenado con esto!!!
        img = tf.image.random_saturation(img,lower=0.8,upper=1.2)
        Img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    # img = tf.clip_by_value(img, 0, 255)
    
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)
        
    img = tf.image.convert_image_dtype(img,dtype=tf.float32)
    """
    # img = color(img)
    
    """
    # Gaussian noise
    mean = 0
    var = 2.5 # 0.5
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,img.shape)
    gauss = gauss.reshape(img.shape)
    img = img + gauss
    """
    # img = img.astype('float32')
    img = rgb2hed( img )
    # print(img.shape)
    
    # Gaussian Filter
    # img = gaussian_filter(img, sigma=0.75)
       
    # Image normalization
    # img = (img - np.min(img))/(np.max(img)-np.min(img)) 
    
    if(flag_multi_class):
        mask = mask.astype('float32')
 
        mask[mask > 128] = 255
        mask[mask <= 128] = 0
        
        mask = rgb_to_onehot(mask, id2code)
        # mask = (mask - np.min(mask))/(np.max(mask)-np.min(mask))
        # mask = to_categorical(mask, num_class)
        
        #M = np.max(img)
        #img /= M
        #m = np.max(mask)
        #mask /= m
        """
        # mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            print(np.unique(mask))
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            new_mask[index] = 1
            # new_mask[mask == i,i] = 1
        # new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
        """
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
        # print(mask)
        
    elif(np.max(mask) > 1):
        mask = mask.astype('float32')
        mask = (mask - np.min(mask))/(np.max(mask)-np.min(mask))
        # M = np.max(img)
        # img /= M
        # m = np.max(mask)
        # mask /= m
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
      
    return (img,mask)

def randomNoise(x, mean=0, var=2.5):
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,x.shape)
    gauss = gauss.reshape(x.shape)
    x = x + gauss
    yield x
    


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        interpolation ='linear',
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        interpolation='nerarest',
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)





def validationGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    validation_generator = zip(image_generator, mask_generator)
    for (img,mask) in validation_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    lista = os.listdir(test_path)
    for i in lista:
        img = io.imread(os.path.join(test_path,i),as_gray = as_gray)
        img = img.astype('float32')
        #img = (img - np.min(img))/(np.max(img)-np.min(img))
        img = rgb2hed( img )
        # for i in range(3):
        #    img[:,:,i] = (img[:,:,i]-np.min(img[:,:,i]))/(np.max(img[:,:,i])-np.min(img[:,:,i]))
        # M = np.max(img)
        # img /= M
        img = trans.resize(img,target_size)
        if as_gray:
            img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr



def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return (img - np.min(img))/(np.max(img)-np.min(img))
        



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        # img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        # item = item * 255
        item = item.astype('float32')
        item = np.round( (item - np.min(item))/(np.max(item)-np.min(item)) * 255)
        item = item.astype('uint8')
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),item)