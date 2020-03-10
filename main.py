from model import *
from data import *
import os
import numpy as np

import matplotlib.pyplot as plt

from keras.callbacks import *
import skimage.io as io
import skimage.transform as trans

from FCN_Vgg16 import *

from random import randint

"""
"""
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores/training3'
validationDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores/validation3'
testDir = 'C:/Users/egarciamarcos/Desktop/Run3/HE-CK-645/Pruebas/Cores/test3'

imageSize = 256
numberOfClasses = 3

seed = randint(0,1e5)

print('Random seed = ' + str(seed))

####

import imgaug as ia
import imgaug.augmenters as iaa

ia.seed(seed)

seq = iaa.Sequential([
    # iaa.AddToHueAndSaturation((-1, 1),per_channel=True)        
    # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
    iaa.MultiplySaturation((0.5, 1.5)),
    
    # iaa.Crop(px=(0, 16)), 
    # crop images from each side by 0 to 16px (randomly chosen)

    # iaa.Fliplr(0.5), 
    # horizontally flip 50% of the images

    iaa.GaussianBlur(sigma=(0, 3.0)), 
    # blur images with a sigma of 0 to 3.0
    
    iaa.LinearContrast((0.75, 1.5)),
    # Strengthen or weaken the contrast in each image.
    
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Add gaussian noise.
    
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    ], random_order=True)

def augment(img):
    img=np.array(img,dtype=np.uint8)
    seq_det = seq.to_deterministic()
    aug_image = seq_det.augment_image(img)

    # return applications.inception_resnet_v2.preprocess_input(aug_image)
    return aug_image
    #return seq(img)
###    

data_gen_args = dict(rotation_range=90,
                    #width_shift_range=0.25,
                    #height_shift_range=0.25,
                    shear_range=5,
                    zoom_range=[0.5,1.5],
                    #channel_shift_range=100,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='reflect',
                    brightness_range = [0.8,1.2],
                    preprocessing_function = augment,
                    featurewise_std_normalization = False)

myGene = trainGenerator(1, trainDir, 'Images','masks', data_gen_args, image_color_mode = 'rgb', seed = seed,
                        mask_color_mode = 'rgb', save_to_dir = None, flag_multi_class=True, num_class = numberOfClasses, 
                        target_size = (imageSize,imageSize))

valGen = validationGenerator(1, validationDir, 'Images','masks', data_gen_args, image_color_mode = 'rgb', seed = seed,
                        mask_color_mode = 'rgb', save_to_dir = None, flag_multi_class=True, num_class = numberOfClasses, 
                        target_size = (imageSize,imageSize))


# model = unet(pretrained_weights = 'model-MultiClass-Drop01-2560.hdf5',  input_size = (imageSize,imageSize,3), numClasses=numberOfClasses)
model = unet(pretrained_weights = None,  input_size = (imageSize,imageSize,3), numClasses=numberOfClasses)

# model = FCN_Vgg16(pretrained_weights = None,  input_size = (imageSize,imageSize,3), numClasses=4)

#  model = load_model('unet_Cores_4x512.hdf5')
#  model_checkpoint = ModelCheckpoint('unet_Cores_4x512-Stroma.hdf5', monitor='loss',verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50) 

history = History()
model.fit_generator(myGene,steps_per_epoch=954,epochs=200,
                    validation_data = valGen, validation_steps = 798,
                    callbacks=[history, reduce_lr, es])

# Fit the model
# History = model.fit(myGene, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['lr'])
# plt.plot(history.history['val_loss'])
plt.title('Reduced learning rate')
plt.ylabel('learning rate')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()


""" 
Save model to json and reload model
"""
"""
# serialize model to JSON
model_json = model.to_json()
with open('model-MultiClass-Drp01-'+ str(imageSize) +'.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('model-MultiClass-Drop01-'+ str(imageSize) +'.hdf5')
print("Saved model to disk")
"""

""" 
# later...
# load json and create model
json_file = open('model-MultiClass-2560.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model-MultiClass-2560.hdf5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
"""

 
"""
# testGene = testGenerator("data/membrane/test")
testGene  = testGenerator(os.path.join(testDir,'Images'),
                          num_image = 81, target_size=(imageSize,imageSize), as_gray=False)

score = model.evaluate_generator(valGen,steps=81, max_queue_size=1, workers=1, verbose=1,pickle_safe=False)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


results = model.predict_generator(testGene,81,verbose=1)
# saveResult("data/membrane/test",results)
saveResult(mainDir,results, num_class = 1)
"""

"""
# testpath = 'C:\\Users\\egarciamarcos\\Desktop\\Imagenes\\Project3\\masks\\Tumor'
lista = os.listdir(os.path.join(testDir,'Images'))

from skimage.color import rgb2hed

for i in lista:
    img = io.imread(os.path.join(testDir,'Images',i),as_gray = False)
    img = img.astype('float32')
    # img = (img - np.min(img))/(np.max(img)-np.min(img))
    img = rgb2hed(img)
    # img = (img - np.min(img))/(np.max(img)-np.min(img))
    # M = np.max(img)
    # img /= M
    # plt.imshow(img)
    # plt.show()
    img = np.reshape(img,(1,)+img.shape)
    res = model.predict(img,verbose=1)
    # plt.imshow(res[0,:,:,0],  cmap='gray')
    # plt.show()
    io.imsave(os.path.join(testDir,i),res[0,:,:,0])

"""