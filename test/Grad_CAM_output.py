from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import datetime
import numpy as np
#from deploy_config import*  #Import Paths and Model Config file
#from loss_funnction_And_matrics import*  #Import Loss functions
#from Resnet_3D import Resnet3D #import the model
import pandas as pd
from tensorflow.keras.optimizers import Adam
import cv2
from All_functions import *

#test file directory
alff_dir='/Users/MRILab/Dropbox/file_0102/alff'
reho_dirr = '/Users/MRILab/Dropbox/file_0102/reho'

#training file directory
alff_dir_2='/Users/MRILab/Desktop/uu'

#Load The model
INPUT_PATCH_SIZE=(64,64,64,1)
Model_3D=load_model('/Users/MRILab/Desktop/logs/Res50_alff/alff_aug_Res_50/Res_50_3D.h5')
Model_3D.summary()

#lAYER-Name--to-visualize
LAYER_NAME='conv3d_52'

# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([Model_3D.inputs], [Model_3D.get_layer(LAYER_NAME).output, Model_3D.output])
grad_model.summary()


# # Loading volume and compute the gradients 
simple_alff = myreadfile_pad(alff_dir_2,64)[1]
simple_alff = simple_alff[...,None]
simple_reho = myreadfile_pad(reho_dirr,64)[1]
simple_reho = simple_reho[...,None]
func_trial = np.concatenate([simple_alff,simple_reho],axis=-1)

#index of the class
CLASS_INDEX=2

#Compute GRADIENT
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(simple_alff)
    loss = predictions[:, CLASS_INDEX]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]


# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0,1,2))
# Build a ponderated map of filters according to gradients importance
cam = np.zeros(output.shape[0:3], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, :, index]


from skimage.transform import resize
from matplotlib import pyplot as plt
capi=resize(cam,(64,64,64))
#print(capi.shape)
capi = np.maximum(capi,0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min())
f, axarr = plt.subplots(8,8,figsize=(15,10))
f.suptitle('Grad-CAM')

def readfile_pad_for_overlap(dirr, pad_size):
    
    #This version can import 3D array regardless of the size
    
    os.chdir(dirr)
    cwd = os.getcwd()

    flag = True
    imgs_array = np.array([])
    for root, dirs, files in os.walk(cwd):
      for file in files:
          if file.endswith(".nii"):
            #print(os.path.join(root, file))
            img = nib.load(os.path.join(root, file))
            img_array = img.get_fdata()
            img_array = tf.keras.utils.normalize(img_array)
            img_array = padding_zeros(img_array, pad_size)
    return img_array

mask_dir = '/Users/MRILab/Desktop/mask_folder/'

simple_alff_2 = readfile_pad_for_overlap(mask_dir,64)

#sag
for slice_count in range(64):
    axial_img=simple_alff_2[slice_count,:,:]
    axial_grad_cmap_img=heatmap[slice_count,:,:]
    axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
    plt.subplot(8,8,slice_count+1)
    plt.imshow(axial_overlay,cmap='jet')
plt.savefig('Grad_CAM_sag.png')

#cor
for slice_count in range(64):
    axial_img=simple_alff_2[:,slice_count,:]
    axial_grad_cmap_img=heatmap[:,slice_count,:]
    axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
    plt.subplot(8,8,slice_count+1)
    plt.imshow(axial_overlay,cmap='jet')
plt.savefig('Grad_CAM_cor.png')
#axial
for slice_count in range(64):
    axial_img=simple_alff_2[:,:,slice_count]
    axial_grad_cmap_img=heatmap[:,:,slice_count]
    axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
    plt.subplot(8,8,slice_count+1)
    plt.imshow(axial_overlay,cmap='jet')
plt.savefig('Grad_CAM_axial.png')