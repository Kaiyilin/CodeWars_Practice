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
#from All_functions import *

#file directory
alff_dir='/Users/MRILab/Dropbox/file_0202/alff'
reho_dirr = '/Users/MRILab/Dropbox/file_0202/reho'


#Load The model
INPUT_PATCH_SIZE=(64,64,64,2)
model=load_model('/Users/MRILab/Dropbox/Res_34_3D.h5')
model.summary()

#lAYER-Name--to-visualize
LAYER_NAME='conv3d_359'

# Create a graph that outputs target convolution and output
conv_layer_list = list([])
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:#check for convolutional layer
        continue
    #print(i, layer.name, layer.output.shape)#summarize output shape
    conv_layer_list.append(i)
    
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=conv_layer_list[-1]).output, model.output])
grad_model.summary()


# # Loading volume and compute the gradients 
simple_alff = myreadfile_pad(alff_dir,64)[1]
simple_alff = simple_alff[...,None]
simple_reho = myreadfile_pad(reho_dirr,64)[1]
simple_reho = simple_reho[...,None]
func_trial = np.concatenate([simple_alff,simple_reho],axis=-1)

#index of the class
CLASS_INDEX=1

#Compute GRADIENT
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(func_trial)
    loss = predictions[:, CLASS_INDEX]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]


# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0, 1,2))
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
f, axarr = plt.subplots(3,3,figsize=(15,10))
f.suptitle('Grad-CAM')
slice_count=50
slice_count2=55


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

simple_alff_2 = readfile_pad_for_overlap(alff_dir,64)

axial_ct_img=np.squeeze(simple_alff_2[:, :, slice_count])
axial_grad_cmap_img=np.squeeze(heatmap[:,:, slice_count])

coronal_ct_img=np.squeeze(simple_alff_2[:,slice_count2,:])
coronal_grad_cmap_img=np.squeeze(heatmap[:,slice_count2,:])

sag_img = np.squeeze(simple_alff_2[slice_count,:,:])
sag_grad_cmap_img=np.squeeze(heatmap[slice_count,:,:])

img_plot = axarr[0,0].imshow(axial_ct_img, cmap='gray');
axarr[0,0].axis('off')
axarr[0,0].set_title('Region')
    
img_plot = axarr[0,1].imshow(axial_grad_cmap_img, cmap='jet');
axarr[0,1].axis('off')
axarr[0,1].set_title('Grad-CAM')
    
axial_overlay=cv2.addWeighted(axial_ct_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
    
img_plot = axarr[0,2].imshow(axial_overlay,cmap='jet')
axarr[0,2].axis('off')
axarr[0,2].set_title('Overlay')


img_plot = axarr[1,0].imshow(coronal_ct_img, cmap='gray')
axarr[1,0].axis('off')
axarr[1,0].set_title('Region2')
    
img_plot = axarr[1,1].imshow(coronal_grad_cmap_img, cmap='jet')
axarr[1,1].axis('off')
axarr[1,1].set_title('Grad-CAM')
    
Coronal_overlay=cv2.addWeighted(coronal_ct_img,0.3,coronal_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
    
img_plot = axarr[1,2].imshow(Coronal_overlay,cmap='jet')
axarr[1,2].axis('off')
axarr[1,2].set_title('Overlay')

img_plot = axarr[2,0].imshow(sag_img, cmap='gray')
axarr[1,0].axis('off')
axarr[1,0].set_title('Region3')
    
img_plot = axarr[2,1].imshow(sag_grad_cmap_img, cmap='jet')
axarr[1,1].axis('off')
axarr[1,1].set_title('Grad-CAM')
    
sag_overlay=cv2.addWeighted(sag_img,0.3,sag_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
    
img_plot = axarr[2,2].imshow(sag_overlay,cmap='jet')
axarr[1,2].axis('off')
axarr[1,2].set_title('Overlay')