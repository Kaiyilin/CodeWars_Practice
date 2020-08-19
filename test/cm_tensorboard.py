from All_functions import *
import io
import itertools
from six.moves import range

"""
make sure you have class_names
since this cannot work with tf.data,Dataset
make sure the data you'd like to plot is named as test_images and test_images
"""
#class_names = ['0','1','2','3','4','5','6','7','8','9']

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(10, 10))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)
  

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

"""
def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)
"""
# tensorboard log directiory
#logdir="/Users/kaiyi/Desktop/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the per-epoch callback.
#file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
#cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


def Grad_CAM(input_dir,background_dir,output_dir,labels,input_size,model_path,pad_size):
    """
    can I use tree?
    background_dir: The directory of your background, better using a mask
    output_dir: Where you'd like to save those Grad_CAM images
    labels: same as the labels you given in your training procedure, this mainly helps you to identify which data were wrong
    pad_size: default is None
    """
    def readfile_pad_for_overlap(dirr, pad_size):
        os.chdir(dirr)
        cwd = os.getcwd()
        for root, dirs, files in os.walk(cwd):
            for file in files:
                if file.endswith(".nii"):
                #print(os.path.join(root, file))
                    img = nib.load(os.path.join(root, file))
                    img_array = img.get_fdata()
                    img_array = tf.keras.utils.normalize(img_array)
                    img_array = padding_zeros(img_array, pad_size)
        return img_array

    _,imgs_tensor, imgs_list = myreadfile_pad(input_dir, pad_size)
    imgs_tensor = imgs_tensor[...,None]
    model=load_model(model_path)
    #model.summary()
    conv_layer_list = list([])
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:#check for convolutional layer
            continue
        #print(i, layer.name, layer.output.shape)#summarize output shape
        conv_layer_list.append(i)
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=conv_layer_list[-1]).output, model.output])
    #grad_model.summary()

    incorrect_list = list([])
    index_init = 0
    for img_tensor in imgs_tensor:
        with tf.GradientTape() as tape:
            #Compute GRADIENT
            img_tensor_2 = img_tensor[None,...]
            conv_outputs, predictions = grad_model(img_tensor_2)
            class_index=int(tf.math.argmax(predictions,axis=1))
            loss = predictions[:, class_index]
        if class_index != labels:
            incorrect_list.append(imgs_list[index_init])

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
        capi=resize(cam,(pad_size,pad_size,pad_size))
        capi = np.maximum(capi,0)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min())
        f, axarr = plt.subplots(8,8,figsize=(15,10))
        f.suptitle('Grad-CAM') 

        background = readfile_pad_for_overlap(background_dir,64) 
        os.chdir(output_dir)
        import math
        #sag
        for slice_count in range(pad_size):
            axial_img=background[slice_count,:,:]
            axial_grad_cmap_img=heatmap[slice_count,:,:]
            axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(pad_size), 0),round(math.sqrt(pad_size), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
        plt.savefig('Grad_CAM_sag_%s.png'%(imgs_list[index_init]))
        plt.close()

        #cor
        for slice_count in range(pad_size):
            axial_img=background[:,slice_count,:]
            axial_grad_cmap_img=heatmap[:,slice_count,:]
            axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(pad_size), 0),round(math.sqrt(pad_size), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
        plt.savefig('Grad_CAM_cor_%s.png'%(imgs_list[index_init]))
        plt.close()
        #axial
        for slice_count in range(pad_size):
            axial_img=background[:,:,slice_count]
            axial_grad_cmap_img=heatmap[:,:,slice_count]
            axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(pad_size), 0),round(math.sqrt(pad_size), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
        plt.savefig('Grad_CAM_axial_%s.png'%(imgs_list[index_init]))
        plt.close()

        incorrect_list_df = pd.DataFrame(incorrect_list)
        incorrect_list_df.to_csv('Incorrect_list.csv')
        index_init+=1


#def log_Grad_CAM(epoch, logs):
  

"""
Example:
#file directory
alff_dir = '/Users/MRILab/Dropbox/file_0202/alff'
reho_dirr = '/Users/MRILab/Dropbox/file_0102/reho'
model_path = '/Users/MRILab/Desktop/logs/Res50_alff/alff_aug_Res_50/Res_50_3D.h5'
mask_dir = '/Users/MRILab/Desktop/mask_folder/'
output_dir = '/Users/MRILab/Desktop/'
INPUT_PATCH_SIZE=(64,64,64,1)

Grad_CAM(alff_dir,mask_dir,output_dir,2,INPUT_PATCH_SIZE,model_path,64)

"""