from All_functions import *
from cm_tensorboard import *

def myreadfile_pad(dirr, pad_size):
    
    #This version can import 3D array regardless of the size
    
    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            #img_array = tf.keras.utils.normalize(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def importdata_resample(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):
    def myreadfile_resample_pad(dirr, pad_size):
    
        #This version can import 3D array regardless of the size
        from nilearn.datasets import load_mni152_template
        from nilearn.image import resample_to_img
        template = load_mni152_template()

        os.chdir(dirr)
        number = 0

        flag = True
        imgs_array = np.array([])
        path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
        path_list.sort()
        for file in path_list:
            if file.endswith(".nii"):
                #print(os.path.join(dirr, file))
                img = nib.load(os.path.join(dirr, file))
                img_array = resample_to_img(img, template)
                img_array = img.get_fdata()
                #img_array = tf.keras.utils.normalize(img_array)
                img_array = padding_zeros(img_array, pad_size)
                img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
                number += 1
                if flag == True:
                    imgs_array = img_array

                else:
                    imgs_array = np.concatenate((imgs_array, img_array), axis=0)

                flag = False
        return number, imgs_array, path_list
    if pad_size == None:
      a_num, first_mo, _ = myreadfile(dirr)
      b_num, second_mo, _ = myreadfile(dirr1)
      h_num, third_mo, _ = myreadfile(dirr2)
      
      a_num2, first_mo2, _ = myreadfile(dirr3)
      b_num2, second_mo2, _ = myreadfile(dirr4)
      h_num2, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      #pad_size = int(input('Which size would you like? '))
      a_num, first_mo, _ = myreadfile_resample_pad(dirr,pad_size)
      b_num, second_mo, _ = myreadfile_resample_pad(dirr1,pad_size)
      h_num, third_mo, _ = myreadfile_resample_pad(dirr2,pad_size)
      
      a_num2, first_mo2, _ = myreadfile_resample_pad(dirr3,pad_size)
      b_num2, second_mo2, _ = myreadfile_resample_pad(dirr4,pad_size)
      h_num2, third_mo2, _ = myreadfile_resample_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

class_weight = {0: 1.4, 1: 1.0, 2: 1.5}

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  val_pred_raw = model.predict(val_images)
  val_pred = np.argmax(val_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(val_labels, val_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

#class_names must defined for cm_tensorboard
class_names = ['HC','C-','C+']

BA_alff, BB_alff, HC_alff, BA_reho, BB_reho, HC_reho = importdata_resample(mul_dir['BA'],mul_dir['BB'],mul_dir['HC'],mul_dir['BA2'],mul_dir['BB2'],mul_dir['HC2'],128)
BA_iso, BB_iso, HC_iso, BA_gfa, BB_gfa, HC_gfa = importdata_resample(mul_dir['BA3'],mul_dir['BB3'],mul_dir['HC3'],mul_dir['BA5'],mul_dir['BB5'],mul_dir['HC5'],pad_size=128)
BA_nqa, BB_nqa, HC_nqa, _, _, _ = importdata_resample(mul_dir['BA4'],mul_dir['BB4'],mul_dir['HC4'],mul_dir['BA5'],mul_dir['BB5'],mul_dir['HC5'],pad_size=128)

BA_alff = tf.keras.utils.normalize(BA_alff)
BB_alff = tf.keras.utils.normalize(BB_alff)
HC_alff = tf.keras.utils.normalize(HC_alff)

"""
BA_gfa = tf.keras.utils.normalize(BA_gfa)
BB_gfa = tf.keras.utils.normalize(BB_gfa)
HC_gfa = tf.keras.utils.normalize(HC_gfa)
"""

BA_alff_tr, BA_alff_val = split(5,BA_alff)
BB_alff_tr, BB_alff_val = split(5,BB_alff)
HC_alff_tr, HC_alff_val = split(5,HC_alff)
del BA_alff, BB_alff, HC_alff

BA_reho_tr, BA_reho_val = split(5,BA_reho)
BB_reho_tr, BB_reho_val = split(5,BB_reho)
HC_reho_tr, HC_reho_val = split(5,HC_reho)
del BA_reho, BB_reho, HC_reho

BA_gfa_tr, BA_gfa_val = split(5,BA_gfa)
BB_gfa_tr, BB_gfa_val = split(5,BB_iso)
HC_gfa_tr, HC_gfa_val = split(5,HC_gfa)
del BA_gfa, BB_gfa, HC_gfa

BA_iso_tr, BA_iso_val = split(5,BA_iso)
BB_iso_tr, BB_iso_val = split(5,BB_iso)
HC_iso_tr, HC_iso_val = split(5,HC_iso)
del BA_iso, BB_iso, HC_iso

BA_nqa_tr, BA_nqa_val = split(5,BA_nqa)
BB_nqa_tr, BB_nqa_val = split(5,BB_nqa)
HC_nqa_tr, HC_nqa_val = split(5,HC_nqa)
del BA_nqa, BB_nqa, HC_nqa

BA_alff_tr, BA_alff_val = BA_alff_tr[...,None], BA_alff_val[...,None]
BB_alff_tr, BB_alff_val = BB_alff_tr[...,None], BB_alff_val[...,None]
HC_alff_tr, HC_alff_val = HC_alff_tr[...,None], HC_alff_val[...,None]

BA_reho_tr, BA_reho_val = BA_reho_tr[...,None], BA_reho_val[...,None]
BB_reho_tr, BB_reho_val = BB_reho_tr[...,None], BB_reho_val[...,None]
HC_reho_tr, HC_reho_val = HC_reho_tr[...,None], HC_reho_val[...,None]

BA_gfa_tr, BA_gfa_val = BA_gfa_tr[...,None], BA_gfa_val[...,None] 
BB_gfa_tr, BB_gfa_val = BB_gfa_tr[...,None], BB_gfa_val[...,None]
HC_gfa_tr, HC_gfa_val = HC_gfa_tr[...,None], HC_gfa_val[...,None]

BA_iso_tr, BA_iso_val = BA_iso_tr[...,None], BA_iso_val[...,None]
BB_iso_tr, BB_iso_val = BB_iso_tr[...,None], BB_iso_val[...,None]
HC_iso_tr, HC_iso_val = HC_iso_tr[...,None], HC_iso_val[...,None]

BA_nqa_tr, BA_nqa_val = BA_nqa_tr[...,None], BA_nqa_val[...,None]
BB_nqa_tr, BB_nqa_val = BB_nqa_tr[...,None], BB_nqa_val[...,None]
HC_nqa_tr, HC_nqa_val = HC_nqa_tr[...,None], HC_nqa_val[...,None]

BA_labels_tr = np.ones(BA_alff_tr.shape[0])+1
BA_labels_val = np.ones(BA_alff_val.shape[0])+1

BB_labels_tr = np.ones(BB_alff_tr.shape[0])
BB_labels_val = np.ones(BB_alff_val.shape[0])

HC_labels_tr = np.zeros(HC_alff_tr.shape[0])
HC_labels_val = np.zeros(HC_alff_val.shape[0])

# image channel concatenate
All_alff_tr = np.concatenate([BA_alff_tr,BB_alff_tr,HC_alff_tr],axis=0)
All_alff_val = np.concatenate([BA_alff_val,BB_alff_val,HC_alff_val],axis=0)

All_reho_tr = np.concatenate([BA_reho_tr,BB_reho_tr,HC_reho_tr],axis=0)
All_reho_val = np.concatenate([BA_reho_val,BB_reho_val,HC_reho_val],axis=0)

All_gfa_tr = np.concatenate([BA_gfa_tr,BB_gfa_tr,HC_gfa_tr],axis=0)
All_gfa_val = np.concatenate([BA_gfa_val,BB_gfa_val,HC_gfa_val],axis=0)

All_iso_tr = np.concatenate([BA_iso_tr,BB_iso_tr,HC_iso_tr],axis=0)
All_iso_val = np.concatenate([BA_iso_val,BB_iso_val,HC_iso_val],axis=0)

All_nqa_tr = np.concatenate([BA_nqa_tr,BB_nqa_tr,HC_nqa_tr],axis=0)
All_nqa_val = np.concatenate([BA_nqa_val,BB_nqa_val,HC_nqa_val],axis=0)

tr_images = np.concatenate([All_alff_tr, All_reho_tr, All_gfa_tr, All_iso_tr, All_nqa_tr],axis=-1)
val_images = np.concatenate([All_alff_val, All_reho_val, All_gfa_val, All_iso_val, All_nqa_val],axis=-1)

tr_labels = np.concatenate([BA_labels_tr,BB_labels_tr,HC_labels_tr],axis=0)
val_labels = np.concatenate([BA_labels_val,BB_labels_val,HC_labels_val],axis=0)

if tr_labels.shape[0] == tr_images.shape[0]:
    print("\nsample size of tr_data is equivalent.")
else:
    sys.exit('\ncheck the size of your tr data')

if val_labels.shape[0] == val_images.shape[0]:
    print("\nsample size of val_data is equivalent.")
else:
    sys.exit('\ncheck the size of your val data')

del BA_alff_tr, BA_alff_val, BA_gfa_tr, BA_gfa_val, BA_iso_tr, BA_iso_val, BA_nqa_tr, BA_nqa_val, BA_reho_tr,BA_reho_val
del BB_alff_tr, BB_alff_val, BB_gfa_tr, BB_gfa_val, BB_iso_tr, BB_iso_val, BB_nqa_tr, BB_nqa_val, BB_reho_tr,BB_reho_val
del HC_alff_tr, HC_alff_val, HC_gfa_tr, HC_gfa_val, HC_iso_tr, HC_iso_val, HC_nqa_tr, HC_nqa_val, HC_reho_tr,HC_reho_val
del All_alff_tr, All_alff_val, All_gfa_tr, All_gfa_val, All_iso_tr, All_iso_val, All_nqa_tr, All_nqa_val, All_reho_tr,All_reho_val

def tf_random_rotate_image_xyz(image, label):
    # 3 axes random rotation
    def rotateit_y(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,2), reshape=False)
        return image

    def rotateit_x(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(1,2), reshape=False)       
        return image

    def rotateit_z(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,1), reshape=False)
        return image

    im_shape = image.shape
    [image,] = tf.py_function(rotateit_x, [image], [tf.float64])
    [image,] = tf.py_function(rotateit_y, [image], [tf.float64])
    [image,] = tf.py_function(rotateit_z, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label

# one axes random rotation
def random_rotate_image(image):
  image = scipy.ndimage.rotate(image, np.random.uniform(-90, 90), reshape=False)
  return image

def tf_random_rotate_image(image, label):
  im_shape = image.shape
  [image,] = tf.py_function(random_rotate_image, [image], [tf.float64])
  image.set_shape(im_shape)
  return image, label

tr_images, tr_labels = shuffle(tr_images, tr_labels)
tr_images, tr_labels = shuffle(tr_images, tr_labels)
val_images, val_labels = shuffle(val_images, val_labels)

print(tr_images.shape, val_images.shape)
print(tr_labels,val_labels)

_buffer_size = tr_labels.shape[0] 
_batch_size = 3

train_dataset = tf.data.Dataset.from_tensor_slices((tr_images,tr_labels))
#train_dataset = train_dataset.map(tf_random_rotate_image_xyz).shuffle(buffer_size=_buffer_size).batch(_batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images,val_labels))
val_dataset = val_dataset.batch(_batch_size)


from resnet3d import Resnet3DBuilder 
opt3 = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True)

project_name =  input('Naming this project: ')

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))
os.mkdir(logdir)
logdir = logdir + project_name
os.mkdir(logdir)
os.mkdir(checkpoint_dir)
# Define the per-epoch callback.
file_writer_cm = tf.summary.create_file_writer(logdir + 'cm')
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

with strategy.scope():
    os.chdir(logdir)
    model = Resnet3DBuilder.build_resnet_50((128, 128, 128, 5), 3,reg_factor=1e-4)
    print("\nBuilding a Res_Net_50 model")
    weights_path = '/home/user/venv/kaiyi_venv/training_checkpoints/20200727-090040/weights.68.hdf5'
    #model.load_weights(weights_path)
    #print("\nLoaded a Res_Net_50 weight from" + weights_path)
    model.compile(optimizer= opt3 , loss=loss, metrics=['sparse_categorical_accuracy'])


            

                
    callbacks = [ tf.keras.callbacks.TensorBoard(log_dir=logdir, 
                                                    histogram_freq=1, 
                                                    write_graph=True, 
                                                    write_images=False,
                                                    update_freq='epoch', 
                                                    profile_batch=2, 
                                                    embeddings_freq=0,
                                                    embeddings_metadata=None),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                   verbose=0,
                                                   save_weights_only=True,
                                                   save_freq='epoch'),
                cm_callback]
    
    history = model.fit(train_dataset.skip(round(0.1*len(tr_labels))).map(tf_random_rotate_image_xyz).shuffle(buffer_size=150).batch(_batch_size),
                             class_weight = class_weight,
                             epochs=300,
                             verbose=1,
                             callbacks=callbacks,
                             validation_split=None,
                             validation_data=train_dataset.take(round(0.1*len(tr_labels))).batch(_batch_size))

"""
    predict2 = model.predict(val_images)
    cm2 = confusion_matrix(val_labels, np.argmax(predict2,axis=1))
    fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names=None)
    fig.savefig('confusion_matrix_test.png')
    plt.close()


    predict = model.predict(tr_images)
    cm = confusion_matrix(tr_labels, np.argmax(predict,axis=1))
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names=None)
    fig.savefig('confusion_matrix_train.png')
    plt.close()

"""


