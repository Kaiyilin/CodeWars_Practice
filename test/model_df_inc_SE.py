
from All_functions import * 
#from Data_concatenate_mult import *
from cm_tensorboard import *

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
                img_array = data_preprocessing(img_array)
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
      _, first_mo,  = myreadfile(dirr)
      _, second_mo, _ = myreadfile(dirr1)
      _, third_mo, _ = myreadfile(dirr2)
      
      _, first_mo2, _ = myreadfile(dirr3)
      _, second_mo2, _ = myreadfile(dirr4)
      _, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      _, first_mo, _ = myreadfile_resample_pad(dirr,pad_size)
      _, second_mo, _ = myreadfile_resample_pad(dirr1,pad_size)
      _, third_mo, _ = myreadfile_resample_pad(dirr2,pad_size)
      
      _, first_mo2, _ = myreadfile_resample_pad(dirr3,pad_size)
      _, second_mo2, _ = myreadfile_resample_pad(dirr4,pad_size)
      _, third_mo2, _ = myreadfile_resample_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

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

def base_model_creator(model, train_para = False):
    base_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-2).output])
    base_model.trainable = train_para
    return base_model

#class_names must defined for cm_tensorboard
class_names = ['HC','C-','C+']

BA_alff, BB_alff, HC_alff, BA_reho, BB_reho, HC_reho = importdata_resample(mul_dir['BA'],mul_dir['BB'],mul_dir['HC'],mul_dir['BA2'],mul_dir['BB2'],mul_dir['HC2'],128)
BA_iso, BB_iso, HC_iso, BA_gfa, BB_gfa, HC_gfa = importdata_resample(mul_dir['BA3'],mul_dir['BB3'],mul_dir['HC3'],mul_dir['BA5'],mul_dir['BB5'],mul_dir['HC5'],pad_size=128)
BA_nqa, BB_nqa, HC_nqa, _, _, _ = importdata_resample(mul_dir['BA4'],mul_dir['BB4'],mul_dir['HC4'],mul_dir['BA5'],mul_dir['BB5'],mul_dir['HC5'],pad_size=128)


BA_alff_tr, BA_alff_val = split(5,BA_alff)
BB_alff_tr, BB_alff_val = split(4,BB_alff)
HC_alff_tr, HC_alff_val = split(5,HC_alff)
del BA_alff, BB_alff, HC_alff

BA_reho_tr, BA_reho_val = split(5,BA_reho)
BB_reho_tr, BB_reho_val = split(4,BB_reho)
HC_reho_tr, HC_reho_val = split(5,HC_reho)
del BA_reho, BB_reho, HC_reho

BA_gfa_tr, BA_gfa_val = split(5,BA_gfa)
BB_gfa_tr, BB_gfa_val = split(4,BB_iso)
HC_gfa_tr, HC_gfa_val = split(5,HC_gfa)
del BA_gfa, BB_gfa, HC_gfa

BA_iso_tr, BA_iso_val = split(5,BA_iso)
BB_iso_tr, BB_iso_val = split(4,BB_iso)
HC_iso_tr, HC_iso_val = split(5,HC_iso)
del BA_iso, BB_iso, HC_iso

BA_nqa_tr, BA_nqa_val = split(5,BA_nqa)
BB_nqa_tr, BB_nqa_val = split(4,BB_nqa)
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

def tfdata_shuffle_and_split(images, labels, split_number):
    images, labels = shuffle(images, labels)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds_tr = ds.skip(split_number)
    ds_val = ds.take(split_number)
    return ds_tr, ds_val

# tr image channel concatenate
BA_images_tr = np.concatenate([BA_alff_tr, BA_gfa_tr, BA_reho_tr, BA_iso_tr, BA_nqa_tr],axis=-1)
BB_images_tr = np.concatenate([BB_alff_tr, BB_gfa_tr, BB_reho_tr, BB_iso_tr, BB_nqa_tr],axis=-1)
HC_images_tr = np.concatenate([HC_alff_tr, HC_gfa_tr, HC_reho_tr, HC_iso_tr, HC_nqa_tr],axis=-1)
BA_ds_tr, BA_ds_val = tfdata_shuffle_and_split(BA_images_tr, BA_labels_tr, 5)
BB_ds_tr, BB_ds_val = tfdata_shuffle_and_split(BB_images_tr, BB_labels_tr, 4)
HC_ds_tr, HC_ds_val = tfdata_shuffle_and_split(HC_images_tr, HC_labels_tr, 5)

# should I using prefetch?
ds_tr = tf.data.Dataset.concatenate(BA_ds_tr,BB_ds_tr)
ds_tr = tf.data.Dataset.concatenate(ds_tr, HC_ds_tr)
ds_val = tf.data.Dataset.concatenate(BA_ds_val,BB_ds_val)
ds_val = tf.data.Dataset.concatenate(ds_val, HC_ds_val)


# val image channel concatenate
BA_images_val = np.concatenate([BA_alff_val, BA_gfa_val, BA_reho_val, BA_iso_val, BA_nqa_val],axis=-1)
BB_images_val = np.concatenate([BB_alff_val, BB_gfa_val, BB_reho_val, BB_iso_val, BB_nqa_val],axis=-1)
HC_images_val = np.concatenate([HC_alff_val, HC_gfa_val, HC_reho_val, HC_iso_val, HC_nqa_val],axis=-1)
val_images = np.concatenate([BA_images_val, BB_images_val, HC_images_val],axis=0)
val_labels = np.concatenate([BA_labels_val,BB_labels_val,HC_labels_val],axis=0)

if len(val_labels) == len(val_images) and val_images.ndim == 5:
    print("\nsample size of val_data is equivalent.")
else:
    sys.exit('\ncheck the size of your val data')

del BA_alff_tr, BA_alff_val, BA_gfa_tr, BA_gfa_val, BA_iso_tr, BA_iso_val, BA_nqa_tr, BA_nqa_val, BA_reho_tr,BA_reho_val
del BB_alff_tr, BB_alff_val, BB_gfa_tr, BB_gfa_val, BB_iso_tr, BB_iso_val, BB_nqa_tr, BB_nqa_val, BB_reho_tr,BB_reho_val
del HC_alff_tr, HC_alff_val, HC_gfa_tr, HC_gfa_val, HC_iso_tr, HC_iso_val, HC_nqa_tr, HC_nqa_val, HC_reho_tr,HC_reho_val

def tf_random_rotate_image_xyz(image, label):
    # 3 axes random rotation
    def rotateit_y(image, Switch=False):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,2), reshape=False)
        return image

    def rotateit_x(image, Switch=False):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(1,2), reshape=False)       
        return image

    def rotateit_z(image, Switch=False):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,1), reshape=False)
        return image

    im_shape = image.shape
    [image,] = tf.py_function(rotateit_x, [image], [tf.float64])
    [image,] = tf.py_function(rotateit_y, [image], [tf.float64])
    [image,] = tf.py_function(rotateit_z, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label

# one axes random rotation
def tf_random_rotate_image(image, label):
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-90, 90), reshape=False)
        return image
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label


_buffer_size = 150
_batch_size = 1

def Conv_block_incep_SE(input_tensors,num_filters,param,ratio):
    #batch10 = BatchNormalization()(input_tensors)
    x = Conv3D(num_filters,(7,7,7),strides=(2,2,2),padding='same',activation='relu', kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input_tensors)
    for i in range(2):
        x = Conv_SE_block(x,num_filters,(3,3,3),param,ratio)
        x = Conv_SE_block(x,num_filters,(3,3,3),param,ratio)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    for i in range(4):
        x = inc_module_A(x,num_filters*2,param)
        x = se_block_3D(x, ratio)

    x = half_reduction(x,num_filters*2)

    for i in range(7):
        x = inc_module_B(x,num_filters*4,param)
        x = se_block_3D(x, ratio)

    x = half_reduction(x,num_filters*4)

    for i in range(4):
        x = inc_module_C(x,num_filters*8,param)
        x = se_block_3D(x, ratio)

    final_output = GlobalAveragePooling3D()(x)
    return final_output


def model_df_inc(num_classes,num_filters,shape,param,ratio,Dropout_rate):
    
    input_tensors = Input(shape=shape)
    final_output = Conv_block_incep_SE(input_tensors, num_filters, param, ratio)

    Dropout_regulise= Dropout(Dropout_rate)(final_output)
    output = Dense(num_classes, activation='softmax')(Dropout_regulise)
    model = Model(inputs=[input_tensors], outputs=output, name='model_df_inc')
    print(model.summary())
    return model

# Defining log and ckpt path
project_name =  input('Naming this project: ')
os.mkdir(logdir)
logdir = logdir + project_name
os.mkdir(logdir)
os.mkdir(checkpoint_dir)


# Setting Scope
strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    print("\nBuilding a DF_SE_incep model")
    ratio_list = [4,8,16]
    for ratio in ratio_list:
        log_ratio = (logdir + "/ratio_%s"%(ratio))
        os.mkdir(log_ratio)
        #os.chdir(log_ratio)
        # Define the per-epoch callback.
        file_writer_cm = tf.summary.create_file_writer(log_ratio + 'cm')
        cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
        print('log file saved to'+ log_ratio)

        model = model_df_inc(3,32,(128,128,128,5),1e-4,ratio,0.5)
        model.compile(optimizer=opt3,loss=loss,metrics=['sparse_categorical_accuracy'])

        print('\nTraining initiation, please wait...')

            
                    
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
        
        history = model.fit(ds_tr.map(tf_random_rotate_image).shuffle(buffer_size=120).batch(_batch_size),
                                epochs=100,
                                verbose=1,
                                callbacks=callbacks,
                                validation_split=None,
                                validation_data=ds_val.batch(_batch_size))

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


    
    