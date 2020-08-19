# For continuous training

from All_functions import *
from Data_aug_alff_only import *

model_dir = '/home/user/venv/kaiyi_venv/model_preserved/Res_50_3D_0610_fixedorder.h5'
#model = load_model(model_dir)

opt3 = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,mult_model.optimizer.lr.numpy()))

def decay(epoch):
    if epoch <= 50:
        return 1e-4
    elif epoch >50 and epoch <= 100:
        return 1e-5
    else:
        return 1e-6

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))


_buffer_size = All_labels_tr.shape[0] 
_batch_size = 5

train_dataset = tf.data.Dataset.from_tensor_slices((All_alff_tr,All_labels_tr))
train_dataset = train_dataset.shuffle(buffer_size=_buffer_size).batch(_batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((All_alff_val,All_labels_val))
val_dataset = val_dataset.batch(_batch_size)

#kf = StratifiedKFold(10, shuffle=True, random_state=None) # Use for StratifiedKFold classification

#fold = 0
os.mkdir(logdir)

#batch_list = [3, 5, 7,15]

with strategy.scope():
    mult_model = load_model(model_dir)
    print("Load a previously trained model")
    #opt = tf.keras.optimizers.Adam(lr=1e-2)
    mult_model.compile(optimizer= opt3 , loss=loss, metrics=['sparse_categorical_accuracy'])


            

                
    
    callbacks = [ tf.keras.callbacks.TensorBoard(log_dir=logdir, 
                                                    histogram_freq=1, 
                                                    write_graph=True, 
                                                    write_images=False,
                                                    update_freq='epoch', 
                                                    profile_batch=2, 
                                                    embeddings_freq=0,
                                                    embeddings_metadata=None),
                tf.keras.callbacks.LearningRateScheduler(decay),
                PrintLR()]

    history = mult_model.fit(train_dataset,
                            epochs=250,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=val_dataset)

    predict2 = mult_model.predict(All_alff_val)

    cm2 = confusion_matrix(All_labels_val, np.argmax(predict2,axis=1))
    
    os.chdir(logdir)

    fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names=None)
    fig.savefig('confusion_matrix_test.png')
    
    plt.close()

    mult_model.save('Res_50_3D_continue.h5')