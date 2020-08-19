# Resnet 3D model 
from All_functions import *
from Data_conca_single import *

#os.chdir('/home/user/Desktop/mult_channel/keras-resnet3d')
from resnet3d import Resnet3DBuilder 
opt3 = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,mult_model.optimizer.lr.numpy()))

def decay(epoch):
    if epoch <= 50:
        return 1e-3
    elif epoch >50 and epoch <= 200:
        return 1e-4
    elif epoch > 200 and epoch <= 300:
        return 1e-5
    else:
        return 1e-6

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))



kf = StratifiedKFold(10, shuffle=True, random_state=None) # Use for StratifiedKFold classification

fold = 0
os.mkdir(logdir)

batch_list = [3, 5, 7,15]

with strategy.scope():

    for batch in batch_list:

        for train, test in kf.split(All_func_tr,All_labels_tr): # Must specify y StratifiedKFold for 
            #Defining a model at here
            mult_model = Resnet3DBuilder.build_resnet_34((64, 64, 64, 2), 3,reg_factor=1e-4)
            print("building a Res_Net_34 model")
            #opt = tf.keras.optimizers.Adam(lr=1e-2)
            mult_model.compile(optimizer= opt3 , loss=loss, metrics=['sparse_categorical_accuracy'])


            fold+=1
            print(f"Fold #{fold}")
                
            x_train = All_func_tr[train]
            y_train = All_labels_tr[train]
            x_test = All_func_tr[test]
            y_test = All_labels_tr[test]
            
            log_fold = (logdir + "/Res_34_fold_%s_batch_%s"%(fold,batch))
            os.mkdir(log_fold)
            os.chdir(log_fold)
            print('log file saved to'+ log_fold)
            

                
            
            callbacks = [ tf.keras.callbacks.TensorBoard(log_dir=log_fold, 
                                                            histogram_freq=1, 
                                                            write_graph=True, 
                                                            write_images=False,
                                                            update_freq='epoch', 
                                                            profile_batch=2, 
                                                            embeddings_freq=0,
                                                            embeddings_metadata=None),
                        tf.keras.callbacks.LearningRateScheduler(decay),
                        PrintLR()]

            history = mult_model.fit(x_train,
                                    y_train,
                                    batch_size=batch,
                                    epochs=500,
                                    verbose=1,
                                    callbacks=callbacks,
                                    validation_data=(x_test,y_test))
            
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv('history.csv')
            pred = mult_model.predict(x_test)
            plt.plot(history.history['sparse_categorical_accuracy'])
            plt.plot(history.history['val_sparse_categorical_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train_accu', 'val_accu'], loc='upper left')
            plt.savefig('accu.png')

            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train_loss', 'val_loss'], loc='upper left')
            plt.savefig('loss.png')
            plt.close()


            predict = mult_model.predict(x_train) 
            
            cm = confusion_matrix(y_train, np.argmax(predict,axis=1)) 
            os.chdir(log_fold) 
            fig, ax = plot_confusion_matrix(conf_mat=cm, 
                                            colorbar=True, 
                                            show_absolute=True, 
                                            class_names=None) 
            fig.savefig('confusion_matrix_tr.png') 
            plt.close()             

            # confusion matrix for val fold
            predict1 = mult_model.predict(x_test) 
            
            cm1 = confusion_matrix(y_test, np.argmax(predict1,axis=1)) 
            os.chdir(log_fold) 
            fig, ax = plot_confusion_matrix(conf_mat=cm1, 
                                            colorbar=True, 
                                            show_absolute=True, 
                                            class_names=None) 
            fig.savefig('confusion_matrix_val.png') 
            plt.close()             


            predict2 = mult_model.predict(All_func_val)

            cm2 = confusion_matrix(All_labels_val, np.argmax(predict2,axis=1))
            
            os.chdir(log_fold)

            fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                            colorbar=True,
                                            show_absolute=True,
                                            show_normed=False,
                                            class_names=None)
            fig.savefig('confusion_matrix_test.png')
            plt.close()

            #mult_model.save('Res_34_3D.h5')


