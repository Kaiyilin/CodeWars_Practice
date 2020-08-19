from All_functions import *
from Data_conca_single import *

epoch = 500
batch = 15

class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      mult_model.optimizer.lr.numpy()))

def decay(epoch):
    if epoch <= 50:
        return 1e-3
    elif epoch >50 and epoch <= 200:
        return 1e-4
    elif epoch > 200 and epoch <= 400:
        return 1e-5
    else:
        return 1e-6



def Conv_block_incep_SE(input_tensors,num_filters,param,ratio):
    #batch10 = BatchNormalization()(input_tensors)
    #conv00 = Conv3D(num_filters,(7,7,7),strides=(2,2,2),padding='same',activation='relu', kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(input_tensors)
    conv11 = Conv_SE_block(input_tensors,num_filters,(3,3,3),param,ratio)
    conv12 = Conv_SE_block(conv11,num_filters,(3,3,3),param,ratio)
    conv13 = Conv_SE_block(conv12,num_filters,(3,3,3),param,ratio)
    pool11 = MaxPooling3D(pool_size=(2, 2, 2))(conv13)
    conv14 = Conv_SE_block(pool11,num_filters,(3,3,3),param,ratio)
    conv15 = Conv_SE_block(conv14,num_filters,(3,3,3),param,ratio)
    pool12 = MaxPooling3D(pool_size=(2, 2, 2))(conv15)
    inc = inc_module_A_2(pool12,num_filters*2,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_A_2(inc,num_filters*2,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_A_2(inc,num_filters*2,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_A_2(inc,num_filters*2,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_A_2(inc,num_filters*2,param)
    inc = se_block_3D(inc, ratio)
    reduction = half_reduction(inc,num_filters*2)
    inc = inc_module_B_2(reduction,num_filters*4,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_B_2(inc,num_filters*4,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_B_2(inc,num_filters*4,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_B_2(inc,num_filters*4,param)
    inc = se_block_3D(inc, ratio)
    reduction=half_reduction(inc,num_filters*4)
    inc = inc_module_C_2(reduction,num_filters*8,param)
    inc = se_block_3D(inc, ratio)
    inc = inc_module_C_2(inc,num_filters*8,param)
    inc = se_block_3D(inc, ratio)
    final_output = GlobalAveragePooling3D()(inc)
    return final_output


def model_df_inc_SE(num_classes,num_filters,shape,param,ratio):
    
    input = Input(shape=shape)
    final_output = Conv_block_incep_SE(input,num_filters,param,ratio)

    Dropout_regulise= Dropout(0.5)(final_output)
    output = Dense(num_classes, activation='softmax')(Dropout_regulise)
    model = Model(inputs=input, outputs=output, name='model_f_inc_SE')
    print(model.summary())
    return model


#model = model_df_inc(3,32,(64,64,64,2),(128,128,128,2),0.0001)

#model_structure(model)

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

kf = StratifiedKFold(10, shuffle=True, random_state=None) 
fold = 0
os.mkdir(logdir)
ratio_list=[4,8,16,32]

with strategy.scope():
  print("\nBuilding a SE_incep model")
  for ratio in ratio_list:

    for train, test in kf.split(All_func_tr,All_labels_tr):    
        mult_model = model_df_inc_SE(3,32,(64,64,64,2),1e-4,16)
        mult_model.compile(optimizer=opt3,loss=loss,metrics=['sparse_categorical_accuracy'])
        fold+=1
        print(f"Fold #{fold}")
        print('\nTraining initiation, please wait...')

        x_train = All_func_tr[train]
        y_train = All_labels_tr[train]
        x_test = All_func_tr[test]
        y_test = All_labels_tr[test]
        
        log_fold = (logdir + "/incep_f_SE_fold_%s_ratio_%s"%(fold,ratio))
        os.mkdir(log_fold)
        os.chdir(log_fold)
        print('log file saved to'+ log_fold)

        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_fold, 
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
                                epochs=epoch,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=[x_test,y_test])
        
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

        # Measure this fold's accuracy
        #y_compare = np.argmax(y_test,axis=1) # For accuracy calculation
        #score = sklearn.metrics.accuracy_score(y_test, pred)
        #print(f"Fold score (accuracy): {score}")

        # confusion matrix for rest fold
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



