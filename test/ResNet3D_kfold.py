# Resnet 3D model 
from Data_alff_BAHC import *
import kerastuner as kt
from resnet3d import Resnet3DBuilder 

def model_builder(hp):
    hp_reg_factor = hp.Choice('reg_factor', values = [1e-2, 1e-3, 1e-4, 1e-5]) 
    model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 2,reg_factor=hp_reg_factor)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4,1e-5])
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate),
                loss = loss,
                metrics=['sparse_categorical_accuracy'])
    return model

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))



kf = StratifiedKFold(5, shuffle=True, random_state=None) # Use for StratifiedKFold classification

fold = 0
hp_dir = '/home/user/venv/kaiyi_venv/hp_logs'

with strategy.scope():
    tuner = kt.Hyperband(model_builder,
                        objective = 'val_sparse_categorical_accuracy',
                        max_epochs = 100,
                        directory = hp_dir,
                        project_name = 'Tuninig_BAHC')

    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    print(f"""
    The hyperparameter search is complete. The optimal number of reg_factor is {best_hps.get('reg_factor')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)
    log_fold = (logdir + "/Res_50_fold_BAHC")
    os.mkdir(log_fold)
    os.chdir(log_fold)
    print('log file saved to'+ log_fold)

    history = model.fit(All_alff_tr,
                        All_labels_tr,
                        batch_size=5,
                        epochs=200,
                        verbose=1,
                        callbacks=None,
                        validation_split=0.1)
    
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('history.csv')
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

    """
    predict = model.predict(x_train) 
    
    cm = confusion_matrix(y_train, np.argmax(predict,axis=1)) 
    os.chdir(log_fold) 
    fig, ax = plot_confusion_matrix(conf_mat=cm, 
                                    colorbar=True, 
                                    show_absolute=True, 
                                    class_names=None) 
    fig.savefig('confusion_matrix_tr.png') 
    plt.close()             

    # confusion matrix for val fold
    predict1 = model.predict(x_test) 
    
    cm1 = confusion_matrix(y_test, np.argmax(predict1,axis=1)) 
    os.chdir(log_fold) 
    fig, ax = plot_confusion_matrix(conf_mat=cm1, 
                                    colorbar=True, 
                                    show_absolute=True, 
                                    class_names=None) 
    fig.savefig('confusion_matrix_val.png') 
    plt.close()             
    """

    predict2 = model.predict(All_alff_val)

    cm2 = confusion_matrix(All_labels_val, np.argmax(predict2,axis=1))
    
    os.chdir(log_fold)

    fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names=None)
    fig.savefig('confusion_matrix_test.png')
    plt.close()