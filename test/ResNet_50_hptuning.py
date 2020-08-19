# Resnet 3D model 
from All_functions import *
from Data_split_with3 import *
import IPython
import kerastuner as kt
from resnet3d import Resnet3DBuilder 

#os.mkdir(logdir)
#os.mkdir(checkpoint_dir)
hp_dir = '/home/user/venv/kaiyi_venv/hp_logs'
strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

def model_builder(hp):
    hp_reg_factor = hp.Choice('reg_factor', values = [1e-2, 1e-3, 1e-4, 1e-5]) 
    model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 3,reg_factor=hp_reg_factor)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4,1e-5])
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate),
                  loss = loss,
                  metrics=['sparse_categorical_accuracy'])
    return model

print('So far so good')



tuner = kt.Hyperband(model_builder,
                     objective = 'val_sparse_categorical_accuracy',
                     max_epochs = 100,
                     directory = hp_dir,
                     project_name = 'trial_Data_split_with3')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

tuner.search(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of reg_factor is {best_hps.get('reg_factor')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, 
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
                                                save_freq='epoch')]


model = tuner.hypermodel.build(best_hps)
model.fit(train_dataset, epochs = 200, callbacks=callbacks, validation_data = val_dataset)

# performance, use tf.math.confusion matrix?
predict2 = model.predict(All_alff_val)

cm2 = confusion_matrix(All_labels_val, np.argmax(predict2,axis=1))

os.chdir(logdir)

fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=None)
fig.savefig('confusion_matrix_test.png')

plt.close()

"""
predict2 = model.predict(All_alff_test)

cm2 = confusion_matrix(All_labels_test, np.argmax(predict2,axis=1))

fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=None)
fig.savefig('confusion_matrix_test.png')

plt.close()
"""

model.save('Res_50_3D_unaug.h5')