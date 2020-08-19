# For continuous training

from All_functions import *
from Data_aug_alff_for2 import *
from resnet3d import Resnet3DBuilder 



strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

checkpoint_dir = '/home/user/venv/kaiyi_venv/training_checkpoints/20200628-134424/weights.49.hdf5'

os.chdir('/home/user/venv/kaiyi_venv/performance/')

with strategy.scope():
    model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 2,reg_factor=1e-5)
    print("Load a previously trained model")
    model.load_weights(checkpoint_dir)
    #opt = tf.keras.optimizers.Adam(lr=1e-2)
    model.compile(optimizer= opt3 , loss=loss, metrics=['sparse_categorical_accuracy'])

    predict2 = model.predict(All_alff_val)

    cm2 = confusion_matrix(All_labels_val, np.argmax(predict2,axis=1))

    fig, ax = plot_confusion_matrix(conf_mat=cm2,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names=None)
    fig.savefig('confusion_matrix_val.png')
    
    plt.close()