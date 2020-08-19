from All_functions import *

# Load model
#model_path = input('Giving your model path:')
#model = load_model(model_path)

def Conv_block_incep_SE(input_tensors,num_filters,param,ratio):
    #batch10 = BatchNormalization()(input_tensors)
    x = Conv3D(num_filters,(7,7,7),strides=(2,2,2),padding='same',activation='relu', kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input_tensors)
    for i in range(3):
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

# Load model weights
shape = (128, 128, 128, 1)
#model = model_df_inc(3,16,shape ,1e-4,8,0.5)
#model_structure(model)

def base_model_creator(model, train_para = False):
    base_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-2).output])
    base_model.trainable = train_para
    return base_model

# Eliminate the last layer of model
from resnet3d import Resnet3DBuilder 
model = Resnet3DBuilder.build_resnet_50((128, 128, 128, 1), 1,reg_factor=1e-4)

base_model = base_model_creator(model, train_para = False)

# Defining the new model
input_layer = tf.keras.Input(shape = shape)
x = base_model(input_layer)
x = Dropout(0.5)(x)
outputs = keras.layers.Dense(3, activation='softmax')(x)
new_model = keras.Model(input_layer, outputs)
print(new_model.summary())

