
from All_functions import * 

# add bias_regulariser

def Conv_block_incep(input_tensors,num_filters,param):
    batch10 = BatchNormalization()(input_tensors)
    conv00 = Conv3D(num_filters,(7,7,7),strides=(2,2,2),padding='same',activation='relu', kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=param,l2=param), bias_regularizer=l1_l2(l1=param,l2=param))(batch10)
    pool11 = MaxPooling3D(pool_size=(2, 2, 2))(conv00)
    conv11 = Conv3D(num_filters, (3,3,3), padding='same', activation='relu', kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(pool11)
    conv12 = Conv3D(num_filters, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(conv11)
    batch2 = BatchNormalization()(conv12)
    conv13 = Conv3D(num_filters, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(batch2)
    conv14 = Conv3D(num_filters, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(conv13)
    conv15 = Conv3D(num_filters*2, (3,3,3),strides=(2,2,2),padding='same',activation='relu', kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(conv14)
    batch3 = BatchNormalization()(conv15)
    inc = inc_module_A(batch3,num_filters*2,param)
    inc = inc_module_A(inc,num_filters*2,param)
    inc = inc_module_A(inc,num_filters*2,param)
    inc = inc_module_A(inc,num_filters*2,param)
    inc = inc_module_A(inc,num_filters*2,param)
    reduction = half_reduction(inc,num_filters*2)
    inc = inc_module_B(reduction,num_filters*4,param)
    inc = inc_module_B(inc,num_filters*4,param)
    inc = inc_module_B(inc,num_filters*4,param)
    inc = inc_module_B(inc,num_filters*4,param)
    reduction=half_reduction(inc,num_filters*4)
    inc = inc_module_C(reduction,num_filters*8,param)
    inc = inc_module_C(inc,num_filters*8,param)
    final_output = GlobalAveragePooling3D()(inc)
    return final_output




def model_df_inc(num_classes,num_filters,shape,shape2,param):
    
    input = Input(shape=shape)
    final_output = Conv_block_incep(input, num_filters, param)

    input2 = Input(shape=shape2)
    final_output2 = Conv_block_incep(input2, num_filters, param)

    feature_conca= tf.keras.layers.concatenate([final_output,final_output2])
    Dropout_regulise= Dropout(0.5)(feature_conca)
    output = Dense(num_classes, activation='softmax')(Dropout_regulise)
    model = Model(inputs=[input,input2], outputs=output, name='model_df_inc')
    print(model.summary())
    return model


model = model_df_inc(3,32,(64,64,64,2),(128,128,128,2),0.0001)

model_structure(model)
