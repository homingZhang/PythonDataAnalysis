from keras.models import *
from keras.layers import Input, Conv3D,BatchNormalization, MaxPooling3D,Dense,Flatten,Dropout,Activation
from keras.optimizers import *
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
# 设置xian cun占用比例
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def sigmoid_y(x):
    if x < 0.5:
        x = 0
    else:
        x = 1
    return x

# import data
H5_load_path_CD = r"/media/root/ZJN/1_AlexNet/whole_brain/Fold_1/TEST/CD"
H5_load_path_HC = r"/media/root/ZJN/1_AlexNet/whole_brain/Fold_1/TEST/HC"
# H5_load_path_CD = r"/AugCD_data/train"

############### train data ##########################
file_name_CD = os.listdir(H5_load_path_CD)
file_name_HC = os.listdir(H5_load_path_HC)
H5_List = []
for file in file_name_CD:
    if file.endswith('.h5'):
        H5_List.append(os.path.join(H5_load_path_CD, file))
for file in file_name_HC:
    if file.endswith('.h5'):
        H5_List.append(os.path.join(H5_load_path_HC, file))

print(H5_List)

H5_num = len(H5_List)
print('The H5 files number:%d'%(H5_num))
Num_list = list(range(H5_num))
########################################################
data_input_1 = np.zeros([1, 121, 145, 121, 1], dtype=np.float32)


def Alexnet():
    inputs = Input(shape=(121, 145, 121, 1), name='input1')

    # 121x145x121
    conv1 = Conv3D(48, 7, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(inputs)
    pool1 = MaxPooling3D(pool_size=2, padding='same', name='pool1')(conv1)
    bn1 = BatchNormalization(axis=1, name='batch_normalization_1')(pool1)
    print("conv1 shape:", conv1.shape)
    print("pool1 shape:", pool1.shape)
    # conv1 shape: (?, 121, 145, 121, 48)
    # pool1 shape: (?, 61, 73, 61, 48)
    conv2 = Conv3D(128, 5, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(bn1)
    pool2 = MaxPooling3D(pool_size=2, padding='same', name='pool2')(conv2)
    bn2 = BatchNormalization(axis=1, name='batch_normalization_2')(pool2)
    print("conv2 shape:", conv2.shape)
    print("pool2 shape:", pool2.shape)
    # conv2 shape: (?, 61, 73, 61, 128)
    # pool2 shape: (?, 31, 37, 31, 128)
    conv3 = Conv3D(192, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(bn2)
    bn3 = BatchNormalization(axis=1, name='batch_normalization_3')(conv3)
    print("conv3 shape:", conv3.shape)
    # conv3 shape: (?, 31, 37, 31, 192)
    conv4 = Conv3D(192, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(bn3)
    bn4 = BatchNormalization(axis=1, name='batch_normalization_4')(conv4)
    print("conv4 shape:", conv4.shape)
    # conv3 shape: (?, 31, 37, 31, 192)
    conv5 = Conv3D(128, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5')(bn4)
    pool3 = MaxPooling3D(pool_size=3, padding='same', name='pool3')(conv5)
    bn5 = BatchNormalization(axis=1, name='batch_normalization_5')(pool3)
    print("conv5 shape:", conv5.shape)
    print("pool3 shape:", pool3.shape)
    # conv5 shape: (?, 31, 37, 31, 128)
    # pool3 shape: (?, 11, 13, 11, 128)

    flatten1 = Flatten()(bn5)
    fc1 = Dense(500, activation='relu', name = 'fc1')(flatten1)
    # fc1_drop = Dropout(rate=0.25)(fc1)
    fc1_drop = Dropout(rate=0)(fc1)
    fc2 = Dense(250, activation='relu', name = 'fc2')(fc1_drop)
    # fc2_drop = Dropout(rate=0.25)(fc2)
    fc2_drop = Dropout(rate=0)(fc2)

    fc3 = Dense(2, name='fc3')(fc2_drop)
    output = Activation(activation='softmax')(fc3)

    model = Model(input=inputs, output=output)
    # model.compile(optimizer=SGD(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer=SGD(0), loss='categorical_crossentropy', metrics=['acc'])
    return model

if __name__ == "__main__":
    H5_load_path_model = r"/AugCD_data/roi_brain/Fold3/train/Save_net"
    file_name_model = os.listdir(H5_load_path_model)
    Num_list_model = len(file_name_model)
    for model_num in range(Num_list_model):

        d_model = Alexnet()
        model_file_path = H5_load_path_model + '/' + file_name_model[model_num]
        d_model.load_weights(model_file_path, by_name=True)
        print(d_model.summary())
        # test
        total_correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for read_num in Num_list:
            # Iter = Iter + 1
            read_name = H5_List[read_num]
            H5_file = h5py.File(read_name, 'r')
            batch_x = H5_file['data'][:]
            batch_y = H5_file['label'][:]
            y = np.reshape(batch_y, [1, 2])
            data_input_1[0, :, :, :, 0] = batch_x[:, :, :]
            H5_file.close()
            result = d_model.train_on_batch(data_input_1, y)

            result_pre = d_model.predict_on_batch(data_input_1)
            result_pre[:, 0] = sigmoid_y(result_pre[:, 0])
            result_pre[:, 1] = sigmoid_y(result_pre[:, 1])

            if (y[0, 0] == 0) and (y[0, 0] == result_pre[0, 0]):
                TP = TP + 1
            elif (y[0, 0] == 1) and (y[0, 0] == result_pre[0, 0]):
                TN = TN + 1
            elif (y[0, 0] == 0) and (result_pre[0, 0] == 1):
                FN = FN + 1
            elif (y[0, 0] == 1) and (result_pre[0, 0] == 0):
                FP = FP + 1

            print('Sample_name', read_name)
            # print('Sample_label', y)
            # print('Sample_pre_label', result_pre)
            print('num', read_num, result)
            # print(d_model.predict_on_batch(data_input_1))
            # accuracy, sensitivity, specificity

        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        print('Sensitivity', Sensitivity)
        print('Specificity', Specificity)
        print('Accuracy', Accuracy)
        Result_save_Path = '/AugCD_data/roi_brain/Fold3/train/Result'
        Final_Path = Result_save_Path + str(file_name_model[model_num]) + '.txt'
        with open(Final_Path, 'w') as f:
            f.write(str(Sensitivity))
            f.write(str(Specificity))
            f.write(str(Accuracy))

        K.clear_session()