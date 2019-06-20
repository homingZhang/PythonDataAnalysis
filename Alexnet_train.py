from keras.models import *
from keras.layers import Input, Conv3D,BatchNormalization, MaxPooling3D,Dense,Flatten,Dropout,Activation
from keras.optimizers import *
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# 设置xian cun占用比例
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

epoch = 10


# import data
# import data
H5_load_path_CD = r"/media/root/ZJN/1_AlexNet/roi_brain/Fold5/train/Aug/CD"
H5_load_path_HC = r"/media/root/ZJN/1_AlexNet/roi_brain/Fold5/train/Aug/HC"
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
    fc1_drop = Dropout(rate=0.25)(fc1)
    fc2 = Dense(250, activation='relu', name = 'fc2')(fc1_drop)
    fc2_drop = Dropout(rate=0.25)(fc2)
    fc3 = Dense(2, name='fc3')(fc2_drop)
    output = Activation(activation='softmax')(fc3)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=SGD(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    return model

if __name__ == "__main__":
    d_model = Alexnet()
    # d_model.load_weights(r'/media/root/EAGET/CD/Save_net/Alexnet/Save_net_10000.h5', by_name=True)
    print(d_model.summary())
    Iter = 0
    for i in range(epoch):
        random.shuffle(Num_list)
        total_correct = 0
        for read_num in Num_list:
            Iter = Iter + 1
            read_name = H5_List[read_num]
            H5_file = h5py.File(read_name, 'r')
            batch_x = H5_file['Aug_data'][:]
            batch_y = H5_file['Aug_label'][:]
            y = np.reshape(batch_y, [1,2])
            data_input_1[0, :, :, :, 0] = batch_x[:, :, :]
            H5_file.close()
            result = d_model.train_on_batch(data_input_1, y)
            print('epoch',i,'Iter',Iter,result)
            print(d_model.predict_on_batch(data_input_1))

            total_correct = total_correct + result[-1]

            # if Iter % 1000 == 0 and Iter >= 15000:
            #     d_model.save('/AugCD_data/roi_brain/Fold3/train/Save_net/Save_net_' + str(Iter) + '.h5')
            #     print('model save')
            #     Accuracy = total_correct/1000
            #     total_correct = 0
            #     print(Accuracy)

            if Iter >= 15000:
                if Iter < 30000 and Iter % 1000 == 0:
                    d_model.save('/media/root/新加卷1/ZJN/Fold1/Save_net/Save_net_' + str(Iter) + '.h5')
                    print('model save')
                    # Accuracy = total_correct/1000
                    # total_correct = 0
                    # print(Accuracy)
                elif Iter >= 30000 and Iter % 500 == 0:
                    d_model.save('/media/root/新加卷1/ZJN/Fold1/Save_net/Save_net_' + str(Iter) + '.h5')
                    print('model save')
                elif Iter >= 70000:
                    input()