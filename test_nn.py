import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import numpy as np

chars = [u"京",u"沪",u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z"]

epochnum = 0
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        print("begain train")

    def on_epoch_end(self, batch, logs={}):
        global epochnum
        epochnum+=1
        print("Epoch:%d" % epochnum)
        print('loss:'+str(logs.get('loss')))

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

def test(base_model):

    data,label = gen_data_label_data(False)
    y_pred = base_model.predict(data)
    shape = y_pred[:, :, :].shape
    out = K.get_value(K.ctc_decode(y_pred[:,:,:],input_length=np.ones(shape[0])*shape[1])[0][0])[:,
          :7]

    right_num = 0
    for i in range(len(data)):
        eco = len(chars)+1
        str_label=''.join([str(x) for x in label[i] if x!=eco])
        str_out = ''.join([str(x) for x in out[i] if x!=eco])
        if str_label == str_out:
            right_num+=1
    acc = (right_num / len(data)) * 100
    print("test acc is :{}%".format(str(acc)))

def gen_data_label_data(x):
    img_path = ""
    label_path = ""
    if x:
        img_path = "./data/pp_train.npy"
        label_path = './data/label_train.txt'
    else:
        img_path = "./data/pp_test.npy"
        label_path = './data/label_test.txt'
    #########data##############
    img_data = np.load(img_path)
    img_data = img_data.transpose(0,2,1,3)
    ###########label###################
    img_label = np.loadtxt(label_path)
    return img_data,img_label

def top_norm(x):
    t = x[:,:,0,:]
    return t
def bottom_norm(x):
    t = x[:,:,1,:]
    return t

def ctc_lamba_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:,:,:]
    return K.ctc_batch_cost(labels,y_pred,input_length,label_length)

if __name__ == '__main__':
    n_class = len(chars) + 1
    print(n_class)
    input_tensor = Input((96,64,3))
    x = input_tensor
    conv_shape = x.get_shape()

    x = Conv2D(64,(3,3),strides=(1,1),padding="same")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(128,(3,3),strides=(1,1),padding="same")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(256,(3,3),strides=(1,1),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256,(3,3),strides=(1,1),padding="same")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(512,(3,3),strides=(1,1),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512,(3,3),strides=(1,1),padding="same")(x)
    x = MaxPooling2D(pool_size=(1,2),strides=(1,2))(x)

    x = ZeroPadding2D(padding=(1,0), data_format='channels_last')(x)

    x = Conv2D(512, (3,3),strides=(1,1))(x)

    top = Lambda(top_norm)(x)
    top = Reshape(target_shape=(12,1,512))(top)
    bottom = Lambda(bottom_norm)(x)
    bottom = Reshape(target_shape=(12,1,512))(bottom)

    x = concatenate([top,bottom],1)
    x = Reshape(target_shape=(24,512))(x)

    gru_1 = GRU(128,return_sequences=True,kernel_initializer='he_normal',name='gru1')(x)
    gru_1b = GRU(128,return_sequences=True,go_backwards=True,kernel_initializer='he_normal', name='gru_1b')(
        x)

    gru1_merged = add([gru_1,gru_1b])

    gru_2 = GRU(256,return_sequences=True,kernel_initializer='he_normal',name='gru2')(gru1_merged)
    gru_2b = GRU(256,return_sequences=True,go_backwards=True,kernel_initializer='he_normal', name='gru_2b')(
        gru1_merged)

    x = concatenate([gru_2,gru_2b])

    x = Dropout(0.25)(x)

    y_pred = Dense(n_class,kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs = input_tensor, outputs = y_pred)
    base_model.load_weights('./Model/e2e_model.h5')
    test(base_model)