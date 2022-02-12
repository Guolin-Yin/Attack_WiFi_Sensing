'''Tensorflow'''
from tensorflow.keras.layers import Dense, Input, Softmax,ZeroPadding2D,MaxPooling2D,Conv2D,Flatten,Lambda,Dropout,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
'''Pytorch'''
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
# from torchsummary import summary
import numpy as np
import sys
import os

current_dir = os.getcwd( )
sys.path.append( current_dir )
sys.path.append( current_dir + '/utils' )
sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig
gpus = tf.config.experimental.list_physical_devices( 'GPU' )
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth( gpu, True )
    except RuntimeError as e:
        print( e )
'''https://www.tensorflow.org/tutorials/generative/adversarial_fgsm'''
# class AlexNetTorch(nn.Module):
#     def __init__( self ):
#         super( AlexNetTorch, self ).__init__( )
#         self.conv1 = nn.Conv2d( in_channels = 3, out_channels = 96, kernel_size = (11, 5), stride = 2, padding = 0 )
#         self.maxpool = nn.MaxPool2d( kernel_size = 3, stride = 1 )
#         self.conv2 = nn.Conv2d( in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2 )
#         self.conv3 = nn.Conv2d( in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1 )
#         self.conv4 = nn.Conv2d( in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1 )
#         self.conv5 = nn.Conv2d( in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1 )
#         # self.fc1 = nn.Linear( in_features = 9216, out_features = 4096 )
#         # self.fc2 = nn.Linear( in_features = 4096, out_features = 4096 )
#         # self.fc3 = nn.Linear( in_features = 4096, out_features = 10 )
#
#     def forward( self, x ):
#         x = F.relu( self.conv1( x ) )
#         x = self.maxpool( x )
#         x = F.relu( self.conv2( x ) )
#         x = self.maxpool( x )
#         x = F.relu( self.conv3( x ) )
#         x = F.relu( self.conv4( x ) )
#         x = F.relu( self.conv5( x ) )
#         x = self.maxpool( x )
#         # x = x.reshape( x.shape[ 0 ], -1 )
#         # x = F.relu( self.fc1( x ) )
#         # x = F.relu( self.fc2( x ) )
#         # x = self.fc3( x )
#         return x
class AlexNetTF:
    def __init__( self,config=None ):
        self.config = config
        # self.initGPU()
    def buildModel( self ,):
        input = Input( self.config.input_shape, name = 'input_layer' )
        conv_1 = Conv2D(
                filters = 96, kernel_size = (11, 5), strides = 2, input_shape = self.config.input_shape,
                padding = 'valid',
                activation = 'relu', name = 'Conv_1'
                )( input )
        MP_1 = MaxPooling2D( pool_size = 3, strides = 1, name = 'Maxpool_1' )( conv_1 )

        PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
        conv_2 = Conv2D( filters = 256, kernel_size = 5, strides = 1, padding = 'valid', name = 'Conv_2' )( PD_1 )
        MP_2 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_2' )( conv_2 )
        Padding_leayer_2 = ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' )( MP_2 )
        Conv_3 = Conv2D(
                filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                name = 'Conv_3'
                )( Padding_leayer_2 )
        Padding_layer_3 = ZeroPadding2D( padding = 1, name = 'Padding_layer_3' )( Conv_3 )
        Conv_4 = Conv2D(
                filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                name = 'Conv_4'
                )( Padding_layer_3 )
        Padding_layer_4 = ZeroPadding2D( padding = 1, name = 'Padding_layer_4' )( Conv_4 )
        Conv_5 = Conv2D(
                filters = 256, activation = 'relu', kernel_size = (4, 3), strides = 1, padding = 'valid',
                name = 'Conv_5'
                )( Padding_layer_4 )
        Maxpool_3 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_3' )( Conv_5 )
        dp = Dropout( 0.5 )( Maxpool_3 )
        ft = Flatten( )( dp )
        FC_1 = Dense( units = 256, name = 'FC_1' )( ft )
        FC_2 = Dense( units = 1280, name = 'FC_2' )( FC_1 )
        output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ),name = 'lambda_layer' )( FC_2 )
        fc = Dense( units=self.config.N_classes, name="fine_tune_layer" )( output )
        output = Softmax( )( fc )
        Net = Model( inputs=input, outputs=output )
        return Net

def load_tf_model(path:str = "/content/drive/MyDrive/Colab Notebooks/AdversarialAttack/signFi_model"):
  return tf.keras.models.load_model(path)
def create_adversarial_pattern(input_CSI, label, pretrained_model, t_label: int = None):
    loss_object = tf.keras.losses.CategoricalCrossentropy( )
    input_CSI = tf.convert_to_tensor(input_CSI, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    if t_label:
        t_label = tf.convert_to_tensor(np.expand_dims(to_categorical( t_label-1, num_classes = 6 ),axis=0),
                dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch( input_CSI )
            prediction = pretrained_model( input_CSI )
            loss = loss_object( label, prediction ) - loss_object( t_label, prediction )
    else:
        with tf.GradientTape() as tape:
            tape.watch( input_CSI )
            prediction = pretrained_model( input_CSI )
            loss = loss_object( label, prediction )
    # Get the gradients of the loss w.r.t to the input CSI.
    gradient = tape.gradient( loss, input_CSI )
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad
def runTrain(config, dataset_name):
    # name = 'widar'
    X_train, X_test, y_train, y_test = gestureDataLoader.getData(config, dataset_name)
    net = AlexNetTF( config )
    Net = net.buildModel( )
    lrScheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1,
            patience=20,
    )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor='val_acc', patience=50, restore_best_weights=True )

    optimizer = tf.keras.optimizers.Adamax(
            learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,
            name='Adamax'
    )
    Net.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
    Net.summary( )
    history = Net.fit(
            X_train, y_train,
            validation_split=0.05,
            batch_size = config.batch_size,
            epochs=1000,
            callbacks=[ earlyStop, lrScheduler ]
    )
    Net.evaluate(X_test, y_test)
    config.test_data = X_test
    config.test_label = y_test
    Net.save( config.pretrained_model_path )
def runTest(input_CSI,labels,pretrained_model):
    pre_label = np.argmax(pretrained_model.predict(input_CSI),axis=1)
    true_label = np.argmax(labels,axis=1)
    Correct_count = np.sum(pre_label == true_label)
    accuracy = Correct_count/labels.shape[0]
    print(f'The accuracy is {accuracy}')
def runAdvExsTest(input_CSI,labels,pretrained_model,eps,ifpltcmd:bool =False,t_label:int=None):
    '''
    labels: should be one hot coded
    '''
    label_adv_pred_container = []
    for i,data in enumerate(input_CSI):
        data,current_label = np.expand_dims(data,axis=0),np.expand_dims(labels[i],axis=0)
        ori_pred_label = np.argmax(pretrained_model.predict(data),axis = 1)
        if ori_pred_label == np.argmax( current_label, axis=1 ):
            advData = data + eps*create_adversarial_pattern(data,current_label,pretrained_model,t_label = t_label)
            label_adv_pred_container.append( np.argmax( pretrained_model.predict( advData ), axis = 1 ) )
        else:
            label_adv_pred_container.append(ori_pred_label)
    true_label = np.argmax( labels, axis = 1 )
    label_adv_pred_container = np.squeeze(np.asarray( label_adv_pred_container ))
    accuracy = np.sum( label_adv_pred_container == true_label ) / labels.shape[0 ]
    if ifpltcmd:
      title = f'eps: {eps}, Accuracy: {accuracy:.2f}, target: {t_label}'
      plotSig.pltcm( label_test_pred = label_adv_pred_container, true_label = true_label, title = title )
    print( f'Accuracy for ep = {eps:.3f} is {accuracy:.2f}' )
    return accuracy
if __name__ == '__main__':
    current_dir = os.getcwd( )
    sys.path.append( current_dir )
    sys.path.append( current_dir + '/utils' )
    sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
    import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig
    config = Config.getconfig( )
    '''Prepare data'''
    # choose dataset
    dataset_name = 'widar'
    _, config.test_data, _, config.test_label = gestureDataLoader.getData(config, dataset_name)

    '''Run training process'''
    # runTrain( config = config, dataset_name = 'widar' )
    '''Load pretrained model'''
    config.pretrained_model_path = 'SavedModel/widar_model_loc[2]_ori[2]Rx123456'
    pretrained_model = tf.keras.models.load_model(config.pretrained_model_path )
    pretrained_model.summary()
    '''Evaluation of adversarial samples'''
    accuracy_attack_free = pretrained_model.evaluate(config.test_data, config.test_label, verbose=1)
    acc = runAdvExsTest(
                        input_CSI = config.test_data,
                        labels = config.test_label,
                        pretrained_model = pretrained_model,
                        eps = 3,
                        ifpltcmd = True
                        )
    # all_acc = []
    # for ep in np.arange( 0, 3, 0.05 ):
    #     accuracy = DeepNet.runAdvExsTest(
    #             input_CSI = config.test_data,
    #             labels = config.test_label,
    #             pretrained_model = pretrained_model,
    #             eps = ep
    #             )
    #
    #     all_acc.append( accuracy )
    '''visualisation of adversarial samples'''
    # X_test = config.test_data[ 0:1 ]
    # y_test = config.test_label[ 0:1 ]
    # eps = np.arange( 0.5, 1.5, 0.2 )
    # range_adv = []
    # for ep in eps:
    #     advData = X_test + ep * create_adversarial_pattern( X_test, y_test, pretrained_model )
    #     range_adv.append(advData[0,:,0,0])
    # plotSig.showSignal(X_test[0,:,0,0],range_adv,eps = list(eps))
    # Spectrogram
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.pcolormesh(config.test_data[0,:,:,1,0])
    # advData = config.test_data[0:1] + 0.1 * create_adversarial_pattern( config.test_data[0:1], config.test_label[0:1],
    #         pretrained_model )
    # plt.figure( )
    # plt.pcolormesh(advData[0,:,:,1,0])
    # label_test_pred = model.predict( data_test )