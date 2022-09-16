'''Tensorflow'''
from tensorflow.keras.layers import Dense, Input, Softmax,ZeroPadding2D,MaxPooling2D,Conv2D,Flatten,\
    GlobalAveragePooling2D,Lambda,Dropout,LSTM,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.python.ops.gen_experimental_dataset_ops import dataset_to_tf_record
from tqdm import tqdm
'''Pytorch'''
# import torch
# import torch.nn as nn
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import random_split
# import torch.nn.functional as F
# from torchsummary import summary
import numpy as np
import re
import h5py
from Experiments import *
import sys
import os
import copy
for data_root, data_dirs, data_files in os.walk( os.getcwd( ) ):
    for rt in data_dirs:
        sys.path.append( os.path.join(data_root,rt) )
import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig, TOOLS
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from DeepFool import deepfool
from Universal_pert import universal_perturbation
# from Experiments import scaleDeepfool
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
config = Config.getconfig( )
procOBJ = gestureDataLoader.preprocessing( )
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if logs.get('val_acc') > 0.95 and logs.get('val_loss')<0.1:
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(0.95*100))
			self.model.stop_training = True
class AlexNetTF:
    def __init__( self,config=None ):
        self.config = config
        # self.initGPU()
    def buildModel( self , choice = 'defult'):
        if choice == 'defult':
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
        elif choice == 'alex1':
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
            FC_2 = Dense( units = 2560, name = 'FC_2' )( FC_1 )
            output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ),name = 'lambda_layer' )( FC_2 )
            fc = Dense( units=self.config.N_classes, name="fine_tune_layer" )( output )
            output = Softmax( )( fc )
        elif choice == 'alex2':
            input = Input( self.config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 48, kernel_size = (11, 5), strides = 2, input_shape = self.config.input_shape,
                    padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )( input )
            MP_1 = MaxPooling2D( pool_size = 3, strides = 1, name = 'Maxpool_1' )( conv_1 )

            PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
            conv_2 = Conv2D( filters = 128, kernel_size = 5, strides = 1, padding = 'valid', name = 'Conv_2' )( PD_1 )
            MP_2 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_2' )( conv_2 )
            Padding_leayer_2 = ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' )( MP_2 )
            Conv_3 = Conv2D(
                    filters = 264, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                    name = 'Conv_3'
                    )( Padding_leayer_2 )
            Padding_layer_3 = ZeroPadding2D( padding = 1, name = 'Padding_layer_3' )( Conv_3 )
            Conv_4 = Conv2D(
                    filters = 128, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                    name = 'Conv_4'
                    )( Padding_layer_3 )
            Padding_layer_4 = ZeroPadding2D( padding = 1, name = 'Padding_layer_4' )( Conv_4 )
            Conv_5 = Conv2D(
                    filters = 128, activation = 'relu', kernel_size = (4, 3), strides = 1, padding = 'valid',
                    name = 'Conv_5'
                    )( Padding_layer_4 )
            Maxpool_3 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_3' )( Conv_5 )
            dp = Dropout( 0.5 )( Maxpool_3 )
            ft = Flatten( )( dp )
            FC_1 = Dense( units = 200, name = 'FC_1' )( ft )
            FC_2 = Dense( units = 1024, name = 'FC_2' )( FC_1 )
            output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ),name = 'lambda_layer' )( FC_2 )
            fc = Dense( units=self.config.N_classes, name="fine_tune_layer" )( output )
            output = Softmax( )( fc )
        elif choice == 'alex3':
            input = Input( self.config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 48, kernel_size = (10, 4), strides = 1, input_shape = self.config.input_shape,
                    padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )( input )
            MP_1 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_1' )( conv_1 )

            PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
            conv_2 = Conv2D( filters = 128, kernel_size = 4, strides = 2, padding = 'valid', name = 'Conv_2' )( PD_1 )
            MP_2 = MaxPooling2D( pool_size = 4, strides = 1, name = 'Maxpool_2' )( conv_2 )
            Padding_leayer_2 = ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' )( MP_2 )
            Conv_3 = Conv2D(
                    filters = 264, activation = 'relu', kernel_size = 4, strides = 2, padding = 'valid',
                    name = 'Conv_3'
                    )( Padding_leayer_2 )
            Padding_layer_3 = ZeroPadding2D( padding = 1, name = 'Padding_layer_3' )( Conv_3 )
            Conv_4 = Conv2D(
                    filters = 128, activation = 'relu', kernel_size = 4, strides = 1, padding = 'valid',
                    name = 'Conv_4'
                    )( Padding_layer_3 )
            Padding_layer_4 = ZeroPadding2D( padding = 1, name = 'Padding_layer_4' )( Conv_4 )
            Conv_5 = Conv2D(
                    filters = 128, activation = 'relu', kernel_size = (3, 3), strides = 1, padding = 'valid',
                    name = 'Conv_5'
                    )( Padding_layer_4 )
            Maxpool_3 = MaxPooling2D( pool_size = 4, strides = 1, name = 'Maxpool_3' )( Conv_5 )
            dp = Dropout( 0.5 )( Maxpool_3 )
            ft = Flatten( )( dp )
            FC_1 = Dense( units = 200, name = 'FC_1' )( ft )
            FC_2 = Dense( units = 1024, name = 'FC_2' )( FC_1 )
            output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( FC_2 )
            fc = Dense( units = self.config.N_classes, name = "fine_tune_layer" )( output )
            output = Softmax( )( fc )
        elif choice == 'cnn':
            input = Input( self.config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 48, kernel_size = (10, 4), strides = 1, input_shape = self.config.input_shape,
                    padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )( input )
            MP_1 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_1' )( conv_1 )

            PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
            conv_2 = Conv2D( filters = 128, kernel_size = 4, strides = 2, padding = 'valid', name = 'Conv_2' )( PD_1 )
            MP_2 = MaxPooling2D( pool_size = 4, strides = 1, name = 'Maxpool_2' )( conv_2 )
            Padding_leayer_2 = ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' )( MP_2 )
            Conv_3 = Conv2D(
                    filters = 264, activation = 'relu', kernel_size = 4, strides = 2, padding = 'valid',
                    name = 'Conv_3'
                    )( Padding_leayer_2 )
            ft = Flatten( )( Conv_3 )
            FC_1 = Dense( units = 200, name = 'FC_1' )( ft )
            FC_2 = Dense( units = 1024, name = 'FC_2' )( FC_1 )
            l2 = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( FC_2 )
            fc = Dense( units = self.config.N_classes, name = "fine_tune_layer" )( l2 )
            output = Softmax( )( fc )
        # elif choice == 'lstm':
        #     input = Input( self.config.input_shape, name = 'input_layer' )
        #
        #     x = LSTM( 256, return_sequences = True )( input )
        #     x = LSTM( 256, return_sequences = True )( x )
        #     x = MaxPooling2D( pool_size = 2, strides = 1, name = 'Maxpool_2' )( x )
        #     output = Dense( self.config.N_classes, activation = "softmax" )( x )

            # model = Model( inputs = inputs, outputs = outputs )
        elif choice == 'cnnlstm':
            input = Input( self.config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 48, kernel_size = (10, 4), strides = 1, input_shape = self.config.input_shape,
                    padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )( input )
            MP_1 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_1' )( conv_1 )
            PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
            conv_2 = Conv2D( filters = 128, kernel_size = 4, strides = 2, padding = 'valid', name = 'Conv_2' )( PD_1 )
            MP_2 = MaxPooling2D( pool_size = 4, strides = 1, name = 'Maxpool_2' )( conv_2 )
            hidden = tf.keras.layers.Reshape( (-1, 128) )( MP_2 )
            # f = Flatten()(MP_2)
            # td = TimeDistributed( MP_2 )( input )
            lstm1 = LSTM( 64, return_sequences = True)( hidden )
            lstm2 = LSTM( 96, return_sequences = True )( lstm1 )
            f = Flatten( )( lstm2 )
            FC_1 = Dense( units = 200, name = 'FC_1' )( f )
            FC_2 = Dense( units = self.config.N_classes, name = 'FC_2' )( FC_1 )
            output = Softmax( )( FC_2 )
        elif choice == 'densenet121':
            model_d = tf.keras.applications.DenseNet121(
                    weights = None, include_top = False, input_shape = self.config.input_shape
                    )
            input = model_d.input


            x = model_d.output

            x = GlobalAveragePooling2D( )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )

            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'densenet169':
            model_d = tf.keras.applications.DenseNet169(
                    weights = None, include_top = False, input_shape = self.config.input_shape
                    )
            input = model_d.input


            x = model_d.output

            x = GlobalAveragePooling2D( )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )

            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'vgg16':
            model_d = tf.keras.applications.vgg16.VGG16(
                    weights = None, include_top = False, input_shape = (200, 60, 3)
                    )
            input = model_d.input
            x = model_d.output
            x = GlobalAveragePooling2D( )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'vgg19':
            model_d = tf.keras.applications.vgg19.VGG19(
                    weights = None, include_top = False, input_shape = (200, 60, 3)
                    )
            input = model_d.input
            x = model_d.output
            x = GlobalAveragePooling2D( )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'vgg13':

            # define model input
            input = Input( shape = self.config.input_shape)
            # add vgg module
            model = self.vgg_block( input, 64, 2 )
            model = self.vgg_block( model, 128, 2 )
            model = self.vgg_block( model, 256, 3 )
            model = self.vgg_block( model, 512, 3 )



            x = GlobalAveragePooling2D( )( model )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'vgg10':

            # define model input
            input = Input( shape = self.config.input_shape)
            # add vgg module
            model = self.vgg_block( input, 64, 2 )
            model = self.vgg_block( model, 128, 2 )
            model = self.vgg_block( model, 256, 3 )




            x = GlobalAveragePooling2D( )( model )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'vgg8':

            # define model input
            input = Input( shape = self.config.input_shape)
            # add vgg module
            model = self.vgg_block( input, 64, 2 )

            model = self.vgg_block( model, 256, 3 )



            x = GlobalAveragePooling2D( )( model )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            output = Dense( self.config.N_classes, activation = 'softmax' )( x )
        elif choice == 'vgg5':

            # define model input
            input = Input( shape = self.config.input_shape)
            # add vgg module
            model = self.vgg_block( input, 256, 3 )


            # x = model.output
            x = GlobalAveragePooling2D( )( model )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            x = Dense( 1024, activation = 'relu' )( x )
            # x = Dense( 512, activation = 'relu' )( x )
            x = BatchNormalization( )( x )
            x = Dropout( 0.5 )( x )
            output = Dense( self.config.N_classes, activation = 'softmax' )( x )


        elif choice == 'resnet':
            input = Input( self.config.input_shape, name = 'input_layer' )
            x = tf.keras.layers.Conv2D( 64, 5, activation = 'relu', padding = 'same' )( input )

            x = tf.keras.layers.MaxPooling2D( pool_size = (2, 2) )( x )

            x = self.resblock( x, 3, 64 )
            x = self.resblock( x, 3, 64 )
            # x = resblock(x, 3, 128)

            x = self.resblock( x, 3, 128, first_layer = True )
            x = self.resblock( x, 3, 128 )
            x = Flatten( )( x )
            x = Dense( units = 200, name = 'FC_1' )( x )
            x = Dense( units = 1024, name = 'FC_2' )( x )
            x = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( x )
            x = Dense( units = self.config.N_classes, name = "fine_tune_layer" )( x )
            output = Softmax( )( x )
        elif choice == 'resnet6':
            input = Input( self.config.input_shape, name = 'input_layer' )
            x = tf.keras.layers.Conv2D( 64, 5, activation = 'relu', padding = 'same' )( input )

            x = tf.keras.layers.MaxPooling2D( pool_size = (2, 2) )( x )

            x = self.resblock( x, 3, 64 )
            x = self.resblock( x, 3, 64 )
            # x = resblock(x, 3, 128)

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )


            x = self.resblock( x, 3, 128, first_layer = True )
            x = self.resblock( x, 3, 128 )
            x = Flatten( )( x )
            x = Dense( units = 200, name = 'FC_1' )( x )
            x = Dense( units = 1024, name = 'FC_2' )( x )
            x = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( x )
            x = Dense( units = self.config.N_classes, name = "fine_tune_layer" )( x )
            output = Softmax( )( x )
        elif choice == 'resnet10':
            input = Input( self.config.input_shape, name = 'input_layer' )
            x = tf.keras.layers.Conv2D( 64, 5, activation = 'relu', padding = 'same' )( input )

            x = tf.keras.layers.MaxPooling2D( pool_size = (2, 2) )( x )

            x = self.resblock( x, 3, 64 )
            x = self.resblock( x, 3, 64 )
            # x = resblock(x, 3, 128)

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )


            x = self.resblock( x, 3, 128, first_layer = True )
            x = self.resblock( x, 3, 128 )
            x = Flatten( )( x )
            x = Dense( units = 200, name = 'FC_1' )( x )
            x = Dense( units = 1024, name = 'FC_2' )( x )
            x = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( x )
            x = Dense( units = self.config.N_classes, name = "fine_tune_layer" )( x )
            output = Softmax( )( x )
        elif choice == 'resnet12':
            input = Input( self.config.input_shape, name = 'input_layer' )
            x = tf.keras.layers.Conv2D( 64, 5, activation = 'relu', padding = 'same' )( input )

            x = tf.keras.layers.MaxPooling2D( pool_size = (2, 2) )( x )

            x = self.resblock( x, 3, 64 )
            x = self.resblock( x, 3, 64 )
            # x = resblock(x, 3, 128)

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )

            x = self.resblock( x, 3, 128, first_layer = True  )
            x = self.resblock( x, 3, 128 )


            x = self.resblock( x, 3, 128, first_layer = True )
            x = self.resblock( x, 3, 128 )
            x = Flatten( )( x )
            x = Dense( units = 200, name = 'FC_1' )( x )
            x = Dense( units = 1024, name = 'FC_2' )( x )
            x = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( x )
            x = Dense( units = self.config.N_classes, name = "fine_tune_layer" )( x )
            output = Softmax( )( x )
        Net = Model( inputs=input, outputs=output )
        return Net

    def vgg_block(self, layer_in, n_filters , n_conv ):
        # add convolutional layers
        for _ in range( n_conv ):
            layer_in = Conv2D( n_filters, (3, 3), padding = 'same', activation = 'relu' )( layer_in )
        # add max pooling layer
        layer_in = MaxPooling2D( (2, 2), strides = (2, 2) )( layer_in )
        return layer_in
    def resblock( self, x, kernelsize, filters, first_layer = False ):

        if first_layer:
            fx = tf.keras.layers.Conv2D( filters, kernelsize, padding = 'same' )( x )
            # fx = layers.BatchNormalization()(fx)
            fx = tf.keras.layers.ReLU( )( fx )

            fx = tf.keras.layers.Conv2D( filters, kernelsize, padding = 'same' )( fx )
            # fx = layers.BatchNormalization()(fx)

            x = tf.keras.layers.Conv2D( filters, 1, padding = 'same' )( x )

            # out = tf.keras.layers.Add( )( [ x, fx ] )
            # out = tf.keras.layers.ReLU( )( out )
        else:
            fx = tf.keras.layers.Conv2D( filters, kernelsize, padding = 'same' )( x )
            # fx = layers.BatchNormalization()(fx)
            fx = tf.keras.layers.ReLU( )( fx )

            fx = tf.keras.layers.Conv2D( filters, kernelsize, padding = 'same' )( fx )
            # fx = layers.BatchNormalization()(fx)
            #
        out = tf.keras.layers.Add( )( [ x, fx ] )
        out = tf.keras.layers.ReLU( )( out )

        return out
def computePSR(a,b):
    return np.mean((a**2))/np.mean((b**2))

def generatePerturbData(psr,data,current_label,pretrained_model,t_label,method:str = 'fgsm',n_iter = None):
    '''One sample at a time'''

    if method == 'fgsm':
        if psr == 0:
            p = np.zeros_like(data)
        else:
            perturbation = generateAdvExsFGSM( data, current_label, pretrained_model, t_label = t_label )
            p = TOOLS.l2_limiter(psr = psr,perturbation = perturbation,data = data)
        delta, data = np.squeeze( p ), np.squeeze( data )
    elif method == 'pgd':
        if psr == 0:
            p = np.zeros_like(data)
        else:
            perturbation = generateAdvExsPGD( data, current_label, pretrained_model,t_label=t_label, psr = psr, \
                                                                                                   n_iter = n_iter )
            p = TOOLS.l2_limiter( psr = psr, perturbation = perturbation, data = data )
        delta, data = np.squeeze( p ), np.squeeze( data )


    adv_data = data + delta
    return np.expand_dims( adv_data, axis = 0 ), np.expand_dims( delta, axis = 0 )
def generateAdvExsFGSM(input_CSI, label, pretrained_model, t_label: int = None):
    '''
    Create adversarial pattern for single input
    The input shape should be:
    (1,shape of data) -> (1,200,60,3)
    (1, shape of label) -> (1,276)
    '''
    loss_object = tf.keras.losses.CategoricalCrossentropy( )
    input_CSI = tf.convert_to_tensor(input_CSI, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    if t_label:
        # if label == t_label:
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
    return np.asarray(signed_grad)
def generateAdvExsPGD(input_CSI, labels, pretrained_model,psr,t_label = None, n_iter:int = 0):
    # print( f'The number of iterations are {n_iter}' )
    loss_object = tf.keras.losses.CategoricalCrossentropy( )
    input_CSI = tf.convert_to_tensor(input_CSI, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32 )
    gradient = np.zeros(input_CSI.shape)
    for i in range(n_iter):
        model_input = input_CSI + gradient
        if t_label:
            # if label == t_label:
            tar_label = tf.convert_to_tensor(
                    np.expand_dims( to_categorical( t_label - 1, num_classes = 6 ), axis = 0 ),
                    dtype = tf.float32
                    )
            with tf.GradientTape(persistent=True ) as tape:
                tape.watch( model_input )
                prediction = pretrained_model( model_input )
                loss = - loss_object( tar_label, prediction )
        else:
            with tf.GradientTape( persistent=True ) as tape:
                tape.watch( model_input )
                prediction = pretrained_model( model_input )
                loss = loss_object( labels, prediction )
        # gradient = gradient + (alpha/(i+1))*tf.sign(tape.gradient( loss, model_input ))
        g = tape.gradient( loss, model_input )
        g_norm = g/np.linalg.norm(g)
        gradient = gradient + (np.sqrt( (psr/n_iter) /  (np.mean( g_norm**2 ) / np.mean( input_CSI**2 ))  )*g_norm)
    return np.asarray(gradient)
def runTrain(config, dataset_name):
    m_callback = myCallback()
    # name = 'widar'

    config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(config, dataset_name)
    test_data = config.test_data
    train_data = config.train_data
    test_label = config.test_label
    train_label = config.train_label

    train_data = np.concatenate((train_data, test_data), axis=0)
    train_label = np.concatenate((train_label, test_label), axis=0)
    if os.path.exists(config.pretrained_model_path):
        Net = tf.keras.models.load_model( config.pretrained_model_path )
        print('=========================================================')
        print(f'loading pretrained model {config.pretrained_model_path} to continue training...')
        print('=========================================================')
        # raise Exception(f'The model {config.pretrained_model_path} is already exists')
    else:
        net = AlexNetTF( config )
        Net = net.buildModel( choice = config.DNN_name)
    print(f'Data range from {test_data.min():.2f} to {test_data.max():.2f} \nwith model {config.pretrained_model_path}')



    lrScheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1,
            patience=15,
    )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor='val_acc', patience=35, restore_best_weights=True )

    optimizer = tf.keras.optimizers.Adamax(
            learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,
            name='Adamax'
    )
    # optimizer = tf.keras.optimizers.SGD(learning_rate=config.lr)
    # optimizer = tf.keras.optimizers.Adam(learning_rate = config.lr)
    Net.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='accpgd_attack' )
    # Net.summary( )
    history = Net.fit(
            train_data, train_label,
            validation_split=0.1,
            batch_size = config.batch_size,
            epochs=config.epoch,
            callbacks=[ earlyStop, lrScheduler ],
            shuffle = True,
            verbose =  1,

    )
    Net.evaluate(test_data, test_label)
    Net.save( config.pretrained_model_path )
def runAdvExsTestPSR(input_CSI,labels,pretrained_model,psr,ifpltcmd:bool =False,t_label:int=None,
        attack_method:str='fgsm',n_iter=None,pdf_name=None):
    '''
    labels_pred: should be one hot coded
    input_CSI: whole dataset
    '''
    advData = [ ]
    perturb = [ ]
    model = Model( inputs = pretrained_model.input, outputs = pretrained_model.layers[ -2 ].output )
    if attack_method != 'gaussian':
        if attack_method == 'deepfool':
            df = loadmat( 'perturbation\\df_signfi_model_lab_276_scale_1.mat' )
            pertEx_all = df[ 'perturbation' ]
            input_CSI = (df[ 'advEx' ] - df[ 'perturbation' ])
        for i, test_data in tqdm(enumerate( input_CSI ),desc = f'attack_method:{attack_method}',position = 0):
            test_data, current_label = np.expand_dims( test_data, axis = 0 ), np.expand_dims(
                    labels[ i ], axis = 0
                    )
            if attack_method == 'fgsm' or attack_method == 'pgd':
                advEx, pertEx = generatePerturbData(
                        psr = psr, data = test_data, current_label = current_label, pretrained_model =
                        pretrained_model, t_label = t_label, method = attack_method,n_iter = n_iter
                        )
            elif attack_method == 'deepfool':
                pertEx, _, _, _, _ = deepfool( test_data, model )
                pertEx = TOOLS.l2_limiter( psr = psr, perturbation = pertEx_all[ i:i + 1, :, :, : ], data = test_data )
                advEx = test_data + pertEx
            perturb.append( pertEx )
            advData.append( advEx )

            # if i == 0:
            #     a = np.random.normal(0, 1, input_CSI.shape)
            # pertEx = scaleDeepfool( psr = psr, test_data = test_data, perturbation = a[ i:i + 1 ] )
            # advEx = test_data + pertEx - pertEx.mean( )
        perturb = np.concatenate( perturb, axis = 0 )
        advData = np.concatenate( advData, axis = 0 )
        _, accuracy = pretrained_model.evaluate( advData, labels, verbose = 1 )

    if attack_method == 'gaussian':
        perturb = np.random.normal( 0, 1, size = (1, 200, 60, 3) )

        advData = input_CSI + TOOLS.l2_limiter(
                psr = psr, perturbation = perturb, data = input_CSI
                )
        _, accuracy = pretrained_model.evaluate( advData, labels, verbose = 1 )
    # print( f'The accuracy of adversarial samples for PSR = {psr:.5f} is {accuracy:.6f}' )
    if 0:
        n_samples = perturb.__len__()
        selected_perturb = np.repeat( perturb[ np.random.choice(n_samples,1) ],n_samples, axis = 0)
        advData_uni = input_CSI + selected_perturb

        _, accuracy_2 = pretrained_model.evaluate(advData_uni, labels, verbose = 0 )
    if ifpltcmd:
        label_pred = np.argmax(pretrained_model.predict( advData ),axis=1)
        label_true = np.argmax(labels,axis=1)
        title = f'PSR: {psr}, Accuracy: {accuracy:.2f}, target: {t_label}'
        plotSig.pltcm( label_test_pred = label_pred, true_label = label_true, title = None )
        if pdf_name is not None:
            out = os.path.join( 'RESULTS_FIGS', pdf_name )
            plt.savefig( out + '.pdf', bbox_inches = 'tight', )
    # print( f'The accuracy of universal adversarial samples for PSR = {psr:.5f} is {accuracy_2:.2f}' )
    # a = [accuracy,accuracy_2]
    return accuracy,perturb,advData
def runAdvExsTestEps(input_CSI,labels,pretrained_model,eps,ifpltcmd:bool =False,t_label:int=None):
    '''
    labels_pred: should be one hot coded
    '''
    # print(f'The eps is {eps}')
    # label_adv_pred_container = []
    # for i,data in enumerate(input_CSI):
    #     data,current_label = np.expand_dims(data,axis=0),np.expand_dims(labels_pred[i],axis=0)
    #     ori_pred_label = np.argmax(pretrained_model.predict(data),axis = 1)
    #     if ori_pred_label == np.argmax( current_label, axis=1 ):
    #         advData = data + eps*create_adversarial_pattern(data,current_label,pretrained_model,t_label = t_label)
    #         label_adv_pred_container.append( np.argmax( pretrained_model.predict( advData ), axis = 1 ) )
    #     else:
    #         label_adv_pred_container.append(ori_pred_label)
    # true_label = np.argmax( labels_pred, axis = 1 )
    # label_adv_pred_container = np.squeeze(np.asarray( label_adv_pred_container ))
    # accuracy = np.sum( label_adv_pred_container == true_label ) / labels_pred.shape[0 ]
    advData = [ ]
    for i, test_data in enumerate( input_CSI ):
        test_data, current_label = np.expand_dims( test_data, axis = 0 ), np.expand_dims(
                labels[ i ], axis = 0
                )
        advData.append(
                test_data + eps * generateAdvExsFGSM( test_data, current_label, pretrained_model, t_label = t_label )
                )
    advData = np.concatenate( advData, axis = 0 )
    loss, accuracy = pretrained_model.evaluate( advData, labels, verbose = 0 )
    if ifpltcmd:
      title = f'eps: {eps}, Accuracy: {accuracy:.2f}, target: {t_label}'
      plotSig.pltcm( label_test_pred = label_adv_pred_container, true_label = true_label, title = None )
    print( f'Accuracy for eps = {eps:.3f} is {accuracy:.2f}' )
    return accuracy
if __name__ == '__main__':
    if 0:
        model_structure_list = ['alex1', 'alex2', 'alex3','cnn', 'vgg8', 'vgg10' ,'defult','vgg16','vgg19','resnet',
                                'resnet6']
        model_depth_list = ['resnet','resnet6','resnet10','resnet12']
        all_model = list(set([*model_structure_list,*model_depth_list]))
        # ==================================== Widar =======================================================================
        # config.data_dir = [config.sensingDataset_Root + 'Widar\\' + '20181109',
        #                    # config.sensingDataset_Root + 'Widar\\' + '20181115',
        #                    # 'E:\\SensingDataset\\Widar\\20181118'
        #                    ]
        # config.D_range = 1
        # config.location = [5]
        # config.receiver = ['r4']
        # config.orientation = [ 2,3,4 ]
        # dataset_name = 'widar'
        #
        # for dnn in ['defult']:
        #     config.DNN_name = dnn
        #     runTrain(config,'widar')

        seed_container = [ 2,3,4,5, 6, 7, 8, 9, 10,42 ]

        config.DNN_name = 'defult'
        model_name = [
                'widar_model_defult_loc5_ori234_Rx1_scale_1_envir_1.h5',
                'widar_model_defult_loc5_ori234_Rx2_scale_1_envir_1.h5',
                'widar_model_defult_loc5_ori234_Rx3_scale_1_envir_1.h5',
                'widar_model_defult_loc5_ori234_Rx4_scale_1_envir_1.h5',
                'widar_model_defult_loc5_ori234_Rx5_scale_1_envir_1.h5',
                'widar_model_defult_loc5_ori234_Rx6_scale_1_envir_1.h5',
                      ]
        for seed in seed_container:
            config.set_seed = seed
            for mn in model_name:
                config.DNN_name = mn.split( '_' )[ 2 ]
                if mn not in os.listdir( config.victim_model_Root ):
                    raise Exception( f'The model {mn} is not defined' )
                model_path = os.path.join( config.victim_model_Root, mn )
                location_all = re.findall( r'\d+', mn.split( 'loc' )[ 1 ] )[ 0 ]
                orientation_all = re.findall( r'\d+', mn.split( 'ori' )[ 1 ] )[ 0 ]
                Rx_all = re.findall( r'\d+', mn.split( 'Rx' )[ 1 ] )[ 0 ]
                config.location = [ int( a ) for a in location_all ]
                config.orientation = [ int( a ) for a in orientation_all ]
                config.receiver = [ f'r{int( a )}' for a in Rx_all ]
                config.data_dir = [ config.sensingDataset_Root + 'Widar\\' + '20181109',
                                    # config.sensingDataset_Root + 'Widar\\' + '20181115'
                                    ]
                model_path = os.path.join( config.victim_model_Root, mn )
                print( f'generating UAP for model {mn}' )
                config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
                        config,
                        'widar'
                        )
                train_data = np.concatenate( (config.train_data, config.test_data), axis = 0 )
                train_label = np.concatenate( (config.train_label, config.test_label), axis = 0 )
                data = copy.deepcopy( train_data )
                test_label = copy.deepcopy( train_label )
                # Attack_model = tf.keras.models.load_model( model_path )
                # Attack_model.evaluate(config.train_data, config.train_label)
                if model_path == config.pretrained_model_path:
                    current_UAP = genereate_UAP( dataset = data, model_path = config.pretrained_model_path ,config = config)
                    per_name = 'UAP_' + mn.split( '.' )[ 0 ] + f'_seed_{config.set_seed}' + '.h5'
                    path = os.path.join( config.pert_Mat_Root, per_name )
                    with h5py.File( path, 'w' ) as hdf:
                        hdf.create_dataset( 'universal_perturbation', data = current_UAP )
    # ====================================SignFi========================================================================
    if 0:
        config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
                config, 'signfi'
                )
        model_dir = os.listdir(config.victim_model_Root)
        for p in model_dir:
            if 'widar' in p:
                continue
            print( p )
            model = tf.keras.models.load_model( os.path.join( config.victim_model_Root, p ) )
            model.evaluate( config.test_data, config.test_label )
    '''Run training process'''
    if 0:
        tree = {
                'home_276': [ 'vgg8', 'vgg10' ],
                'lab_276' : [ 'cnn', 'vgg8', 'vgg10' ]
                }
        for envir in tree:
            config.source = envir
            dnns = tree[envir]
            for dnn in tqdm(dnns):
                print(f'Training the model: {dnn} with source: {config.source}.....')
                config.DNN_name = dnn
                runTrain(config,'signfi')
    '''UAP generation'''
    if 0:

        '''Prepare data'''
        "for model 1. cnn, 2.cnn-lstm"

        # type_data = 'wiar'
        # config.data_dir = 'E:\\SensingDataset\\WiAR'
        # config.data_dir = [ config.sensingDataset_Root + 'Widar\\' + '20181109',
        #                     config.sensingDataset_Root + 'Widar\\' + '20181115' ]
        # model_name = os.listdir(path=config.victim_model_Root )
        # model_name = ['signfi_model_defult_lab_276_scale_1.h5',]
        tree = {
                'home_276': [ 'defult' ],
                'lab_276' : [ 'defult' ]
                }

        model_name = []
        for source in tree:
            for model in tree[source]:
                st = f'signfi_model_{model}_{source}_scale_1.h5'
                model_name.append(st)
        # model_name = ['signfi_model_defult_lab_276_scale_1.h5',
        #               'signfi_model_defult_home_276_scale_1.h5',
        #               'signfi_model_vgg16_home_276_scale_1.h5',
        #               'signfi_model_vgg19_home_276_scale_1.h5',
        #               'signfi_model_resnet_home_276_scale_1.h5',
        #               'signfi_model_resnet6_home_276_scale_1.h5']
        seed_container = [42]

        type_data = 'signfi'
        method = 'pgd'
        for seed in seed_container:
            config.set_seed = seed
            for mn in model_name:
                config.DNN_name = mn.split( '_' )[ 2 ]
                if type_data not in mn:
                    continue
                # if config.DNN_name not in all_model:
                #     continue
                if mn not in os.listdir(config.victim_model_Root):
                    raise Exception(f'The model { mn } is not defined')
                if type_data == 'widar':
                    model_path = os.path.join(config.victim_model_Root,mn )
                    location_all = re.findall( r'\d+',  mn.split('loc')[1])[0]
                    config.location = [int(a) for a in location_all]
                    config.orientation = [2]
                    config.data_dir = [ config.sensingDataset_Root + 'Widar\\' + '20181109',
                                        config.sensingDataset_Root + 'Widar\\' + '20181115' ]
                elif type_data == 'signfi':
                    model_path = os.path.join(config.victim_model_Root,mn )
                    if 'home_276' in mn:
                        config.source = 'home_276'
                    elif 'lab_276' in mn:
                        config.source = 'lab_276'
                elif type_data == 'wiar':
                    model_path = os.path.join( config.victim_model_Root, mn )
                print(f'generating UAP for model {mn}')
                config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
                        config,
                        type_data
                        )
                test_data = copy.deepcopy( config.test_data )
                test_label = copy.deepcopy( config.test_label )

                if model_path == config.pretrained_model_path:
                    # current_UAP = genereate_UAP(dataset = test_data, model_path = config.pretrained_model_path)
                    current_UAP = genereate_UAP(dataset = test_data, model_path = config.pretrained_model_path,
                            config =config, method = method,psr=0.003,
                            labels = test_label)
                    per_name = 'UAP_' + mn.split('.')[0]+ f'_seed_{config.set_seed}' + f'_method_{method}_'+'.h5'
                    path = os.path.join( config.pert_Mat_Root, per_name )
                    with h5py.File( path, 'w' ) as hdf:
                        hdf.create_dataset( 'universal_perturbation', data = current_UAP )
        with h5py.File( path, 'r' ) as f:
            UAP_data = np.asarray( list( f[ 'universal_perturbation' ] ) )
        victim_model = tf.keras.models.load_model( os.path.join(config.victim_model_Root,model_name[0]) )
        victim_model.evaluate(config.test_data+UAP_data,config.test_label)
        print(f'PSR = {computePSR(UAP_data,config.test_data[0])}')
        # ==================================================================================================================
    acc_fgsm_all = []
    acc_uni_all = []
    config.D_range = 50
    config.attacker_model_Root = 'SavedModel\\PSR'
    model_name = 'signfi_model_lab_276_scale_1.h5'
    model_path = os.path.join(config.attacker_model_Root, model_name)
    Attack_model = tf.keras.models.load_model( model_path )
    config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
            config,
            'signfi'
            )
    test_data = procOBJ.scale( config.test_data, config.D_range )
    for psr in np.arange(0,0.001,0.0001):
        acc_fgsm, _, _ = DeepNet.runAdvExsTestPSR(
                input_CSI = test_data,
                labels = config.test_label,
                pretrained_model = Attack_model,
                attack_method = 'fgsm',
                psr = psr,
                t_label = None
                )
        acc_fgsm_all.append(acc_fgsm[0])
        acc_uni_all.append(acc_fgsm[1])
    plt.plot( np.arange(0,0.001,0.0001), acc_fgsm_all, marker = 'o', label = 'Adversarial samples specific to input' )
    plt.plot( np.arange(0,0.001,0.0001), acc_uni_all, marker = 'x',label = 'Adversarial samples not specific to input' )
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlabel('PSR')
    plt.grid()



