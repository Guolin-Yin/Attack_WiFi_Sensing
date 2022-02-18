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
sys.path.append( current_dir + '\\utils' )
sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
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
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_acc') > 0.88):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(0.88*100))
			self.model.stop_training = True
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
def generatePerturbData(psr,data,current_label,pretrained_model,t_label):
    global n_p
    n_p = 0
    '''One sample at a time'''
    perturbation = create_adversarial_pattern( data, current_label, pretrained_model, t_label = t_label )
    perturbation = np.squeeze(perturbation)
    data = np.squeeze(data)
    p_perturbation = np.var( perturbation  , )
    p_data = np.var( data  )

    # p_perturbation = np.mean( (perturbation - np.mean(perturbation))**2  )
    # p_data = np.mean( (data - np.mean(data,)) **2   )

    # print(f'The perturbation maximum is {perturbation.max():.2f} and the minimum is {perturbation.min():.2f}')
    # if np.mean(p_perturbation/p_data) != 0:
        # eps = np.sqrt( psr * (1 / np.mean( p_perturbation / p_data )) )
    eps = np.sqrt( psr / np.mean(p_perturbation/p_data) )
    adv_data = data + eps * perturbation
    return np.expand_dims(adv_data,axis=0)
        # print(f'eps is {psr * np.sqrt((1/np.mean(p_perturbation/p_data)))}')
    # else:
        # print('\nNo perturbation added, this may happen in targeted attack, input label is same as the targeted '
        #       'label\n')
        # n_p += 1
        # flag = np.expand_dims(data,axis=0)
        # return flag
def test():
    per = perturbation[:,0,0]
    ori_signal = data[:,0,0]
    adv_data = data + 0.5*perturbation
    plt.plot(per)
    plt.ylim(-5,5)
    plt.plot( ori_signal,label = 'Attack free' )
    plt.plot( adv_data[:,0,0], label = 'Attacked')
    plt.ylabel('Amplitude')
    # print(f'The input PSR is {psr}')
def load_tf_model(path:str = "/content/drive/MyDrive/Colab Notebooks/AdversarialAttack/signFi_model"):
  return tf.keras.models.load_model(path)
def create_adversarial_pattern(input_CSI, label, pretrained_model, t_label: int = None):
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
    return signed_grad
def runTrain(config, dataset_name):
    m_callback = myCallback()
    # name = 'widar'
    X_train, X_test, y_train, y_test = gestureDataLoader.getData(config, dataset_name,ifzscore = False,ifscale=True)
    print(f'The training data range from {X_train.min()} to {X_train.max()}')
    net = AlexNetTF( config )
    Net = net.buildModel( )
    lrScheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1,
            patience=20,
    )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor='val_acc', patience=15, restore_best_weights=True )

    optimizer = tf.keras.optimizers.Adamax(
            learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,
            name='Adamax'
    )
    Net.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
    # Net.summary( )
    history = Net.fit(
            X_train, y_train,
            validation_split=0.05,
            batch_size = config.batch_size,
            epochs=1000,
            callbacks=[ m_callback, lrScheduler ],
            verbose = 1
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
def runAdvExsTestPSR(input_CSI,labels,pretrained_model,psr,ifpltcmd:bool =False,t_label:int=None):
    '''
    labels: should be one hot coded
    '''
    method = 'eval'
    if method == 'pred':
        print( f'Testing the accuracy of adversarial sampels for PSR = {psr}, using predition method' )
        label_adv_pred_container = []
        for i,data in enumerate(input_CSI):
            data,current_label = np.expand_dims(data,axis=0),np.expand_dims(labels[i],axis=0)
            ori_pred_label = np.argmax(pretrained_model.predict(data),axis = 1)
            if ori_pred_label == np.argmax( current_label, axis=1 ):
                advData = generatePerturbData(
                        psr = psr, data = data, current_label = current_label, pretrained_model =
                        pretrained_model, t_label = t_label
                        )
                if advData is str:
                    print('see a string')
                    continue
                # advData = data + eps*create_adversarial_pattern(data,current_label,pretrained_model,t_label = t_label)
                label_adv_pred_container.append( np.argmax( pretrained_model.predict( advData ), axis = 1 ) )
            else:
                label_adv_pred_container.append(ori_pred_label)
        true_label = np.argmax( labels, axis = 1 )
        label_adv_pred_container = np.squeeze(np.asarray( label_adv_pred_container ))
        accuracy = np.sum( label_adv_pred_container == true_label ) / labels.shape[0 ]
    elif method == 'eval':
        # print( f'Testing the accuracy of adversarial sampels for PSR = {psr}, using evaluation method' )
        advData = [ ]
        for i, test_data in enumerate( input_CSI ):
            test_data, current_label = np.expand_dims( test_data, axis = 0 ), np.expand_dims(
                    labels[ i ], axis = 0
                    )
            advData.append(
                    generatePerturbData(
                            psr = psr, data = test_data, current_label = current_label, pretrained_model =
                            pretrained_model, t_label = t_label
                            )
                    )
        advData = np.concatenate( advData, axis = 0 )
        loss, accuracy = pretrained_model.evaluate( advData, labels, verbose = 0 )
    if ifpltcmd:
        label_pred = np.argmax(pretrained_model.predict( advData ),axis=1)
        label_true = np.argmax(labels,axis=1)
        title = f'PSR: {psr}, Accuracy: {accuracy:.2f}, target: {t_label}'
        plotSig.pltcm( label_test_pred = label_pred, true_label = label_true, title = title )
    print(f'The accuracy of adversarial samples for PSR = {psr:.5f} is {accuracy:.2f}')
    return accuracy
def runAdvExsTestEps(input_CSI,labels,pretrained_model,eps,ifpltcmd:bool =False,t_label:int=None):
    '''
    labels: should be one hot coded
    '''
    # print(f'The eps is {eps}')
    # label_adv_pred_container = []
    # for i,data in enumerate(input_CSI):
    #     data,current_label = np.expand_dims(data,axis=0),np.expand_dims(labels[i],axis=0)
    #     ori_pred_label = np.argmax(pretrained_model.predict(data),axis = 1)
    #     if ori_pred_label == np.argmax( current_label, axis=1 ):
    #         advData = data + eps*create_adversarial_pattern(data,current_label,pretrained_model,t_label = t_label)
    #         label_adv_pred_container.append( np.argmax( pretrained_model.predict( advData ), axis = 1 ) )
    #     else:
    #         label_adv_pred_container.append(ori_pred_label)
    # true_label = np.argmax( labels, axis = 1 )
    # label_adv_pred_container = np.squeeze(np.asarray( label_adv_pred_container ))
    # accuracy = np.sum( label_adv_pred_container == true_label ) / labels.shape[0 ]
    advData = [ ]
    for i, test_data in enumerate( input_CSI ):
        test_data, current_label = np.expand_dims( test_data, axis = 0 ), np.expand_dims(
                labels[ i ], axis = 0
                )
        advData.append(
                test_data + eps*create_adversarial_pattern(test_data,current_label,pretrained_model,t_label = t_label)
                )
    advData = np.concatenate( advData, axis = 0 )
    loss, accuracy = pretrained_model.evaluate( advData, labels, verbose = 0 )
    if ifpltcmd:
      title = f'eps: {eps}, Accuracy: {accuracy:.2f}, target: {t_label}'
      plotSig.pltcm( label_test_pred = label_adv_pred_container, true_label = true_label, title = title )
    print( f'Accuracy for eps = {eps:.3f} is {accuracy:.2f}' )
    return accuracy
if __name__ == '__main__':
    current_dir = os.getcwd( )
    sys.path.append( current_dir )
    sys.path.append( current_dir + '/utils' )
    sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
    import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig, TOOLS
    config = Config.getconfig( )
    '''Prepare data'''
    # dataset_name = 'widar'
    #
    # _, config.test_data, _, config.test_label = gestureDataLoader.getData(
    #         config, dataset_name, ifscale = True
    #         )
    # choose dataset
    # dataset_name = 'widar'
    # config.D_range = 500
    # _, config.test_data, _, config.test_label = gestureDataLoader.getData(config, dataset_name, ifzscore = False)
    # config.pretrained_model_path = 'SavedModel/widar_model_loc[2]_ori[2]Rx123456'
    '''Run training process'''
    # for D_range in [5,10,20,30,40,50,500]:
    #     config.D_range = D_range
    #     runTrain( config = config, dataset_name = 'widar' )
    '''Load pretrained model'''
    # pretrained_model = tf.keras.models.load_model(config.pretrained_model_path )
    '''Evaluation of adversarial samples'''
    procOBJ = gestureDataLoader.preprocessing()
    EPS_ACC,PSR_ACC = { },{ }
    dataset_name = 'signfi'
    _, config.test_data, _, config.test_label = gestureDataLoader.getData(
            config, dataset_name, ifscale = True
            )
    for d_range in [ 5,10,20,30,40, 50,500 ]:
        EPS_ACC[ f'range{d_range}' ] = [ ]
        PSR_ACC[ f'range{d_range}' ] = [ ]
        config.D_range = d_range
        config.pretrained_model_path = f'SavedModel/signfi_model_lab_276_scale_{d_range}'+'.h5'
        pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
        test_data = procOBJ.scale(config.test_data,d_range)
        test_label = config.test_label
        print( f'The training data range from {test_data.min( )} to {test_data.max( )}' )
        # for eps in np.arange( 0, 0.07, 0.011 ):
        #     # pretrained_model.summary( )
        #     acc = DeepNet.runAdvExsTestEps(
        #             input_CSI = test_data,
        #             labels = test_label,
        #             pretrained_model = pretrained_model,
        #             eps = eps,
        #             t_label = None
        #             )
        #     EPS_ACC[ f'range{d_range}' ].append( acc )
        for psr_val in np.arange( 0,0.009, 0.001 ):
            acc = runAdvExsTestPSR(
                    input_CSI = test_data,
                    labels = test_label,
                    pretrained_model = pretrained_model,
                    psr = psr_val,
                    t_label = None
                    )
            PSR_ACC[ f'range{d_range}' ].append( acc )
    # EPS_ACC['eps'] = np.arange( 0, 0.07, 0.011 )
    # PSR_ACC['PSR'] = np.arange( 0, 0.7, 0.1 )
    # savemat('utils\\resultsMat\\PSR_ACC_signfi.mat',PSR_ACC)
    # savemat( 'utils\\resultsMat\\EPS_ACC_signfi.mat', EPS_ACC )



    # dataset_name = 'widar'
    # _, config.test_data, _, config.test_label = gestureDataLoader.getData(
    #         config, dataset_name, ifscale = True
    #         )
    # PSR_ACC = {}
    # procOBJ = gestureDataLoader.preprocessing( )
    # for flag in ['original','zscore']:
    #     if flag == 'original':
    #         config.pretrained_model_path = [ 'SavedModel/widar_model_loc[2]_ori[2]Rx123456',
    #                                          'SavedModel/widar_model_loc[2]_ori[2]Rx123456_zscore' ]
    #         pretrained_model = tf.keras.models.load_model( config.pretrained_model_path[ 0 ] )
    #         test_data = config.test_data
    #         test_label = config.test_label
    #         PSR_ACC[ 'original' ] = [ ]
    #     elif flag == 'zscore':
    #         config.pretrained_model_path = ['SavedModel/widar_model_loc[2]_ori[2]Rx123456','SavedModel/widar_model_loc[2]_ori[2]Rx123456_zscore']
    #         pretrained_model = tf.keras.models.load_model( config.pretrained_model_path[1] )
    #         test_data = procOBJ.norm(config.test_data)
    #         test_label = config.test_label
    #         PSR_ACC[ 'zscore' ] = [ ]
    #
    #     print( f'The data mean {test_data.mean( )} variance {test_data.var( )}' )
    #     for psr_val in np.arange( 0, 0.02, 0.002 ):
    #         # pretrained_model.summary( )
    #         acc = runAdvExsTestPSR(
    #                 input_CSI = test_data, labels = test_label, pretrained_model = pretrained_model,
    #                 psr = psr_val, t_label = None
    #                 )
    #         PSR_ACC[ f'{flag}' ].append( acc )
    # savemat('utils/resultsMat/ori_zscore_var.mat',PSR_ACC)





    # for d_range in [ 5,10,20,30,40,50,500 ]:
    #     PSR_ACC[ f'range{d_range}' ] = [ ]
    #     config.test_data = procOBJ.scale(config.test_data,D_range = d_range)
    #     print( f'The training data range from {config.test_data.min( )} to {config.test_data.max( )}' )
    #     config.pretrained_model_path = 'widar_model_loc[2]_ori[2]_scale_' + f'{d_range}' + '.h5'
    #     pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
    #     for psr_val in np.arange( 0, 0.13, 0.02 ):
    #
    #         # pretrained_model.summary( )
    #         acc = DeepNet.runAdvExsTestPSR(
    #                 input_CSI = config.test_data, labels = config.test_label, pretrained_model = pretrained_model,
    #                 psr = psr_val, t_label = None
    #                 )
    #         PSR_ACC[ f'range{d_range}' ].append( acc )
    # PSR_ACC[f'eps'] = np.arange( 0, 0.13, 0.02 )
    # savemat('utils/resultsMat/PSR_ACC_widar1.mat',PSR_ACC)
    # PSR_ACC['eps'] = np.arange( 0, 0.13, 0.02 )


    # testAdvExsPSR(config.test_data,config.test_label,pretrained_model,psr =0.2)
    # for i in np.arange(0,0.01,0.0001):
    # runAdvExsTestEps( config.test_data, config.test_label, pretrained_model, eps = 0.3,t_label = 3,ifpltcmd=True)
    # runAdvExsTestPSR( config.test_data, config.test_label, pretrained_model, psr = 0.03,t_label = None,ifpltcmd=True)
    # accuracy_attack_free = pretrained_model.evaluate(config.test_data, config.test_label, verbose=1)
    # all_acc = []
    # all_t_acc = {}
    # for t in np.arange(1,7,1):
    #     all_t_acc[f'Targeted{t}'] = []
    #     for ep in np.arange( 0, 0.2, 0.01 ):
    #         accuracy = DeepNet.runAdvExsTest(
    #                             input_CSI = config.test_data,
    #                             labels = config.test_label,
    #                             pretrained_model = pretrained_model,
    #                             eps = ep,
    #                             t_label = t
    #                             )
    #         all_t_acc[ f'Targeted{t}' ].append(accuracy)
    # from scipy.io import savemat
    # all_t_acc[ 'eps' ] = []
    # all_t_acc[ 'eps' ].append(np.arange( 0, 0.2, 0.01 ))
    # savemat( "utils\\resultsMat\\targeted-acc.mat", all_t_acc )
    # for ep in np.arange( 0, 0.2, 0.01 ):
    #     accuracy = DeepNet.runAdvExsTest(
    #             input_CSI = config.test_data,
    #             labels = config.test_label,
    #             pretrained_model = pretrained_model,
    #             eps = ep
    #             )
    #     all_acc.append( accuracy )
    # for t in [None,1,2,3,4,5,6]:
    #     DeepNet.runAdvExsTest(
    #                 input_CSI = config.test_data,
    #                 labels = config.test_label,
    #                 pretrained_model = pretrained_model,
    #                 eps = 0.15,
    #                 ifpltcmd = True,
    #                 t_label = t
    #                 )
    '''visualisation of adversarial samples'''
    # X_test = config.test_data[ 0:1 ]
    # y_test = config.test_label[ 0:1 ]
    # eps = np.arange( 0.01, 0.05, 0.01 )
    # range_adv = []
    # for ep in eps:
    #     advData = X_test + ep * create_adversarial_pattern( X_test, y_test, pretrained_model )
    #     range_adv.append(advData[0,:,0,0])
    # plotSig.showSignal(X_test[0,:,0,0],range_adv,eps = list(eps))
    # Spectrogram
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.pcolormesh(config.test_data[0,:,:,1,0])
    # advData = config.test_data[0:1] + 0.025 * create_adversarial_pattern( config.test_data[0:1], config.test_label[
    # 0:1],pretrained_model )
    # plt.figure( )
    # plt.pcolormesh(advData[0,:,:,1,0])
    '''SPR'''
    # for ep in np.arange( 0, 0.2, 0.01 ):
    #     PSR = [ ]
    #     for i in range(len(config.test_data)):
    #         data = config.test_data[i:i+1]
    #         label = config.test_label[i:i+1]
    #         perturbation = ep * create_adversarial_pattern( data, label, pretrained_model )
    #         # advData = data + perturbation
    #         PSR.append(TOOLS.PSRCompute(perturbation, data,  ))
    #     print(f'SPR is {np.mean(PSR)}')

