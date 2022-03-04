'''Tensorflow'''
from tensorflow.keras.layers import Dense, Input, Softmax,ZeroPadding2D,MaxPooling2D,Conv2D,Flatten,Lambda,Dropout,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
'''Pytorch'''
# import torch
# import torch.nn as nn
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import random_split
# import torch.nn.functional as F
# from torchsummary import summary
import numpy as np
import sys
import os
import copy
current_dir = os.getcwd( )
sys.path.append( current_dir )
sys.path.append( current_dir + '\\utils' )
sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
sys.path.append( "/Users/guolinyin/Google 云端硬盘/Colab Notebooks/AdvAttackandDefense/utils" )

import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig, TOOLS


from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from DeepFool import deepfool
from Universal_pert import universal_perturbation
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
def generatePerturbData(psr,data,current_label,pretrained_model,t_label,method:str = 'fgsm'):
    '''One sample at a time'''

    if method == 'fgsm':
        perturbation = generateAdvExsFGSM( data, current_label, pretrained_model, t_label = t_label )
    elif method == 'pgd':
        perturbation = generateAdvExsPGD( data, current_label, pretrained_model,alpha = psr, n_iter=20)
    perturbation, data = np.squeeze( perturbation ), np.squeeze( data )
    eps = np.sqrt( psr / np.mean( np.var( perturbation ) / np.var( data ) ) )
    delta = perturbation.clip(-eps,eps)
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
    return signed_grad
def generateAdvExsPGD(input_CSI, labels, pretrained_model,alpha = 1e4,n_iter:int = 20):
    loss_object = tf.keras.losses.CategoricalCrossentropy( )
    input_CSI = tf.convert_to_tensor(input_CSI, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32 )
    gradient = np.zeros(input_CSI.shape)
    for i in range(n_iter):
        model_input = input_CSI + gradient
        with tf.GradientTape( persistent=True ) as tape:
            tape.watch( model_input )
            prediction = pretrained_model( model_input )
            loss = loss_object( labels, prediction )
        gradient = gradient + (alpha/(i+1))*tf.sign(tape.gradient( loss, model_input ))
    return gradient
def runTrain(config, dataset_name):
    m_callback = myCallback()
    # name = 'widar'
    config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(config, dataset_name)
    # test_data = procOBJ.scale( config.test_data, config.D_range )
    # train_data = procOBJ.scale( config.train_data, config.D_range )
    test_data = procOBJ.scale(config.test_data,config.D_range)
    train_data = procOBJ.scale(config.train_data,config.D_range)
    test_label = config.test_label
    train_label = config.train_label
    print(f'Data range from {test_data.min():.2f} to {test_data.max():.2f} \nwith model {config.pretrained_model_path}')
    net = AlexNetTF( config )
    Net = net.buildModel( )
    lrScheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1,
            patience=30,
    )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor='val_acc', patience=40, restore_best_weights=True )

    optimizer = tf.keras.optimizers.Adamax(
            learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,
            name='Adamax'
    )
    # optimizer = tf.keras.optimizers.SGD(learning_rate=config.lr)
    # optimizer = tf.keras.optimizers.Adam(learning_rate = config.lr)
    Net.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
    Net.summary( )
    history = Net.fit(
            train_data, train_label,
            validation_split=0.05,
            batch_size = config.batch_size,
            epochs=200,
            callbacks=[ earlyStop, lrScheduler ],
            shuffle = True,
            verbose = 1
    )
    Net.evaluate(test_data, test_label)
    Net.save( config.pretrained_model_path )

def runAdvExsTestPSR(input_CSI,labels,pretrained_model,psr,ifpltcmd:bool =False,t_label:int=None,
        attack_method:str='fgsm'):
    '''
    labels: should be one hot coded
    '''
    # method = 'eval'
    # if method == 'pred':
    #     print( f'Testing the accuracy of adversarial sampels for PSR = {psr}, using predition method' )
    #     label_adv_pred_container = []
    #     for i,data in enumerate(input_CSI):
    #         data,current_label = np.expand_dims(data,axis=0),np.expand_dims(labels[i],axis=0)
    #         ori_pred_label = np.argmax(pretrained_model.predict(data),axis = 1)
    #         if ori_pred_label == np.argmax( current_label, axis=1 ):
    #             advData = generatePerturbData(
    #                     psr = psr, data = data, current_label = current_label, pretrained_model =
    #                     pretrained_model, t_label = t_label,method = 'pgd'
    #                     )
    #             if advData is str:
    #                 print('see a string')
    #                 continue
    #             # advData = data + eps*create_adversarial_pattern(data,current_label,pretrained_model,t_label = t_label)
    #             label_adv_pred_container.append( np.argmax( pretrained_model.predict( advData ), axis = 1 ) )
    #         else:
    #             label_adv_pred_container.append(ori_pred_label)
    #     true_label = np.argmax( labels, axis = 1 )
    #     label_adv_pred_container = np.squeeze(np.asarray( label_adv_pred_container ))
    #     accuracy = np.sum( label_adv_pred_container == true_label ) / labels.shape[0 ]
    # elif method == 'eval':
    advData = [ ]
    perturb = [ ]
    model = Model( inputs = pretrained_model.input, outputs = pretrained_model.layers[ -2 ].output )
    for i, test_data in enumerate( input_CSI ):
        # print(f'Generating advData for {i+1} sample')
        test_data, current_label = np.expand_dims( test_data, axis = 0 ), np.expand_dims(
                labels[ i ], axis = 0
                )
        if attack_method == 'fgsm' or attack_method == 'pgd':
            advEx, pertEx = generatePerturbData(
                    psr = psr, data = test_data, current_label = current_label, pretrained_model =
                    pretrained_model, t_label = t_label, method = attack_method
                    )
        elif attack_method == 'deepfool':
            pertEx, _, _, _, advEx = deepfool( test_data, model )
        perturb.append( pertEx )
        advData.append( advEx )
    perturb = np.concatenate( perturb, axis = 0 )
    advData = np.concatenate( advData, axis = 0 )
    # Choose one perturbation
    # n_samples = perturb.__len__()
    # selected_perturb = np.repeat( perturb[ np.random.choice(n_samples,1) ],n_samples, axis = 0)
    # advData_uni = input_CSI + selected_perturb
    _, accuracy = pretrained_model.evaluate( advData, labels, verbose = 0 )
    # _, accuracy_2 = pretrained_model.evaluate(advData_uni, labels, verbose = 0 )
    if ifpltcmd:
        label_pred = np.argmax(pretrained_model.predict( advData ),axis=1)
        label_true = np.argmax(labels,axis=1)
        title = f'PSR: {psr}, Accuracy: {accuracy:.2f}, target: {t_label}'
        plotSig.pltcm( label_test_pred = label_pred, true_label = label_true, title = title )
    # print(f'The accuracy of adversarial samples for PSR = {psr:.5f} is {accuracy:.6f}')
    # print( f'The accuracy of universal adversarial samples for PSR = {psr:.5f} is {accuracy_2:.2f}' )
    return accuracy,perturb,advData
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
                test_data + eps * generateAdvExsFGSM( test_data, current_label, pretrained_model, t_label = t_label )
                )
    advData = np.concatenate( advData, axis = 0 )
    loss, accuracy = pretrained_model.evaluate( advData, labels, verbose = 0 )
    if ifpltcmd:
      title = f'eps: {eps}, Accuracy: {accuracy:.2f}, target: {t_label}'
      plotSig.pltcm( label_test_pred = label_adv_pred_container, true_label = true_label, title = title )
    print( f'Accuracy for eps = {eps:.3f} is {accuracy:.2f}' )
    return accuracy
if __name__ == '__main__':
    config = Config.getconfig( )
    '''Prepare data'''
    config.D_range = 1
    # config.pretrained_model_path = f'SavedModel/signfi_model_lab_276_scale_{config.D_range}' + '.h5'
    procOBJ = gestureDataLoader.preprocessing()
    config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(config,'widar')
    test_data = copy.deepcopy(procOBJ.scale(config.test_data,config.D_range))
    train_data = copy.deepcopy(procOBJ.scale(config.train_data,config.D_range))
    test_label = copy.deepcopy(config.test_label)
    train_label = copy.deepcopy(config.train_label)
    config.attack_model_Root = 'SavedModel\\Attack_target_model'
    '''Run training process'''
    # runTrain( config = config, dataset_name = 'widar' )

    Attack_model = tf.keras.models.load_model( os.path.join(config.attack_model_Root,
            'widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5'))
    f = Model(Attack_model.input,Attack_model.get_layer('FC_2').output)
    '''Get UAP'''
    uni_per_widar = loadmat( os.path.join(config.pert_Mat_Root,
            'uni_per_widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.mat' ))['universal_perturbation']
    uni_per_signfi = loadmat(os.path.join(config.pert_Mat_Root, 'signfi_lab_276_universal_perturbation.mat' ) )[ 'universal_perturbation' ]
    uni_per_widar_in_domain = universal_perturbation(dataset = test_data,f = f,overshoot=0.002)
    out = {'universal_perturbation': uni_per_widar_in_domain }
    savemat(os.path.join(config.pert_Mat_Root,
            'uni_per_widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.mat' ),out )
    '''Cross domain universal perturbation testing'''

    # model = tf.keras.models.load_model( os.path.join(model_path,
    #         'widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.h5'))

    acc_all = {}
    name = ['Widar U/P, envir 2','signfi U/P', 'Widar U/P, same envir']
    for i, perturb in enumerate( [ uni_per_widar, uni_per_signfi, uni_per_widar_in_domain]):
        acc_all[name[i]] = []
        v = perturb
        psr_range = np.arange(0.00,0.055,0.002)
        print(f'Testing the attack performance of the {name[i]} generated universal perturbation')
        for psr in psr_range:
            # Perturbation calibration
            scale_factor = np.sqrt(psr*test_data.var()*((v.max()-v.min())**2)/v.var())
            scaled_uni_per = procOBJ.scale(v,scale_factor)
            adv_data = test_data + np.repeat( scaled_uni_per, test_data.__len__( ), axis = 0 )
            adapted_adv_data = adv_data - np.abs( np.mean( adv_data ) - np.mean( test_data ) )
            # Attack_model.evaluate( test_data, test_label )
            _,acc = Attack_model.evaluate(adapted_adv_data,test_label,verbose = 0)
            print(f'The PSR is {scaled_uni_per.var()/test_data.var():.4f}, accuracy is {acc:.4f}')
            acc_all[ name[ i ] ].append(acc)
    else:
        for k in range(i+1):
            plt.plot(psr_range, acc_all[name[k]],label = name[k],marker = 'o',markersize=5)
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('PSR')
        plt.grid(alpha = 0.4)
        plt.show()
    # def compare_wave():
    #     v = uni_per_signfi
    #     psr = 0.01
    #     scale_factor = np.sqrt( psr * test_data.var( ) * ((v.max( ) - v.min( )) ** 2) / v.var( ) )
    #     scaled_uni_per = procOBJ.scale( v, scale_factor )
    #     adv_data = test_data + np.repeat( scaled_uni_per, test_data.__len__( ), axis = 0 )
    #     adapted_adv_data = adv_data - np.abs( np.mean( adv_data ) - np.mean( test_data ) )
    #     plt.plot( adapted_adv_data[ 0 ].mean(axis =1).mean(axis=1), label = 'attacked'  )
    #     plt.plot( test_data[ 0 ].mean(axis =1).mean(axis=1) ,label = 'attack free'   )
    #     plt.ylabel('Amplitude')
    #     plt.title('PSR = 0.01')
    #     plt.legend()


    # import h5py
    # with h5py.File('utils\\perturbationMatFiles\\widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5','w'
    #         ) as hdf:
    #     hdf.create_dataset('universal_perturbation',data = uni_per_widar_in_domain)
    '''Evaluation of adversarial samples'''
    # print(f'The accuracy for universal_perturbation {acc * 100:.2f}%')
    # _, acc = pretrained_model.evaluate( test_data, test_label, verbose = 0 )
    # print(f'The accuracy for original model {acc * 100:.2f}%')
    # for i in range(test_data.__len__()):
    #     psr = np.var(uni_per_widar[i,:,0,0])/np.var(test_data[i,:,0,0])
    #     if psr > 0.01:
    #         print(i)
    #         plt.plot( adv_data[ i, :, 0, 0 ] )
    #         plt.plot( test_data[ i, :, 0, 0 ] )
    #         break

    # df = {'perturbation':perturb,
    #         'advData': advData}
    # savemat('signfi_lab_276_deepfool.mat',df)
    # for d_range in [ 5,10,20,30,40, 50,500 ]:
    #     EPS_ACC[ f'range{d_range}' ] = [ ]
    #     PSR_ACC[ f'range{d_range}' ] = [ ]
    #
    #     pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
    #     test_data = procOBJ.scale(config.test_data,d_range)
    #     test_label = config.test_label
    #     print( f'The training data range from {test_data.min( )} to {test_data.max( )}' )
    #     # for eps in np.arange( 0, 0.07, 0.011 ):
    #     #     # pretrained_model.summary( )
    #     #     acc = DeepNet.runAdvExsTestEps(
    #     #             input_CSI = test_data,
    #     #             labels = test_label,
    #     #             pretrained_model = pretrained_model,
    #     #             eps = eps,
    #     #             t_label = None
    #     #             )
    #     #     EPS_ACC[ f'range{d_range}' ].append( acc )
    #     for psr_val in np.arange( 0,0.009, 0.001 ):
    #         acc = runAdvExsTestPSR(
    #                 input_CSI = test_data,
    #                 labels = test_label,
    #                 pretrained_model = pretrained_model,
    #                 psr = psr_val,
    #                 t_label = None
    #                 )
    #         PSR_ACC[ f'range{d_range}' ].append( acc )
    # PSR_ACC['PSR'] = np.arange( 0,0.009, 0.001 )
    # savemat('utils/resultsMat/PSR_ACC_signfi.mat',PSR_ACC)
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



