import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.keras import backend as K
current_dir = os.getcwd( )
sys.path.append( current_dir )
sys.path.append( current_dir + '/utils' )
sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig
config = Config.getconfig( )
'''Prepare data'''
# choose dataset
dataset_name = 'widar'
_, config.test_data, _, config.test_label = gestureDataLoader.getData( config, dataset_name )
'''Load pretrained model'''
config.pretrained_model_path = 'SavedModel/widar_model_loc[2]_ori[2]Rx123456'
pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
pretrained_model.summary( )
'''Performance evaluation under FGSM adversarial attacks (Non-targeted)'''
# all_acc = []
# for ep in np.arange( 1.9, 2.2, 0.1 ):
#     ifpltcmd = False
#     if int(ep) == 2:
#         ifpltcmd = True
#     else:
#         ifpltcmd = False
#     accuracy = DeepNet.runAdvExsTest(
#           input_CSI = config.test_data,
#           labels = config.test_label,
#           pretrained_model = pretrained_model,
#           eps = ep,
#           ifpltcmd = ifpltcmd
#           )
#
#     all_acc.append( accuracy )
'''visualisation of adversarial samples'''
'''1. Visualise the waveform'''
# X_test = config.test_data[ 0:1 ]
# y_test = config.test_label[ 0:1 ]
# eps = np.arange( 0.5, 2.2, 0.5 )
# range_adv = []
# for ep in eps:
#     advData = X_test + ep * DeepNet.create_adversarial_pattern( X_test, y_test, pretrained_model )
#     range_adv.append(advData[0,:,0,0])
# plotSig.showSignal(X_test[0,:,0,0],range_adv,eps = list(eps))

'''Targeted FGSM attack'''
acc_container = {}
for t in np.arange( 1, 7, 1 ):
  acc_container[f'target{t}'] = []
  for ep in np.arange( 0.0, 3, 0.1 ):
    accuracy = DeepNet.runAdvExsTest(
            input_CSI = config.test_data,
            labels = config.test_label,
            pretrained_model = pretrained_model,
            eps = ep,
            ifpltcmd = False,
            t_label = int(t)
            )
    acc_container[ f'target{t}' ].append(accuracy)
print(acc_container)