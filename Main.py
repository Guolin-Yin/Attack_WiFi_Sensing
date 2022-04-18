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
#           labels_pred = config.test_label,
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
    accuracy = DeepNet.runAdvExsTestPSR(
            input_CSI = config.test_data, labels = config.test_label, pretrained_model = pretrained_model, psr =3e-4,
            ifpltcmd = False, t_label = int( t )
            )
    acc_container[ f'target{t}' ].append(accuracy)
print(acc_container)
'''Test PSR and EPS for different range of input'''
# config.D_range = 1
# scale_container = [1,5,10,20,30,40,50,60,70,80]
# for i in [50,60,70,80]:
#     config.D_range = i
#     runTrain( config = config, dataset_name = 'signfi' )
# config.pretrained_model_path = 'SavedModel/signFi_model_lab_276_zscore'
def h5data():
	uni_per_widar_in_domain = loadmat(os.path.join(config.pert_Mat_Root,
		'uni_per_widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.mat' ))['universal_perturbation']
	path = os.path.join(config.pert_Mat_Root, 'UAP_data.h5')
	with h5py.File(path,'w') as hdf:
		hdf.create_dataset('universal_perturbation',data = uni_per_widar_in_domain)
	with h5py.File( path, "r" ) as f:
		# List all groups
		print( "Keys: %s" % f.keys( ) )
		a_group_key = list( f.keys( ) )[ 0 ]
		# Get the data
		data = list( f[ a_group_key ] )
PSR_ACC = {}
for d_range in [5,10,20,30,40,50]:
    PSR_ACC[f'range{d_range}'] = []
    for psr_val in np.arange(0,0.13,0.02):
        dataset_name = 'signfi'
        config.D_range = d_range
        _, config.test_data, _, config.test_label = gestureDataLoader.getData( config, dataset_name, ifscale = True )
        print( f'The training data range from {config.test_data.min( )} to {config.test_data.max( )}' )
        pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
        # pretrained_model.summary( )
        acc = DeepNet.runAdvExsTestPSR(
                input_CSI = config.test_data, labels = config.test_label, pretrained_model = pretrained_model,
                psr = psr_val, t_label = None
                )
        PSR_ACC[ f'range{d_range}' ].append( acc )
EPS_ACC = { }
for d_range in [ 40, 50 ]:
    EPS_ACC[ f'range{d_range}' ] = [ ]
    for eps in np.arange( 0, 0.07, 0.011 ):
        dataset_name = 'signfi'
        config.D_range = d_range
        _, config.test_data, _, config.test_label = gestureDataLoader.getData( config, dataset_name, ifscale = True )
        print( f'The training data range from {config.test_data.min( )} to {config.test_data.max( )}' )
        pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
        # pretrained_model.summary( )
        acc = DeepNet.runAdvExsTestEps(
                input_CSI = config.test_data,
                labels = config.test_label,
                pretrained_model = pretrained_model,
                eps = eps,
                t_label = None
                )
        EPS_ACC[ f'range{d_range}' ].append( acc )
'''Deep fool and universal perturbation'''
uni_per = loadmat( 'universal_perturbation_0.02.mat' )[ 'universal_perturbation' ]

deepfool_per = loadmat('utils\\resultsMat\\signfi_lab_276_deepfool.mat')[ 'perturbation' ]

acc, perturb, Advex = runAdvExsTestPSR(
        input_CSI = test_data,
        labels = test_label,
        pretrained_model = pretrained_model,
        psr = 5.6e-3,
        t_label = None,
        attack_method = 'fgsm'
        )
t_data = np.mean( test_data[ 0 ], axis = 1 ).mean( axis = 1 )
uni_t_data = (test_data + np.repeat( uni_per, test_data.__len__( ), axis = 0 ))[ 0 ].mean( axis = 1 ).mean( axis = 1 )
df_t_data = (test_data + deepfool_per)[ 0 ].mean( axis = 1 ).mean( axis = 1 )
fg_t_data = Advex[ 0 ].mean( axis = 1 ).mean( axis = 1 )
plt.plot( label='original' )
plt.plot( label='Universal perturbation (PSR = 5.6e-3)' )
plt.plot( label='Deepfool (PSR = 2.3e-4)' )
plt.plot( label='One-step FGSM (PSR = 5.6e-3)' )
plt.ylabel( 'Amplitude' )
plt.legend( )

def pgd(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()