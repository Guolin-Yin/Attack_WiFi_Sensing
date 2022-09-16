import numpy as np
import sys
import os
import copy
import h5py
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
for data_root, data_dirs, data_files in os.walk( os.getcwd( ) ):
    for rt in data_dirs:
        sys.path.append( os.path.join(data_root,rt) )
import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig, TOOLS
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from DeepFool import deepfool
from Experiments import scaleDeepfool
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from Experiments import heatmap
from matplotlib.ticker import StrMethodFormatter
import time
import os
gpus = tf.config.experimental.list_physical_devices( 'GPU' )
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth( gpu, True )
    except RuntimeError as e:
        print( e )
config = Config.getconfig( )
procOBJ = gestureDataLoader.preprocessing( )
'''============================White box attacks============================'''
"""1.1.1 FGSM attack method"""
# Non-targeted attacks ------------- SignFi
def FGSM_signfi_NTA(psr_container = None):
	config.source = 'lab_276'
	config.D_range = 1
	pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
	X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'signfi')
	acc_container = []
	if psr_container is None:
		psr_container = np.arange(0,0.0041,0.0005)
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
															input_CSI,
															labels,
															pretrained_model,
															psr,
															ifpltcmd =False,
															t_label=None,
													        attack_method='fgsm')
		acc_container.append( accuracy )
	plt.plot(psr_container,acc_container,label = 'signFi, lab')
	plt.legend()
	plt.grid()
	plt.xlabel('PSR')
	plt.ylabel('Accuracy Difference')
	return psr_container,acc_container
# Non-targeted attacks ------------- Widar
def FGSM_widar_NTA():
	config.D_range = 1
	config.orientation = [  1, 2, 3, 4, 5, 6 ]
	config.location = [ 2, ]
	config.receiver = [ 'r1', 'r2', 'r3', 'r4', 'r5', 'r6' ]
	config.data_dir = [
	                   'E:\\SensingDataset\\Widar\\20181118\\user2'
	                    ]
	pretrained_model = tf.keras.models.load_model(
			'SavedModel\\victim_model\\widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5')
	X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'widar')
	acc_container = []
	psr_container = np.arange(0,0.031,0.002)
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
															input_CSI,
															labels,
															pretrained_model,
															psr,
															ifpltcmd =False,
															t_label=None,
													        attack_method='fgsm')
		acc_container.append(1 - accuracy)
	plt.plot(psr_container,acc_container,label = 'widar')
	plt.legend()
	plt.grid()
	plt.xlabel('PSR')
	plt.ylabel('Accuracy Difference')
#     Targeted attacks ------------- Widar
def FGSM_widar(n_rx,t_label):
	victim_model_source = [
			'widar_model_defult_loc5_ori234_Rx1_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx2_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx3_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx4_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx5_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx6_scale_1_envir_1.h5'
			]
	config.D_range = 1
	config.receiver = [ f'r{n_rx}' ]
	config.location = [ 5 ]
	config.orientation = [ 2, 3, 4 ]
	config.DNN_name = "defult"
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109' ]
	config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
			config,
			'widar'
			)
	# config.pretrained_model = victim_model_source[n_rx - 1]
	pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
	# if psr_container is None:
	# psr_container = np.linspace(0.0,0.004,7)
	psr_container = [0.00066667]
	acc_container = []
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
												config.train_data,
												config.train_label,
												pretrained_model,
												psr,
												ifpltcmd =True,
												t_label=t_label,
										        attack_method='fgsm',
												n_iter = 4,
												pdf_name = f'cf_FGSM_{t_label}'
				)
		acc_container.append( accuracy )
	return psr_container, acc_container

"""1.1.2 PGD attack method"""
def PGD_signfi_NTA(psr_container =None,n_iter=None):
	config.source = 'lab_276'
	config.D_range = 1
	pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
	X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'signfi')
	acc_container = []

	if psr_container is None:
		psr_container = np.linspace(0.0005,0.004,8)
	for psr in psr_container:
	# for n_iter_i in range(n_iter):
	# 	start = time.time()
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
															input_CSI,
															labels,
															pretrained_model,
															psr,
															ifpltcmd =False,
															t_label=None,
													        attack_method='pgd',
															n_iter = n_iter)
		# end = time.time()
		# tc = (end - start)/input_CSI.shape[0]
		# print(f'the time cost is {tc:.2f} per sample')
		#
		acc_container.append( tc )
		# acc_container.append( accuracy)
	# plt.plot(psr_container,acc_container,label = 'signFi, lab')
	# plt.legend()
	# plt.grid()
	# plt.xlabel('PSR')
	# plt.ylabel('Accuracy Difference')
	return psr_container,acc_container
# Non-targeted attacks ------------- Widar
def PGD_widar_NTA():
	config.D_range = 1
	config.orientation = [ 1, 2, 3, 4, 5, 6 ]
	config.location = [ 2, ]
	config.receiver = [ 'r1', 'r2', 'r3', 'r4', 'r5', 'r6' ]
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109', 'E:\\SensingDataset\\Widar\\20181115' ]
	pretrained_model = tf.keras.models.load_model(
			'SavedModel\\victim_model\\widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.h5'
			)
	X_train, input_CSI, y_train, labels = gestureDataLoader.getData( config = config, dataset_name = 'widar' )
	acc_container = [ ]
	psr_container = np.arange( 0, 0.031, 0.002 )
	for psr in psr_container:
		accuracy, _, _ = DeepNet.runAdvExsTestPSR(
				input_CSI,
				labels,
				pretrained_model,
				psr, #0.0001 => accpgd_attack = 0.543
				ifpltcmd = False,
				t_label = True,
				attack_method = 'pgd'
				)
		acc_container.append(1 - accuracy)
	plt.plot(psr_container,acc_container,label = 'widar')
	plt.legend()
	plt.grid()
	plt.xlabel('PSR')
	plt.ylabel('Accuracy Difference')
def PGD_widar(n_rx,t_label):
	victim_model_source = [
			'widar_model_defult_loc5_ori234_Rx1_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx2_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx3_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx4_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx5_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx6_scale_1_envir_1.h5'
			]
	config.D_range = 1
	config.receiver = [ f'r{n_rx}' ]
	config.location = [ 5 ]
	config.orientation = [ 2, 3, 4 ]
	config.DNN_name = "defult"
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109' ]
	config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
			config,
			'widar'
			)
	# config.pretrained_model = victim_model_source[n_rx - 1]
	pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
	# if psr_container is None:
	# psr_container = np.linspace(0.0,0.004,7)
	psr_container = [np.linspace(0.0,0.004,7)[0]]
	acc_container = []
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
												config.train_data,
												config.train_label,
												pretrained_model,
												psr,
												ifpltcmd =True,
												t_label=t_label,
										        attack_method='pgd',
												n_iter = 4,
				pdf_name = f'cf_Original_{t_label}'
				)
		acc_container.append( accuracy )
	return psr_container, acc_container
def Gaussian_widar(n_rx,t_label):
	victim_model_source = [
			'widar_model_defult_loc5_ori234_Rx1_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx2_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx3_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx4_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx5_scale_1_envir_1.h5',
			'widar_model_defult_loc5_ori234_Rx6_scale_1_envir_1.h5'
			]
	config.D_range = 1
	config.receiver = [ f'r{n_rx}' ]
	config.location = [ 5 ]
	config.orientation = [ 2, 3, 4 ]
	config.DNN_name = "defult"
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109' ]
	config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
			config,
			'widar'
			)
	# config.pretrained_model = victim_model_source[n_rx - 1]
	pretrained_model = tf.keras.models.load_model( config.pretrained_model_path )
	# if psr_container is None:
	psr_container = np.linspace(0.0,0.004,7)
	# psr_container = [0.001]
	acc_container = []
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
												config.train_data,
												config.train_label,
												pretrained_model,
												psr,
												ifpltcmd =False,
												t_label=t_label,
										        attack_method='gaussian',
												n_iter = 4)
		acc_container.append( accuracy )
	return psr_container, acc_container
"""1.1.3 Deepfool attack method"""
# Non-targeted attacks ------------- SignFi
def deepfool_signfi_NTA(psr_container =None):
	config.source = 'lab_276'
	config.D_range = 1
	pretrained_model = tf.keras.models.load_model( 'SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5' )
	X_train, input_CSI, y_train, labels = gestureDataLoader.getData( config = config, dataset_name = 'signfi' )
	acc_container = []
	if psr_container is None:
		psr_container = np.arange(0,0.0041,0.0005)
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(input_CSI,
												labels,
												pretrained_model,
												psr,
												attack_method='deepfool')
		acc_container.append( accuracy )
	return psr_container,acc_container
# Non-targeted attacks ------------- Widar
def deepfool_widar_NTA():
	config.D_range = 1
	config.orientation = [ 1, 2, 3, 4, 5, 6 ]
	config.location = [ 2, ]
	config.receiver = [ 'r1', 'r2', 'r3', 'r4', 'r5', 'r6' ]
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109', 'E:\\SensingDataset\\Widar\\20181115' ]
	pretrained_model = tf.keras.models.load_model(
			'SavedModel\\victim_model\\widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.h5'
			)
	accuracy,_,_ = DeepNet.runAdvExsTestPSR(input_CSI,
											labels,
											pretrained_model,
											attack_method='deepfool')
'''============================Black box attacks============================'''
# Victim model: signfi_model_defult_lab_276_scale_50.h5
#Attack UAP: UAP_signfi_model_defult_lab_276_scale_50.h5
def UAPTest(
		# victim_model_path = 'SavedModel\\PSR\\' + 'signfi_model_lab_276_scale_1.h5',
		# victim_model_path = config.attacker_model_Root  + '\\wiar_model_defult_scale_1numuser_10.h5',
		victim_model_path = config.victim_model_Root + '\\signfi_model_defult_lab_276_scale_1.h5',
		psr_range = None,
		use_Guas = False,
		delay = None,
		# 'SavedModel\\victim_model\\widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5',
		*args,**UAP_file_names):
	victim_name = victim_model_path.split('model_')[1].split('_')[0]
	victim_model = tf.keras.models.load_model( victim_model_path )
	file_name_list = UAP_file_names.keys()
	acc_all = {}
	# psr_range = np.arange( 0.000, 0.007, 0.0005 )
	if psr_range is None:
		psr_range = [0,0.0035,0.004]
	# psr_range = np.arange(0,0.0041,0.0005)
	acc_all['Guassian_noise'] = []
	# for name in tqdm(file_name_list,desc = 'Testing...',position = 0):
	for name in file_name_list:
		DNN_name = UAP_file_names[name].split('model_')[1].split('_')[0]
		data_name = UAP_file_names[ name ].split( '_' )[ 1 ]
		if 'seed' in UAP_file_names[ name ]:
			n_seed = UAP_file_names[ name ].split('_')[-1].split('.')[0]
			acc_name = data_name + '_' + DNN_name + '_' + n_seed
			desc = f'Dataset: {data_name}, Attacker {DNN_name}, victim: {victim_name}, seed {n_seed}'
		else:
			acc_name = data_name + '_' + DNN_name
			desc = f'Dataset: {data_name}, Attacker {DNN_name}, victim: {victim_name}'
		acc_all[acc_name] = []
		UAP_path = os.path.join(config.pert_Mat_Root,UAP_file_names[name])

		with h5py.File(UAP_path,'r') as f:
			a_group_key = 'universal_perturbation'
			# Get the data
			UAP_data = np.asarray(list( f[ a_group_key ] ))
		for psr in tqdm(psr_range,desc = desc,position = 0):
			scaled_uni_per = []
			for data in config.test_data:
				current_uni_per = TOOLS.l2_limiter( psr = psr, perturbation = UAP_data, data = data )
				if delay:
					sig_len = current_uni_per.shape[1]
					delay_uni_per = np.zeros_like(current_uni_per)

					delay_uni_per[ :, 0 : sig_len - delay, :, : ] = current_uni_per[ :, delay : sig_len, :, : ]
					delay_uni_per[ :, sig_len - delay:sig_len, :, : ] = current_uni_per[ :, 0 : delay , :, : ]


					current_uni_per = copy.deepcopy(delay_uni_per)
				scaled_uni_per.append(current_uni_per)
			scaled_uni_per = np.concatenate(scaled_uni_per,axis = 0)
			adv_data = config.test_data + scaled_uni_per
			_,acc = victim_model.evaluate(adv_data,config.test_label,batch_size=32, verbose = 0)
			acc_all[ acc_name ].append(acc)

	if use_Guas:
		noise = np.random.normal( 0, 1, size = (1, 200, 60, 3) )
		for psr in tqdm(psr_range,desc = f'Testing performance of UAP from Guassian_noise',position = 0):
			# acc_awgn = awgn_samples_test( victim_model, config.test_data, config.test_label, psr )
			# scaled_uni_per = scaleDeepfool( psr = psr, test_data = config.test_data, perturbation = noise )
			# noise_data = config.test_data + scaled_uni_per - scaled_uni_per.mean( )
			noise_data = config.test_data + TOOLS.l2_limiter( psr = psr, perturbation = noise, data = config.test_data )
			_, acc_awgn = victim_model.evaluate( noise_data, config.test_label, verbose = 0 )
			acc_all[ 'Guassian_noise' ].append( acc_awgn )
	# results_path = 'utils/resultsMat/cross_model_test.mat'
	# acc_all['psr'] = psr_range
	# savemat(results_path,acc_all)
	# plotting(psr_range,acc_all)

	return psr_range,acc_all

'''2.1.1 cross-domain'''
'''2.1.2 cross-model'''
# Victim model: signfi_model_lab_276_scale_1.h5
#Attack UAP: UAP_signfi_model_defult_home_276_scale_1_seed_8.h5
def cross_domain_test():
	victim_model = [
					'signfi_model_defult_lab_276_scale_1.h5',
	                'signfi_model_defult_home_276_scale_1.h5'
			]
	resultsMat_name = ['UAP_signfi_atk_home_vic_lab_scale_1.mat','UAP_signfi_atk_lab_vic_home_scale_1.mat']
	UAP_PATH = {
			'lab_276':{},
			'home_276':{},
			}

	for UAP_name in victim_model:
		UAP_name = 'UAP_'+UAP_name.split('.')[0]
		i = 0
		for UAP in os.listdir( config.pert_Mat_Root ):
			if UAP_name not in UAP:
				continue
			if 'lab_276' in UAP_name:
				UAP_PATH['lab_276'][f'UAP_{i}'] = UAP
				i += 1
			if 'home_276' in UAP_name:
				UAP_PATH[ 'home_276' ][f'UAP_{i}'] = UAP
				i += 1
	for model_name in victim_model:
		victim_model_path = os.path.join(config.victim_model_Root,model_name)
		if 'lab_276' in model_name:
			key = 'home_276'
			save_name = f'UAP_signfi_atk_home_vic_lab_scale_1_method_2.mat'
			config.source = 'lab_276'
			_, config.test_data, _, config.test_label = gestureDataLoader.getData( config, 'signfi', ifscale = True )
		elif 'home_276' in model_name:
			key = 'lab_276'
			save_name = f'UAP_signfi_atk_lab_vic_home_scale_1_method_2.mat'
			config.source = 'home_276'
			_, config.test_data, _, config.test_label = gestureDataLoader.getData( config, 'signfi', ifscale = True )
		atk_UAP = UAP_PATH[key]
		psr_range,acc_all = UAPTest(
				victim_model_path = victim_model_path, psr_range = np.linspace( 0, 0.009, 10 ), use_Guas = 1, **atk_UAP
				)
		acc_to_save = {}
		acc_to_save['Guassian_noise'] = copy.deepcopy(acc_all['Guassian_noise'])
		acc_to_save['psr'] = np.linspace(0,0.009,10)
		# acc_all.pop('psr',None)
		acc_all.pop('Guassian_noise',None)
		bf = 1
		for key in acc_all:
			mean = np.asarray(acc_all[key]).mean()
			if bf > mean:
				bf = mean
				out = copy.deepcopy(acc_all[key])
		acc_to_save['accpgd_attack'] = out
		savemat('resultsMat\\Pub_results\\'+save_name,acc_to_save)
def cross_model_test(atk_model_source='home_276',victim_model_source='home_276'):
	widar_dir = {
			'environment_1':[ 'E:\\SensingDataset\\Widar\\20181109',
                                'E:\\SensingDataset\\Widar\\20181115' ],
			'environment_2':['E:\\SensingDataset\\Widar\\20181118\\user2']
	}
	acc_all = { }
	d_set = 'signfi'
	config.source = victim_model_source
	_, config.test_data, _, config.test_label = gestureDataLoader.getData(config, 'signfi', ifscale=True)

	if d_set == 'signfi' and atk_model_source != 'guassian_noise':
		g_flag = False
		# atk_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'cnn', 'vgg8', 'vgg10', 'vgg16', 'vgg19', 'resnet',
		#                      'resnet6' ]
		# vic_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'cnn', 'vgg8', 'vgg10', 'vgg16', 'vgg19', 'resnet',
		#                      'resnet6' ]
		atk_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19']
		vic_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19']
		if atk_model_source == 'home_276':
			config.pert_Mat_Root = 'perturbation\\home_276'
		elif atk_model_source == 'lab_276':
			config.pert_Mat_Root = 'perturbation\\lab_276'
		UAP_files = os.listdir( config.pert_Mat_Root )
		mat_Path = os.path.join(
				'resultsMat\\Pub_results\\cross_model_test\\eleven_model_test', f'cross_model_atk'
				                                                                f'_{atk_model_source}_vic_'
				                                                                f'{victim_model_source}_2.mat'
				)
	elif atk_model_source == 'guassian_noise':
		g_flag = True
		# vic_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'cnn', 'vgg8', 'vgg10', 'vgg16', 'vgg19', 'resnet','resnet6' ]
		atk_architecture = [ 'guassian_noise' ]
		config.pert_Mat_Root = None

	try:
		old_rec = loadmat(mat_Path,squeeze_me = 1)
	except:
		old_rec = {}
		print("There is no records")
	if atk_model_source != 'guassian_noise':
		for vic_model_name in vic_architecture:
			for atk_model_name in atk_architecture:
				perm_name = d_set + '_vic_' + vic_model_name+'_atk_' + atk_model_name
				# skip the tested combinations
				if perm_name in list(old_rec.keys()):
					continue
				acc_buf = [ ]
				t_buffer = { }
				print('===============================================================================================')
				for file_name in UAP_files:
					# select UAP
					if 'UAP' not in file_name:
						continue
					# select dataset name
					if d_set not in file_name:
						continue
					# select attacker source environment
					if atk_model_name != file_name.split( '_' )[ 3 ]:
						continue
					# select victim source environment
					if atk_model_source not in file_name:
						continue
					t_buffer[ file_name ] = file_name
				psr_range,acc = UAPTest(
						victim_model_path = config.victim_model_Root + f'\\signfi_model_'
						                                               f'{vic_model_name}_'
						                                               f'{victim_model_source}_scale_1.h5',
						psr_range = np.concatenate(
								(np.linspace( 0, 0.004, 5 ), 5 * np.logspace( 0.0, 1.0, 5 ) / 1000), axis = 0
								), use_Guas = g_flag, **t_buffer
						)

				acc.pop('Guassian_noise',None)
				# accpgd_attack.pop('psr',None)
				for key in list(acc.keys()):
					acc_buf.append(acc[key])
				min_idx = np.asarray(acc_buf).mean(axis=1).argmin()
				acc_all[perm_name] = np.asarray(acc_buf)[min_idx]
				if 'psr' not in list(old_rec.keys()):
					acc_all['psr'] = psr_range
				old_rec.update(acc_all)
				savemat( mat_Path, old_rec )
				savemat( 'resultsMat\\Pub_results\\cross_model_test\\eleven_model_test\\separate_model\\'+perm_name
				         +'_2.mat',
						acc)
	elif atk_model_source == 'guassian_noise':
		t_buffer = {}
		acc_guassian_noise = {}
		perm_name = d_set + '_vic_' + victim_model_source + '_atk_' + atk_model_source
		for vic_model_name in vic_architecture:

			psr_range, acc = UAPTest(
					victim_model_path = config.victim_model_Root + f'\\signfi_model_'
					                                               f'{vic_model_name}_{victim_model_source}_scale_1.h5',
					psr_range = np.concatenate(
							(np.linspace( 0, 0.004, 5 ), 5 * np.logspace( 0.0, 1.0, 5 ) / 1000), axis = 0
							), use_Guas = g_flag, **t_buffer
					)
			if 'psr' not in list(acc_guassian_noise.keys()):
				acc['psr'] = psr_range

			acc_guassian_noise[vic_model_name] = acc['Guassian_noise']
		savemat(
				'resultsMat\\Pub_results\\cross_model_test\\eleven_model_test\\' + perm_name + '.mat',
				acc_guassian_noise
				)
def pseudo_label_testing(atk_receiver,victim_receiver,UAP_form):
		mat_name_save = f'Unsupervised_labelForm_{UAP_form}_atkRx_{atk_receiver}_vicRx_{victim_receiver}_method_2.mat'
		UAP_PATH = {
				'full_label':{},
				'pseudo_label':{},
				}
		ALL_UAPs = os.listdir( config.pert_Mat_Root )
		n_rx = int(victim_receiver.split('Rx')[1])
		i_model =  n_rx - 1
		victim_model_source = [
				'widar_model_defult_loc5_ori234_Rx1_scale_1_envir_1.h5',
				'widar_model_defult_loc5_ori234_Rx2_scale_1_envir_1.h5',
				'widar_model_defult_loc5_ori234_Rx3_scale_1_envir_1.h5',
				'widar_model_defult_loc5_ori234_Rx4_scale_1_envir_1.h5',
				'widar_model_defult_loc5_ori234_Rx5_scale_1_envir_1.h5',
				'widar_model_defult_loc5_ori234_Rx6_scale_1_envir_1.h5'
		                       ]
		m = 1
		n = 1
		acc_buf = [ ]
		acc_all = { }
		for path in tqdm(ALL_UAPs,desc = 'Selecting UAPs'):
			if 'UAP' not in path or 'widar' not in path:
				continue
			if 'loc5' not in path or 'ori234' not in path or 'envir_1' not in path:
				continue
			if atk_receiver in path:
				if 'surrogateModel_' in path:
					UAP_PATH['pseudo_label'][f'UAP_pseudo_surrogateModel_{m}'] = path
					m+=1
				if 'UAP_widar' in path:
					UAP_PATH['full_label'][f'UAP_full_{n}'] = path
					n+=1
		config.D_range = 1
		config.receiver = [f'r{n_rx}']
		config.location = [ 5 ]
		config.orientation = [ 2,3,4 ]
		config.DNN_name = "defult"
		config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109']
		# config.data_dir = ['/Users/guolinyin/Google 云端硬盘/Colab Notebooks/SensingDataset/Widar/20181109']
		config.train_data, config.test_data, config.train_label,config.test_label  = gestureDataLoader.getData(config,
				'widar' )
		data = np.concatenate((config.train_data,config.test_data),axis=0)
		label = np.concatenate((config.train_label,config.test_label),axis=0)
		psr_range,acc = UAPTest(
				victim_model_path = 'SavedModel/victim_model/' + victim_model_source[ i_model ],
				psr_range = np.linspace( 0, 0.1, 20 ), use_Guas = True, **UAP_PATH[ UAP_form ]
				)
		Guassian = {}
		Guassian['accpgd_attack'] = copy.deepcopy(acc["Guassian_noise"])
		Guassian['psr'] = psr_range
		g_path = os.path.join('resultsMat/Pub_results',f'Widar_atk_Guassian_victim_{victim_receiver}_method_2.mat')


		acc.pop( 'Guassian_noise', None )
		# accpgd_attack.pop('psr',None)
		for key in list( acc.keys( ) ):
			acc_buf.append( acc[ key ] )
		min_idx = np.asarray( acc_buf ).mean( axis = 1 ).argmin( )
		acc_all[ 'accpgd_attack' ] = np.asarray( acc_buf )[ min_idx ]
		acc_all['psr'] = psr_range

		path_to_save = os.path.join('resultsMat/Pub_results',mat_name_save)
		savemat( g_path, Guassian )
		savemat(path_to_save,acc_all)
'''========================Plot========================'''
def plotGuassian_noiseForModel(fname,**model_names):
	marker_dict = {
			'Deepfool': 'o',
			'FGSM': "v",
			'Guassian_Noise': 'X',
			'UAP':'s',
			'UAP_home_to_lab':'h',
			'UAP_lab_to_lab':'s',
			'UAP_lab_to_home':'D',
			'defult': 'H',
			'alexnet': 'H',
			'alex1': 'o',
			'alex2': 'v',
			'alex3': '^',
			'cnn': '<',
			'vgg8': '>',
			'vgg10': '8',
			'vgg16': 's',
			'vgg19': 'p',
			'resnet': 'P',
			'resnet6': '*',
	}
	legend = {
			'alexnet':'Alexnet',
			'alex1':'A1',
			'alex2':'A2',
			'alex3':'A3',
			'vgg19':'VGG19'
			}
	ax = plt.figure( figsize=(8, 5) ).gca( )
	ax.xaxis.set_major_locator( MaxNLocator( integer=True ) )
	psr = model_names['psr']
	model_names.pop('psr')
	model_names.pop('__header__', None)
	model_names.pop('__version__',None)
	model_names.pop('__globals__',None)
	for name in model_names:
		if name not in ['alexnet','alex1','alex2','alex3','vgg19',]:
			continue
		result = model_names[name]
		acc = (result[ 0 ] - result)/result[ 0 ]
		ax.plot(psr[0:8],
				acc[0:8],
				label=legend[name],
				marker = marker_dict[name],
				fillstyle = Line2D.fillStyles[-1])
	ax.ticklabel_format( style='sci', scilimits=(0, 0), axis='x' )
	plt.legend()
	plt.grid(True)
	fsize = 14
	plt.xticks( fontsize=fsize )
	plt.yticks( fontsize=fsize )
	plt.ylim(-0.03,1)
	plt.grid(True)
	ax.set_xlabel( 'PSR', fontsize=fsize )
	ax.set_ylabel( 'Attack Success Rate (ASR)', fontsize=fsize )
	out = os.path.join( 'RESULTS_FIGS', fname )
	plt.savefig( out + '.pdf', bbox_inches = 'tight', )
def plot(pdf_name=None,pltGuassian=0,marker_dict = None,label_dict = None,**mat_names):
	if marker_dict == None:
		marker_dict = {
			'Deepfool': 'o',
			'FGSM': "v",
			'Guassian_Noise': 'X',
			'Guassian_noise': 'X',
			'UAP':'s',
			'UAP_home_to_Lab':'h',
			'UAP_lab_to_lab':'s',
			'UAP_lab_to_Home':'D',
			'defult': 'H',
			'alexnet': 'H',
			'alex1': 'o',
			'alex2': 'v',
			'alex3': '^',
			'cnn': '<',
			'vgg8': '>',
			'vgg10': '8',
			'vgg16': 's',
			'vgg19': 'p',
			'resnet': 'P',
			'resnet6': '*',
			'Rx1': 'P',
			'Rx2': 'p',
			'Rx3': '^',
			'Rx4': '<',
			'Rx5': '>',
			'Rx6': 'v',
			'PGD':'h',
			'PGD_1':'H',
			'PGD_2':'^',
			'PGD_3':'v',
			'PGD_4':'P',
			'PGD_5':'8',
			'PGD_7':'s',
			'PGD_10':'*',
			'PGD_15':'D',
			'PGD_17':'<',
			'PGD_20':'>',

	}

	color_dict = {
			'Deepfool': 'b',
			'FGSM'    : 'm'
	}
	result_dir = 'resultsMat/Pub_results/'
	keys = list( mat_names.keys( ) )
	ax = plt.figure( figsize=(8, 5) ).gca( )
	ax.xaxis.set_major_locator( MaxNLocator( integer=True ) )
	for key in keys:
		path = os.path.join(result_dir,mat_names[key])
		result = loadmat(path,squeeze_me=True)
		psr = result['psr']
		acc = (result[ 'acc' ][0] - result[ 'acc' ])/result[ 'acc' ][0]
		if label_dict == None:
			label = key
		else:
			label = label_dict[key]
		ax.plot(psr[0:7],
				acc[0:7],
				label=label,
				marker = marker_dict[key],
				fillstyle = Line2D.fillStyles[-1])
		if 'Guassian_noise' in result and pltGuassian:
			ax.plot(
					psr,(result[ 'Guassian_noise' ][0] - result[ 'Guassian_noise' ])/result[ 'Guassian_noise' ][0],
					label = 'Guassian noise '+ '(' +key.split('_')[-1] + ')',
			        marker = marker_dict[ key ],
					fillstyle = Line2D.fillStyles[ -1 ]
					)
	ax.ticklabel_format( style='sci', scilimits=(0,0), axis='x' )
	fsize = 14
	plt.xticks( fontsize=fsize )
	plt.yticks( fontsize=fsize )
	plt.ylim(-0.03,1)
	plt.grid(True)
	ax.set_xlabel( 'PSR', fontsize=fsize )
	ax.set_ylabel( 'Attack Success Rate (ASR)', fontsize=fsize )
	ax.legend( fontsize=10, ncol=2,loc = 'lower right',bbox_to_anchor=(1, 0.1))
	if pdf_name is not None:
		out = os.path.join('RESULTS_FIGS',pdf_name)
		plt.savefig( out + '.pdf',bbox_inches='tight',  )
def plot_model_compare(psr_val = 0.0005,ifsave = False,vic_model = ['defult','alex1','alex2','alex3','cnn','vgg8','vgg10','vgg19',],**mat_names):
	for key in mat_names:
		acc_all = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test/'+mat_names[key],squeeze_me=1)
		acc_all.pop( '__header__', None )
		acc_all.pop( '__version__', None )
		acc_all.pop( '__globals__', None )
		title = 'Attacker_'+mat_names[key].split('atk_')[1].split('_')[0] +'_Victim_model_'+mat_names[key].split(
				'vic_')[1].split('_')[0]
		if 'psr' in acc_all.keys():
			psr_range = acc_all['psr']
			# idx = np.where( psr_range == psr_val )[ 0 ][ 0 ]
			idx = ((psr_range - psr_val) ** 2).argmin( )
		else:
			idx = -1

		heatmap( acc_dict = acc_all,
				title=title + f'_PSR={psr_range[idx]:.4f}',
				vic_model = vic_model,
				idx = idx,
				ifsave = ifsave)
def plt_PGD(pdf_name = None,*args, **kwargs):
	'''
	:param args: the index of the psr
	:param kwargs: the filename of the results
	'''
	ax = plt.figure( figsize=(8, 5) ).gca( )
	ax.xaxis.set_major_locator( MaxNLocator( integer=True ) )
	result_dir = 'resultsMat/Pub_results/'
	acc = []
	n_iter = []
	i = 0
	fsize = 14
	for k in kwargs:
		path = os.path.join(result_dir,kwargs[k])
		result = loadmat(path,squeeze_me=True)
		if i == 0:
			psr = result[ 'psr' ]
			print( f'Draw the figure with PSR = {psr[ args ]}' )
			i += 1
		n_iter.append(int(kwargs[k].split('_')[1]))
		# acc.append((result[ 'accpgd_attack' ][0] - result[ 'accpgd_attack' ][args])/result[ 'accpgd_attack' ][0])
		acc.append((result[ 'acc' ][0] - result[ 'acc' ][args])/result[ 'acc' ][0])
	ax.plot(n_iter,acc,label = 'PGD',marker = 'H',
			fillstyle = Line2D.fillStyles[ -1 ])
	ax.set_ylabel( 'Attack Success Rate (ASR)', fontsize = fsize )
	ax.set_xlabel( 'Number of iterations', fontsize = fsize )
	plt.xticks( fontsize=fsize )
	plt.yticks( fontsize=fsize )
	plt.grid(True)
	plt.ylim(0.6,1)
	ax.legend( fontsize = 10, ncol = 2, )
	# plt.gca( ).yaxis.set_major_formatter( StrMethodFormatter( '{x:,.1f}' ) )
	if pdf_name is not None:
		out = os.path.join('RESULTS_FIGS',pdf_name)
		plt.savefig( out + '.pdf',bbox_inches='tight',  )
if __name__ == '__main__':
	'''Cross domains'''
	# cross_domain_test()
	'''Cross models'''
	# for atk_model_source in ['home_276','lab_276']:
	# 	for victim_model_source in ['home_276','lab_276']:
	# 		cross_model_test(atk_model_source,victim_model_source)
	'''PGD attack'''
	# Targeted attack using PGD
	if 0:
		accpgd_TA_attack = {}
		accfgsm_TA_attack = {}
		accgaussian_TA_attack = {}
		for i_rx in [2,]:
			accpgd_TA_attack[ f'Rx{i_rx}' ] = []
			accfgsm_TA_attack[ f'Rx{i_rx}' ] = []
			accgaussian_TA_attack[ f'Rx{i_rx}' ] = []
			for i_label in [4]:
				psr_container, acc_pgd_ta = PGD_widar( n_rx = i_rx, t_label = i_label )
			# psr_container, acc_fgsm_ta = FGSM_widar( n_rx = i_rx, t_label = i_label )
			# psr_container, acc_gaussian_ta = Gaussian_widar( n_rx = i_rx, t_label = i_label )
	# 		accpgd_TA_attack[ f'Rx{i_rx}' ].append(acc_pgd_ta )
	# 		accfgsm_TA_attack[ f'Rx{i_rx}' ].append(acc_fgsm_ta )
	# 		# accgaussian_TA_attack[ f'Rx{i_rx}' ].append(acc_gaussian_ta )
	# psr_container, acc_pgd_ta = PGD_widar( n_rx = 2, t_label = None )
	# accpgd_TA_attack[ 'psr' ] = psr_container
	# accfgsm_TA_attack[ 'psr' ] = psr_container
	# accgaussian_TA_attack[ 'psr' ] = psr_container
	#
	#
	# savemat(f'target_PGD_Rx_{i_rx}_TL_{i_label}.mat',accpgd_TA_attack)
	# savemat(f'target_FGSM_Rx_{i_rx}_TL_{i_label}.mat',accfgsm_TA_attack)
	# savemat(f'target_GAUSSIAN_Rx_{i_rx}_TL_{i_label}.mat',accgaussian_TA_attack)
	'''FGSM attack'''
	# Targeted attack using FGSM


	# psr = loadmat('resultsMat/Pub_results/deepfool_signfi_lab_PSR0to0.004.mat',squeeze_me=True)
	# psr_container, acc_DF = deepfool_signfi_NTA( psr )
	# save_to_mat('resultsMat/Pub_results/deepfool_signfi_lab_PSR0to0.004.mat',
	# 		psr = psr,
	# 		accpgd_attack = acc_DF
	# 		)

		# save = {}
	# psr = 0.0005
		# psr_container, acc_FGSM = FGSM_signfi_NTA( psr )
		# psr_container, acc_DF = deepfool_signfi_NTA( psr )
		# fgsm = {'accpgd_attack':acc_FGSM,
		# 		'psr':psr}
		# deepfool = {'accpgd_attack':acc_DF,
		# 		'psr':psr}
		# savemat(f'resultsMat/Pub_results/fgsm_signfi_lab_PSR0to{psr.max()}.mat',fgsm)
		# savemat(f'resultsMat/Pub_results/deepfool_signfi_lab_PSR0to{psr.max()}.mat',deepfool)
		# Gaussian noise compare
	# a = PGD_signfi_NTA( 0.0005, 20 )
	if 0:
		psr = np.linspace( 0, 0.0005, 9 )

		for i in [4,6,8,9,11,12,13,14,16,18,19]:
			psr_container, acc_PGD = PGD_signfi_NTA( psr ,i)
			pgd = {'accpgd_attack':acc_PGD,
					'psr':psr}
			savemat(f'resultsMat/Pub_results/pgd_{i}_signfi_lab_PSR0to{psr.max()}.mat',pgd)
		# psr_container = np.linspace( 0, 0.0005, 9 )
		# config.source = 'lab_276'
		# config.D_range = 1
		# pretrained_model = tf.keras.models.load_model( 'SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5' )
		# X_train, input_CSI, y_train, labels = gestureDataLoader.getData( config = config, dataset_name = 'signfi' )
		# acc_container = [ ]
		# for psr in psr_container:
		# 	accuracy, _, _ = DeepNet.runAdvExsTestPSR(
		# 			input_CSI,
		# 			labels,
		# 			pretrained_model,
		# 			psr,
		# 			attack_method = 'gaussian'
		# 			)
		# 	acc_container.append( accuracy )
		# save = {
		# 		'psr': psr_container,
		# 		'accpgd_attack': acc_container
		# 		}
		# savemat( 'resultsMat/Pub_results/gaussian_signfi.mat', save )
		# plot(pdf_name = None,
		# 		PGD_1 ='pgd_1_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_2 ='pgd_2_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_3 ='pgd_3_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_5 ='pgd_5_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_7 ='pgd_7_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_10 ='pgd_10_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_15 ='pgd_15_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_17 ='pgd_17_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_20 ='pgd_20_signfi_lab_PSR0to0.0005.mat',
		# 		)
		# plt_PGD(
		# 		'PGD_with_Diiferent_iters',
		# 		3,
		# 		PGD_1 = 'pgd_1_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_2 = 'pgd_2_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_3 = 'pgd_3_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_5 = 'pgd_5_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_7 = 'pgd_7_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_10 = 'pgd_10_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_15 = 'pgd_15_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_17 = 'pgd_17_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_20 = 'pgd_20_signfi_lab_PSR0to0.0005.mat',
		# 		)
		# plot(
		# 		pdf_name = 'compare_deepfool_PGD_FGSM',
		# 		FGSM = 'fgsm_signfi_lab_PSR0to0.0005.mat',
		# 		Deepfool = 'deepfool_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_1 = 'pgd_1_signfi_lab_PSR0to0.0005.mat',
		# 		PGD_2 = 'pgd_2_signfi_lab_PSR0to0.0005.mat',
		#
		# 		PGD_5 = 'pgd_5_signfi_lab_PSR0to0.0005.mat',
		#
		#
		# 		Guassian_Noise = 'gaussian_signfi.mat'
		# 		)
	# plt.plot( psr, acc_FGSM, label = 'FGSM' )
		# plt.plot( psr, acc_DF, label = 'deepfool' )
		# plt.plot( psr, acc_PGD, label = 'PGD' )
		# save = {
		# 		'psr':psr,
		# 		'accpgd_attack':acc_PGD
		# 		'deepfool':acc_DF,
		# 		'FGSM':acc_FGSM,
		# 		'PGD':acc_PGD,
		# 		}
		# # records = loadmat(f'resultsMat/Pub_results/compare_FGSM_DF_PGD20_PSR0to{psr.max()}.mat',squeeze_me = 1)
		# savemat(f'resultsMat/Pub_results/compare_FGSM_DF_PGD20_PSR0to{psr.max()}.mat', save)
		# fgsm = {'accpgd_attack':acc_FGSM,
		# 		'psr':psr}
		# deepfool = {'accpgd_attack':acc_DF,
		# 		'psr':psr}
		# pgd = {'accpgd_attack':acc_PGD,
		# 		'psr':psr}
		# savemat(f'resultsMat/Pub_results/fgsm_signfi_lab_PSR0to{psr.max()}.mat',fgsm)
		# savemat(f'resultsMat/Pub_results/deepfool_signfi_lab_PSR0to{psr.max()}.mat',deepfool)
		# savemat(f'resultsMat/Pub_results/pgd_10_signfi_lab_PSR0to{psr.max()}.mat',pgd)
		# plot(
		# 		pdf_name = 'compare_deepfool_and_FGSM_2',
		# 		FGSM = f'fgsm_signfi_lab_PSR0to{psr.max()}.mat',
		# 		Deepfool = f'deepfool_signfi_lab_PSR0to{psr.max()}.mat',
		# 		PGD = f'pgd_signfi_lab_PSR0to{psr.max()}.mat',
		# 		# Guassian_Noise = 'gaussian_signfi.mat'
		# 		)
	'''Deepfool'''
	# config.source = 'lab_276'
	# config.D_range = 1
	# pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
	# X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'signfi')
	# accuracy,perturb,advData = DeepNet.runAdvExsTestPSR(input_CSI,
	# 											labels,
	# 											pretrained_model,
	# 											psr = None,
	# 											attack_method='deepfool')

	'''Unsupervised learning-based attack'''
	# atk_receiver = 'Rx2'
	# victim_receiver = 'Rx1'
	# UAP_form = 'pseudo_label'
	# pseudo_label_testing( atk_receiver='Rx1',
	# 					  victim_receiver='Rx2',
	# 					  UAP_form='pseudo_label'
	# 					  )
	if 0:
		for UAP_form in ['pseudo_label','full_label']:
			for victim_receiver in ['Rx1', 'Rx2','Rx3', 'Rx4', 'Rx5', 'Rx6']:
				for atk_receiver in ['Rx1', 'Rx2','Rx3', 'Rx4', 'Rx5', 'Rx6']:
					mat_name_save = f'Unsupervised_labelForm_{UAP_form}_atkRx_{atk_receiver}_vicRx_{victim_receiver}_method_2.mat'
					if not os.path.exists(os.path.join( 'resultsMat/Pub_results', mat_name_save )):
						pseudo_label_testing(atk_receiver = atk_receiver,
											victim_receiver = victim_receiver,
											UAP_form = UAP_form
								)
	# In-environment attacks

		vic_r = 3
		plot(
				pdf_name = None,
				pltGuassian = 0,
				Guassian_noise =  f'Widar_atk_Guassian_victim_Rx{vic_r}.mat',
				Rx1 = f'Unsupervised_labelForm_full_label_atkRx_Rx1_vicRx_Rx{vic_r}_method_2.mat',
				Rx2 = f'Unsupervised_labelForm_full_label_atkRx_Rx2_vicRx_Rx{vic_r}_method_2.mat',
				Rx3 = f'Unsupervised_labelForm_full_label_atkRx_Rx3_vicRx_Rx{vic_r}_method_2.mat',
				Rx4 = f'Unsupervised_labelForm_full_label_atkRx_Rx4_vicRx_Rx{vic_r}_method_2.mat',
				Rx5 = f'Unsupervised_labelForm_full_label_atkRx_Rx5_vicRx_Rx{vic_r}_method_2.mat',
				Rx6 = f'Unsupervised_labelForm_full_label_atkRx_Rx6_vicRx_Rx{vic_r}_method_2.mat',
			 )
	#
		plot(
				pdf_name = None,
				pltGuassian = 0,
				Guassian_noise =  f'Widar_atk_Guassian_victim_Rx{vic_r}.mat',
				Rx1=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx1_vicRx_Rx{vic_r}_method_2.mat',
				Rx2=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx2_vicRx_Rx{vic_r}_method_2.mat',
				Rx3=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx3_vicRx_Rx{vic_r}_method_2.mat',
				Rx4=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx4_vicRx_Rx{vic_r}_method_2.mat',
				Rx5=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx5_vicRx_Rx{vic_r}_method_2.mat',
				Rx6 = f'Unsupervised_labelForm_pseudo_label_atkRx_Rx6_vicRx_Rx{vic_r}_method_2.mat',
			 )

		atk_r =3
		vic_r = 4
		plot(
				pdf_name = None,
				pltGuassian = 0,
				Guassian_noise =  f'Widar_atk_Guassian_victim_Rx{vic_r}_method_2.mat',
				marker_dict = {'full_label':'*',
								'pseudo_label':'v',
								'Guassian_noise': 'X',},
				label_dict = {'full_label':'True label',
						'pseudo_label':'Pseudo label',
						'Guassian_noise':'Guassian noise',},
				full_label=f'Unsupervised_labelForm_full_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
				pseudo_label=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
			 )

		atk_r =5
		vic_r = 4
		plot(
				pdf_name = 'Oringinal_label_vs_pseudo_label',
				pltGuassian = 0,
				Guassian_noise =  f'Widar_atk_Guassian_victim_Rx{vic_r}_method_2.mat',
				marker_dict = {'full_label':'*',
								'pseudo_label':'v',
								'Guassian_noise': 'X',},
				label_dict = {'full_label':'True label',
						'pseudo_label':'Pseudo label',
						'Guassian_noise':'Guassian noise',},
				full_label=f'Unsupervised_labelForm_full_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
				pseudo_label=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
			 )
		atk_r =1
		vic_r = 3
		plot(
				pdf_name = None,
				pltGuassian = 0,
				Guassian_noise =  f'Widar_atk_Guassian_victim_Rx{vic_r}_method_2.mat',
				marker_dict = {'full_label':'*',
								'pseudo_label':'v',
								'Guassian_noise': 'X',},
				label_dict = {'full_label':'True label',
						'pseudo_label':'Pseudo label',
						'Guassian_noise':'Guassian noise',},
				full_label=f'Unsupervised_labelForm_full_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
				pseudo_label=f'Unsupervised_labelForm_pseudo_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
			 )
	# a = loadmat(
	# 		'resultsMat/Pub_results/cross_model_test/eleven_model_test/signfi_vic_lab_276_atk_guassian_noise.mat',
	# 		squeeze_me = 1
	# 		)
	# plotGuassian_noiseForModel( fname = 'Guassian_noise_lab', **a )
	# home_to_home = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test'
	#                        '/cross_model_atk_home_276_vic_home_276.mat',squeeze_me=True)
	# home_to_lab = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test'
	#                        '/cross_model_atk_home_276_vic_lab_276.mat',squeeze_me=True)
	# lab_to_home = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test'
	#                        '/cross_model_atk_lab_276_vic_home_276.mat',squeeze_me=True)
	# lab_to_lab = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test'
	#                        '/cross_model_atk_lab_276_vic_lab_276.mat',squeeze_me=True)
	# for i in [0.05]:


	# cross_domain_test()
	# plot('compare_cross_domain',
	# 		UAP_lab_to_home = 'UAP_signfi_atk_lab_vic_home_scale_1.mat',
	# 		UAP_home_to_lab = 'UAP_signfi_atk_home_vic_lab_scale_1.mat')
	# a = loadmat('resultsMat/Pub_results/UAP_signfi_atk_home_vic_lab_scale_1.mat',squeeze_me=1)
	# for env in ['lab_276']:
	# 	cross_model_test(atk_model_source=env,victim_model_source=env)
	if 0:
		print('plot figures')
		plot(
				pdf_name = 'compare_deepfool_and_FGSM',
				FGSM = 'fgsm_signfi_lab.mat',
				Deepfool = 'deepfool_signfi_lab.mat',
				Guassian_Noise = 'gaussian_signfi.mat'
				)

		plot( pdf_name='compare_deepFool_and_UAP_indomain_cross_domain',
			  Deepfool='deepfool_signfi_lab.mat',
			  UAP_lab_to_lab='UAP_signfi_lab_scale_1.mat',
			  Guassian_Noise='gaussian_signfi.mat'
			  )
		plot( pdf_name='compare_deepfool_and_FGSM',
				FGSM = 'fgsm_signfi_lab.mat',
			  Deepfool='deepfool_signfi_lab.mat',
			  Guassian_Noise='gaussian_signfi.mat' )
		plot( pdf_name='Cross_domain_atk_compare', pltGuassian=1,
			  UAP_home_to_lab='UAP_signfi_atk_home_vic_lab_scale_1.mat',
			  UAP_lab_to_home='UAP_signfi_atk_lab_vic_home_scale_1.mat',
			  # UAP_lab_to_lab = 'UAP_signfi_lab_scale_1.mat',
			  )
	plot_model_compare(
			psr_val = 0.0158,
			ifsave = True,
			vic_model = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19', ],
			home_to_home = 'cross_model_atk_home_276_vic_home_276_2',
			home_to_lab = 'cross_model_atk_home_276_vic_lab_276_2',
			lab_to_home = 'cross_model_atk_lab_276_vic_home_276_2',
			lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276_2'
			)
		# plot_model_compare(
		# 		a1 = 'cross_model_test\\eleven_model_test\\cross_model_atk_home_276_vic_lab_276.mat'
		#
		# 		)
	'''UAP atk lab vic lab'''
	# model = [
	# 				'signfi_model_defult_lab_276_scale_1.h5',
	#                 'signfi_model_defult_home_276_scale_1.h5'
	# 		]
	# UAP_files = os.listdir( config.pert_Mat_Root )
	# a = UAPTest(victim_model_path = os.path.join(config.victim_model_Root,model[0]),psr_range=np.linspace(0.0,
	# 		0.00004,9),
	# 		u1 = 'lab_276\\UAP_signfi_model_defult_lab_276_scale_1_seed_7.h5'
	# 		)
# '''=========================================================================================================================================='''
	'''Delayed testing'''
	# config.source = 'lab_276'
	# config.D_range = 1
	# # pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
	# X_train ,  config.test_data , y_train, config.test_label = gestureDataLoader.getData(config = config, dataset_name
	# = 'signfi')
	# acc = []
	# for end in np.linspace(0,200,201):
	# 	psr_container,acc_container = UAPTest(
	# 			victim_model_path = 'SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5', psr_range = [ 0.009 ],
	# 			use_Guas = 0, delay = int( end ), a6 = 'UAP_signfi_model_defult_home_276_scale_1_seed_6.h5'
	# 			)
	# 	acc.append(acc_container[list(acc_container.keys())[1]])
	# plt.plot(np.linspace(100,200,201),acc,label='UAP ($PSR = 10^{-3}$)')
	# plt.xlabel('Time delay')
	# plt.ylabel('Accuracy')
	# plt.grid(True)
	# plt.legend()
	# for i,a in enumerate(acc):
	# 	plt.plot(a,marker = 'o',label = np.linspace(100,200,11)[i])
	# plt.legend()

	#
	# UAP_ACC = acc_container['signfi_defult']
	# GS_ACC = acc_container['Guassian_noise']
	# a = {}
	# a['psr'] = psr_container
	# a['Guassian_noise'] = GS_ACC
	# a['accpgd_attack'] = np.asarray(UAP_ACC)
	# path = 'resultsMat/Pub_results/UAP_signfi_lab_scale_1.mat'
	# savemat(path,a)
'''=========================================================================================================================================='''

'''=========================================================================================================================================='''


	# df_ACC =loadmat('resultsMat/Pub_results/deepfool_signfi_lab.mat',squeeze_me=1)['accpgd_attack']
	#

	# psr_container,acc_container = deepfool_signfi_NTA()
	# a = {}
	# a['psr'] = psr_container
	# a['accpgd_attack'] = np.asarray(acc_container)
	# path = 'resultsMat/Pub_results/deepfool_signfi_lab.mat'
	# savemat(path,a)
	# b = loadmat('resultsMat/Pub_results/gaussian_signfi.mat')