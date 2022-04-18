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
def FGSM_signfi_NTA():
	config.source = 'lab_276'
	config.D_range = 1
	pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
	X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'signfi')
	acc_container = []
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
		acc_container.append(1 - accuracy)
	plt.plot(psr_container,acc_container,label = 'signFi, lab')
	plt.legend()
	plt.grid()
	plt.xlabel('PSR')
	plt.ylabel('Fooling rate')
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
	plt.ylabel('Fooling rate')
#     Targeted attacks ------------- Widar
def FGSM_widar_TA():
	config.D_range = 1
	config.orientation = [  1, 2, 3, 4, 5, 6 ]
	config.location = [ 2, ]
	config.receiver = [ 'r1', 'r2', 'r3', 'r4', 'r5', 'r6' ]
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109','E:\\SensingDataset\\Widar\\20181115' ]
	pretrained_model = tf.keras.models.load_model(
			'SavedModel\\victim_model\\widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.h5')
	X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'widar')
	accuracy,_,_ = DeepNet.runAdvExsTestPSR(
											input_CSI,
											labels,
											pretrained_model,
											psr,
											ifpltcmd =True,
											t_label=1,
									        attack_method='fgsm')

"""1.1.2 PGD attack method"""
def PGD_signfi_NTA():
	config.source = 'lab_276'
	config.D_range = 1
	pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
	X_train ,  input_CSI , y_train, labels = gestureDataLoader.getData(config = config, dataset_name = 'signfi')
	acc_container = []
	psr_container = np.arange(0,0.0041,0.0005)
	for psr in psr_container:
		accuracy,_,_ = DeepNet.runAdvExsTestPSR(
															input_CSI,
															labels,
															pretrained_model,
															psr,
															ifpltcmd =False,
															t_label=None,
													        attack_method='pgd')
		acc_container.append(1 - accuracy)
	plt.plot(psr_container,acc_container,label = 'signFi, lab')
	plt.legend()
	plt.grid()
	plt.xlabel('PSR')
	plt.ylabel('Fooling rate')
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
				psr, #0.0001 => acc = 0.543
				ifpltcmd = False,
				t_label = True,
				attack_method = 'pgd'
				)
		acc_container.append(1 - accuracy)
	plt.plot(psr_container,acc_container,label = 'widar')
	plt.legend()
	plt.grid()
	plt.xlabel('PSR')
	plt.ylabel('Fooling rate')
"""1.1.3 Deepfool attack method"""
# Non-targeted attacks ------------- SignFi
def deepfool_signfi_NTA():
	config.source = 'lab_276'
	config.D_range = 1
	pretrained_model = tf.keras.models.load_model( 'SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5' )
	X_train, input_CSI, y_train, labels = gestureDataLoader.getData( config = config, dataset_name = 'signfi' )
	acc_container = []
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
		# 'SavedModel\\victim_model\\widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5',
		*args,**UAP_file_names):
	# config.D_range = 1
	# if 'widar' in victim_model_path:
	# 	# config.data_dir = ['E:\\SensingDataset\\Widar\\20181118\\user2']
	# 	config.data_dir = [ config.sensingDataset_Root + 'Widar\\' + '20181109',
	# 	                    config.sensingDataset_Root + 'Widar\\' + '20181115' ]
	# 	config.location = [1,2,3,4,5,6]
	# 	config.orientation = [1,2,3,4,5,6]
	# 	config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
	# 			config, 'widar',ifscale = True
	# 			)
	# elif 'signfi' in victim_model_path:
	# 	if 'lab_276' in victim_model_path:
	# 		config.source = 'lab_276'
	# 	if 'home_276' in victim_model_path:
	# 		config.source = 'home_276'
	# 	config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
	# 		config, 'signfi', ifscale=True
	# 	)
	# else:
	# 	raise ValueError('Root path is wrong')
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
		# acc_all[acc_name + '_sim'] = []
		UAP_path = os.path.join(config.pert_Mat_Root,UAP_file_names[name])

		with h5py.File(UAP_path,'r') as f:
			a_group_key = 'universal_perturbation'
			# Get the data
			UAP_data = np.asarray(list( f[ a_group_key ] ))
		# print( f'Testing the attack performance of the {acc_name} generated universal perturbation' )
		for psr in tqdm(psr_range,desc = desc,position = 0):
			# Perturbation calibration
			scaled_uni_per = scaleDeepfool(psr = psr,test_data = config.test_data, perturbation = UAP_data)
			adv_data = config.test_data + scaled_uni_per - scaled_uni_per.mean()
			_,acc = victim_model.evaluate(adv_data,config.test_label,batch_size=32, verbose = 0)
			# acc_sim = simi_pred(
			# 		trained_model = victim_model,
			# 		config = config,
			# 		perturbation = scaled_uni_per - scaled_uni_per.mean()
			# 		)
			acc_all[ acc_name ].append(acc)
			# acc_all[ acc_name + '_sim' ].append(acc_sim)

	if use_Guas:
		noise = np.random.normal( 0, 1, size = (1, 200, 60, 3) )
		for psr in tqdm(psr_range,desc = f'Testing performance of UAP from Guassian_noise',position = 0):
			# acc_awgn = awgn_samples_test( victim_model, config.test_data, config.test_label, psr )
			scaled_uni_per = scaleDeepfool( psr = psr, test_data = config.test_data, perturbation = noise )
			noise_data = config.test_data + scaled_uni_per - scaled_uni_per.mean( )
			_, acc_awgn = victim_model.evaluate( noise_data, config.test_label, verbose = 0 )
			acc_all[ 'Guassian_noise' ].append( acc_awgn )
	# results_path = 'utils/resultsMat/cross_model_test.mat'
	# acc_all['psr'] = psr_range
	# savemat(results_path,acc_all)
	# plotting(psr_range,acc_all)

	return psr_range,acc_all

'''2.1.1 cross-domain'''
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
			save_name = f'UAP_signfi_atk_home_vic_lab_scale_1.mat'
			config.source = 'lab_276'
			_, config.test_data, _, config.test_label = gestureDataLoader.getData( config, 'signfi', ifscale = True )
		elif 'home_276' in model_name:
			key = 'lab_276'
			save_name = f'UAP_signfi_atk_lab_vic_home_scale_1.mat'
			config.source = 'home_276'
			_, config.test_data, _, config.test_label = gestureDataLoader.getData( config, 'signfi', ifscale = True )
		atk_UAP = UAP_PATH[key]
		psr_range,acc_all = UAPTest(victim_model_path = victim_model_path,
				psr_range = np.linspace(0,0.009,10),
				use_Guas = 1,
				**atk_UAP
				)
		acc_to_save = {}
		acc_to_save['Guassian_noise'] = copy.deepcopy(acc_all['Guassian_noise'])
		acc_to_save['psr'] = copy.deepcopy(acc_all['psr'])
		acc_all.pop('psr')
		acc_all.pop('Guassian_noise')
		bf = 1
		for key in acc_all:
			mean = np.asarray(acc_all[key]).mean()
			if bf > mean:
				bf = mean
				out = copy.deepcopy(acc_all[key])
		acc_to_save['acc'] = out
		savemat('resultsMat\\Pub_results\\'+save_name,acc_to_save)
'''2.1.2 cross-model'''
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
		atk_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'cnn', 'vgg8', 'vgg10', 'vgg16', 'vgg19', 'resnet',
		                     'resnet6' ]
		vic_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'cnn', 'vgg8', 'vgg10', 'vgg16', 'vgg19', 'resnet',
		                     'resnet6' ]
		if atk_model_source == 'home_276':
			config.pert_Mat_Root = 'perturbation\\home_276'
		elif atk_model_source == 'lab_276':
			config.pert_Mat_Root = 'perturbation\\lab_276'
		UAP_files = os.listdir( config.pert_Mat_Root )
		mat_Path = os.path.join(
				'resultsMat\\Pub_results\\cross_model_test\\eleven_model_test', f'cross_model_atk'
				                                                                f'_{atk_model_source}_vic_'
				                                                                f'{victim_model_source}.mat'
				)
	elif atk_model_source == 'guassian_noise':
		g_flag = True
		vic_architecture = [ 'defult', 'alex1', 'alex2', 'alex3', 'cnn', 'vgg8', 'vgg10', 'vgg16', 'vgg19', 'resnet','resnet6' ]
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
						                                                 f'{vic_model_name}_{victim_model_source}_scale_1.h5',
						use_Guas = g_flag,
						psr_range = np.concatenate((np.linspace(0,0.004,5),5*np.logspace(0.0, 1.0, 5) / 1000),axis=0),
						**t_buffer
						)

				acc.pop('Guassian_noise',None)
				# acc.pop('psr',None)
				for key in list(acc.keys()):
					acc_buf.append(acc[key])
				min_idx = np.asarray(acc_buf).mean(axis=1).argmin()
				acc_all[perm_name] = np.asarray(acc_buf)[min_idx]
				if 'psr' not in list(old_rec.keys()):
					acc_all['psr'] = psr_range
				old_rec.update(acc_all)
				savemat( mat_Path, old_rec )
				savemat( 'resultsMat\\Pub_results\\cross_model_test\\eleven_model_test\\separate_model\\'+perm_name+'.mat',
						acc)
	elif atk_model_source == 'guassian_noise':
		t_buffer = {}
		acc_guassian_noise = {}
		perm_name = d_set + '_vic_' + victim_model_source + '_atk_' + atk_model_source
		for vic_model_name in vic_architecture:

			psr_range, acc = UAPTest(
					victim_model_path = config.victim_model_Root + f'\\signfi_model_'
					                                               f'{vic_model_name}_{victim_model_source}_scale_1.h5',
					use_Guas = g_flag,
					psr_range = np.concatenate(
							(np.linspace( 0, 0.004, 5 ), 5 * np.logspace( 0.0, 1.0, 5 ) / 1000), axis = 0
							),
					**t_buffer
					)
			if 'psr' not in list(acc_guassian_noise.keys()):
				acc['psr'] = psr_range

			acc_guassian_noise[vic_model_name] = acc['Guassian_noise']
		savemat(
				'resultsMat\\Pub_results\\cross_model_test\\eleven_model_test\\' + perm_name + '.mat',
				acc_guassian_noise
				)
'''2.1.3 cross-domain and cross-model'''
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
		acc = result[ 0 ] - result
		ax.plot(psr[0:8],
				acc[0:8],
				label=name,
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
	ax.set_ylabel( 'Fooling rate', fontsize=fsize )
	out = os.path.join( 'RESULTS_FIGS', fname )
	plt.savefig( out + '.pdf', bbox_inches = 'tight', )
def plot(pdf_name,pltGuassian=0,**mat_names):

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
		acc = result[ 'acc' ][0] - result[ 'acc' ]
		ax.plot(psr,
				acc,
				label=key,
				marker = marker_dict[key],
				fillstyle = Line2D.fillStyles[-1])
		if 'Guassian_noise' in result and pltGuassian:
			ax.plot(
					psr,result[ 'Guassian_noise' ][0] - result[ 'Guassian_noise' ],
					label = 'Guassian noise '+key.split('_')[-1],
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
	ax.set_ylabel( 'Fooling rate', fontsize=fsize )
	ax.legend( fontsize=10, ncol=2,)
	if pdf_name is not None:
		out = os.path.join('RESULTS_FIGS',pdf_name)
		plt.savefig( out + '.pdf',bbox_inches='tight',  )
def plot_model_compare(psr_val = 0.0005,ifsave = False,vic_model = ['defult','alex1','alex2','alex3','cnn','vgg8','vgg10','vgg19',],**mat_names):
	for key in mat_names:
		acc_all = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test/'+mat_names[key],squeeze_me=1)
		acc_all.pop( '__header__', None )
		acc_all.pop( '__version__', None )
		acc_all.pop( '__globals__', None )
		title = 'Attacker '+mat_names[key].split('atk_')[1].split('_')[0]+ ' Victim model '+mat_names[key].split(
				'vic_')[1].split('_')[0]
		if 'psr' in acc_all.keys():
			psr_range = acc_all['psr']
			# idx = np.where( psr_range == psr_val )[ 0 ][ 0 ]
			idx = ((psr_range - psr_val) ** 2).argmin( )
		else:
			idx = -1

		heatmap( acc_dict = acc_all,
				title=title + f' PSR = {psr_range[idx]:.4f}',
				vic_model = vic_model,
				idx = idx,
				ifsave = ifsave)
if __name__ == '__main__':
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
	# plot_model_compare(
	# 		psr_val = 0.0158,
	# 		ifsave = True,
	# 		vic_model = ['defult','alex1','alex2','alex3','vgg19',],
	# 		home_to_home = 'cross_model_atk_home_276_vic_home_276',
	# 		home_to_lab = 'cross_model_atk_home_276_vic_lab_276',
	# 		lab_to_home = 'cross_model_atk_lab_276_vic_home_276',
	# 		lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276')

	# cross_domain_test()
	# plot('compare_cross_domain',
	# 		UAP_lab_to_home = 'UAP_signfi_atk_lab_vic_home_scale_1.mat',
	# 		UAP_home_to_lab = 'UAP_signfi_atk_home_vic_lab_scale_1.mat')
	# a = loadmat('resultsMat/Pub_results/UAP_signfi_atk_home_vic_lab_scale_1.mat',squeeze_me=1)
	# for env in ['lab_276']:
	# 	cross_model_test(atk_model_source=env,victim_model_source=env)
	plot( pdf_name='compare_deepFool_and_UAP_indomain_cross_domain',
		  Deepfool='deepfool_signfi_lab.mat',
		  UAP_lab_to_lab='UAP_signfi_lab_scale_1.mat',
		  Guassian_Noise='gaussian_signfi.mat'
		  )

	if 0:
		print('plot figures')
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
				a1 = 'cross_model_test\\eleven_model_test\\cross_model_atk_home_276_vic_lab_276.mat'

				)
# '''=========================================================================================================================================='''
# 	config.source = 'lab_276'
# 	config.D_range = 1
# 	# pretrained_model = tf.keras.models.load_model('SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5')
# 	X_train ,  config.test_data , y_train, config.test_label = gestureDataLoader.getData(config = config, dataset_name = 'signfi')
# 	psr_container,acc_container = UAPTest(
# 											victim_model_path =
# 											'SavedModel\\PSR\\signfi_model_lab_276_scale_1.h5',
# 											# 'SavedModel\\victim_model\\signfi_model_defult_lab_276_scale_1.h5',
# 											use_Guas=1,
# 											psr_range = np.linspace(0,0.009,10),
# 											a1 = 'UAP_signfi_model_defult_home_276_scale_1_seed_1.h5',
# 											a2 = 'UAP_signfi_model_defult_home_276_scale_1_seed_2.h5',
# 											a3 = 'UAP_signfi_model_defult_home_276_scale_1_seed_3.h5',
# 											a4 = 'UAP_signfi_model_defult_home_276_scale_1_seed_4.h5',
# 											a6 = 'UAP_signfi_model_defult_home_276_scale_1_seed_6.h5',
# 											a7 = 'UAP_signfi_model_defult_home_276_scale_1_seed_7.h5',
# 											a8 = 'UAP_signfi_model_defult_home_276_scale_1_seed_8.h5',
# 											a9 = 'UAP_signfi_model_defult_home_276_scale_1_seed_9.h5',
# 											a10 = 'UAP_signfi_model_defult_home_276_scale_1_seed_10.h5',
# 			)
	#
	# UAP_ACC = acc_container['signfi_defult']
	# GS_ACC = acc_container['Guassian_noise']
	# a = {}
	# a['psr'] = psr_container
	# a['Guassian_noise'] = GS_ACC
	# a['acc'] = np.asarray(UAP_ACC)
	# path = 'resultsMat/Pub_results/UAP_signfi_lab_scale_1.mat'
	# savemat(path,a)
'''=========================================================================================================================================='''

'''=========================================================================================================================================='''


	# df_ACC =loadmat('resultsMat/Pub_results/deepfool_signfi_lab.mat',squeeze_me=1)['acc']
	#

	# psr_container,acc_container = deepfool_signfi_NTA()
	# a = {}
	# a['psr'] = psr_container
	# a['acc'] = np.asarray(acc_container)
	# path = 'resultsMat/Pub_results/deepfool_signfi_lab.mat'
	# savemat(path,a)
	# b = loadmat('resultsMat/Pub_results/gaussian_signfi.mat')