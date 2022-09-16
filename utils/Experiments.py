import numpy as np
import sys
import os
import copy
import tensorflow as tf
from tensorflow.keras import backend as K
current_dir = os.getcwd( )
sys.path.append( current_dir )
sys.path.append( current_dir + '\\utils' )
sys.path.append( 'G:\\我的云端硬盘\\Colab Notebooks\\SensingDataset\\SignFi\\Dataset' )
sys.path.append( "/Users/guolinyin/Google 云端硬盘/Colab Notebooks/AdvAttackandDefense/utils" )
import Config, SignalPreprocess, gestureDataLoader, DeepNet, plotSig
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from DeepFool import deepfool
from Universal_pert import universal_perturbation,universal_perturbation_PGD
from matplotlib.lines import Line2D
import h5py
from TOOLS import *
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import gc
# from tensorflow.keras.layers import Dense, Input, Softmax,ZeroPadding2D,MaxPooling2D,Conv2D,Flatten,Lambda,Dropout,LSTM
from tensorflow.keras.models import Model
procOBJ = gestureDataLoader.preprocessing( )
config = Config.getconfig()
'''Tools'''
def UAPTest_with_psr(Attack_model,test_data,test_label,v,psr):
	scale_factor = np.sqrt( psr * test_data.var( ) * ((v.max( ) - v.min( )) ** 2) / v.var( ) )
	scaled_uni_per = procOBJ.scale( v, scale_factor )
	adv_data = test_data + np.repeat( scaled_uni_per, test_data.__len__( ), axis = 0 )
	adapted_adv_data = adv_data - np.abs( np.mean( adv_data ) - np.mean( test_data ) )
	# Attack_model.evaluate( test_data, test_label )
	_, acc = Attack_model.evaluate( adapted_adv_data, test_label, verbose = 0 )
	return acc
def compute_scale_factor( psr, v, ori_data ):
	'''
	:param psr: input psr for each sample
	:param v: input perturbation
	'''
	n = v.shape[ 0 ]
	re_v = v.reshape( n, -1 )
	scale_factor = np.sqrt(
			psr * ori_data.var( ) * ((re_v.max( axis = 1 ) - re_v.min( axis = 1 )) ** 2) / re_v.var(
					axis = 1
					)
			)
	return scale_factor.expand_dims( axis = 1 )
def awgn_samples_test(Attack_model,test_data,test_label,psr):
	'''
	:param Attack_model: the test model
	:param test_data: the test data
	:param test_label: the label
	:param psr: the required psr under testing
	:return: the test accuracy
	'''
	noise = np.random.normal( 0, np.sqrt(psr*test_data.var( )), size = (1,200, 60, 3) )
	noisy_data = test_data+noise
	_, acc = Attack_model.evaluate( noisy_data, test_label, verbose = 0 )
	return acc
def genereate_UAP(dataset,model_path,config,method = 'deepfool',labels = None,psr = None):
	'''
	:param dataset: the dataset to loop over
	:param model_path: the attack model path
	:return: the UAP
	'''
	victim_model = tf.keras.models.load_model(model_path )


	if method == 'deepfool':
		if 'vgg' in model_path:
			victim_model.layers[ -1 ].activation = None
			f = tf.keras.Model( victim_model.input, victim_model.layers[ -1 ].output )
		elif 'densenet' in model_path:
			victim_model.layers[ -1 ].activation = None
			f = tf.keras.Model( victim_model.input, victim_model.layers[ -1 ].output )
		else:
			f = tf.keras.Model( victim_model.input, victim_model.layers[ -2 ].output )
		if f.output_shape[ 1 ] != config.N_classes:
			raise Exception(
					f'The output of the feed forward function is wrong, the output should be {config.N_classes}, '
					f'but it is {f.output_shape[ 1 ]}'
					)
		# if f.layers[-1].activation != None:
		# 	print(f'The last layer activation function has not been close for model {model_path}')
		# 	f.layers[ -1 ].activation = None
		UAP = universal_perturbation( dataset = dataset, f = f, overshoot = 0.002 )
	elif method == 'pgd':
		UAP = universal_perturbation_PGD(dataset = dataset,labels = labels,f = victim_model,psr = psr)
	return UAP
def scaleDeepfool(psr,test_data,perturbation):
	scale_factor = np.sqrt(
			psr * test_data.var( ) * ((perturbation.max( ) - perturbation.min( )) ** 2) /
			perturbation.var( )
			)
	scaled_perturbation = procOBJ.scale( perturbation, scale_factor )
	return scaled_perturbation
def simi_pred(trained_model,config,perturbation):
	train_label = np.argmax(config.train_label,axis = 1)
	n_classes = train_label.max() + 1
	idx_all = []
	for i in range(n_classes):
		idx = np.where(train_label == i)[0]
		id_this = np.random.choice(idx,1,replace = 0)
		idx_all.append(id_this)
	idx_all = np.squeeze(np.asarray(idx_all))
	support_data = config.train_data[idx_all]
	adv_data = config.test_data + perturbation
	fe = Model( inputs = trained_model.input, outputs = trained_model.layers[-2].output )
	# ori_pred = fe.predict(config.test_data)
	adv_pred = fe.predict(adv_data)
	sup_pred = fe.predict(support_data)
	pred_labels = [ cosine_similarity( adv_pred[ i:i + 1 ], sup_pred ).argmax( axis=1 ) for i in range(
			adv_pred.__len__( )	) ]
	acc = np.sum(np.asarray(pred_labels).squeeze() == config.test_label.argmax(axis=1))/config.test_data.__len__( )
	return acc
def heatmap(acc_dict,title,vic_model = ['defult', 'alex1', 'alex2', 'alex3', 'vgg19'],idx = -1,ifsave = False):
	import numpy as np
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	atk_model = copy.deepcopy( vic_model )
	atk_model.reverse( )
	print_name_dict = {
			'defult': 'Alexnet',
			'alex1': 'A1',
			'alex2': 'A2',
			'alex3': 'A3',
			'vgg19': 'VGG19'
			}
	atk_model_print = [print_name_dict[key] for key in atk_model]
	vic_model_print = [print_name_dict[key] for key in vic_model]
	def getAccMatrix( acc_dict, idx, vic_model,atk_model ):
		# atk_model = copy.deepcopy( vic_model )
		# atk_model.reverse( )
		keys = list( acc_dict.keys( ) )
		acc_matrix = [ ]
		acc_matrix_ori = []
		for atk in atk_model:
			vic_acc = [ ]
			vic_ori_acc = [ ]
			for vic in vic_model:
				for key in keys:
					if key =='psr':
						continue
					if vic != key.split( '_' )[ 2 ] or atk != key.split( '_' )[ 4 ]:
						continue
					vic_acc.append( acc_dict[ key ][ idx ] )
					vic_ori_acc.append(acc_dict[key][0])
			acc_matrix.append( vic_acc )
			acc_matrix_ori.append(vic_ori_acc)
		return np.asarray(acc_matrix),np.asarray(acc_matrix_ori)

	acc,acc_ori = getAccMatrix(acc_dict = acc_dict,
								vic_model = vic_model,
								atk_model = atk_model,
								idx = idx)
	'''Model specific factors'''
	# atk_model = [ 'cnnlstm', 'cnn', 'alex3', 'alex2', 'alex1', 'default' ]
	# victim = [ 'default', 'alex1', 'alex2', 'alex3', 'cnn', 'cnnlstm' ]

	# accpgd_attack = np.round( 1 - np.array( [
	# 		[ 0.395, 0.355, 0.551, 0.404, 0.298, 0.098, ],
	# 		[ 0.812, 0.766, 0.992, 0.687, 0.55, 0.475, ],
	# 		[ 0.506, 0.4438, 0.669, 0.393, 0.324, 0.21, ],
	# 		[ 0.364, 0.2857, 0.572, 0.416, 0.375, 0.158, ],
	# 		[ 0.33, 0.1896, 0.504, 0.424, 0.273, 0.192, ],
	# 		[ 0.383, 0.274, 0.584, 0.433, 0.332, 0.196, ],] ),2 )
	acc = np.round((acc_ori-acc)/acc_ori,2)
	'''Task specific factors'''
	# target models using default model
	# atk_model= ['wiar','widar','signfi']
	# victim= ['signfi','widar','wiar']
	# accpgd_attack = np.round(1 - np.array([
	# 		[0.38,0.25,0.329],
	# 		[0.5765,0.194,0.972],
	# 		[0.2346,0.42,0.966]
	# 		]),2)
	fig, ax = plt.subplots( )
	im = ax.imshow( acc,cmap = 'magma_r',vmin=0.65, vmax=1 )
	# im = plt.imshow( acc, cmap = 'magma_r' )
	# Show all ticks and label them with the respective list entries
	plt.yticks(  np.arange( len(atk_model ) ),atk_model_print )
	plt.xticks(  np.arange( len(vic_model ) ),vic_model_print )
	# plt.title(title)
	# Rotate the tick labels_pred and set their alignment.
	plt.setp( ax.get_xticklabels( ), rotation=45, ha="right",
			  rotation_mode="anchor" )
	# Loop over data dimensions and create text annotations.
	for i in range( len( vic_model ) ):
		for j in range( len( atk_model ) ):
			text = ax.text( j, i, acc[ i, j ],
							ha="center", va="center", color="white" )

	plt.colorbar(im)
	# plt.clim( 0, 1 )
	fig.tight_layout( )
	plt.show( )
	if ifsave:
		out = os.path.join('RESULTS_FIGS',title)
		plt.savefig( out + '.pdf',bbox_inches='tight',  )
'''Experiments'''
def compare_deepFool_and_FGSM_UAP():
	config = Config.getconfig( )
	config.pert_Mat_Root = 'perturbation'
	config.attacker_model_Root = 'SavedModel\\PSR'
	results_path = os.path.join( config.results_dir, 'compare_deepFool_and_FGSM.mat' )
	# if os.path.exists(results_path):
	if False:
		psr_all = loadmat(results_path,squeeze_me = 1)['psr']
		fgsm_all = loadmat( results_path,squeeze_me = 1 )[ 'fgsm_acc' ]
		awgn_all = loadmat( results_path,squeeze_me = 1 )[ 'awgn_acc' ]
		df_all = loadmat( results_path, squeeze_me = 1)[ 'deepfool' ]
		UAP_all = loadmat( results_path, squeeze_me = 1 )[ 'UAP' ]
	else:
		path = 'perturbation/UAP_widar_model_original_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5'
		with h5py.File(path, 'r') as f:
			a_group_key = list( f.keys( ) )[ 0 ]
			# Get the data
			cross_uni_per = np.asarray(list( f[ a_group_key ] ))
		uni_per = loadmat( os.path.join(config.pert_Mat_Root,'signfi_lab_276_universal_perturbation_0.02.mat') )[ 'universal_perturbation' ]
		deepfool_per = loadmat( os.path.join(config.pert_Mat_Root,'signfi_lab_276_deepfool.mat'))[ 'perturbation' ]
		# uni_per_signfi = loadmat( os.path.join( config.pert_Mat_Root, 'signfi_lab_276_universal_perturbation.mat' ) )['universal_perturbation' ]
		'''Preparing dataset and model'''
		# model_name_widar = 'widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118'+'.h5'
		# widar_path = 'SavedModel\\\victim_model\\' +model_name_widar
		model_name = 'signfi_model_lab_276_scale_50'+'.h5'
		Attack_model = tf.keras.models.load_model(
				os.path.join(
						config.attacker_model_Root,
						model_name
						)
				)
		config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
				config, 'signfi'
				)
		config.test_data = procOBJ.scale(config.test_data,50)
		df_all = []
		fgsm_all = []
		UAP_all = []
		psr_all = []
		awgn_all = []
		sca_range = np.arange(0.9,2.5,0.03)
		for sca in sca_range:
			# Evaluation of the deepfool
			add_per = deepfool_per*sca
			adv_data = config.test_data + add_per
			_,acc_df = Attack_model.evaluate(adv_data,config.test_label,verbose = 0)

			psr = add_per.reshape(add_per.__len__(),-1).var(axis=1).mean()/config.test_data.var( )
			acc_awgn = awgn_samples_test( Attack_model, config.test_data, config.test_label, psr )
			acc_fgsm,_,_ = DeepNet.runAdvExsTestPSR(
					input_CSI = config.test_data,
					labels = config.test_label,
					pretrained_model = Attack_model,
					psr = psr,
					t_label = None
					)
			acc_Uni = UAPTest_with_psr( Attack_model, test_data = config.test_data, test_label = config.test_label,v = uni_per,
					psr = psr )
			acc_Uni_cross = UAPTest_with_psr( Attack_model, test_data = config.test_data, test_label =
			config.test_label,
					v = uni_per,
					psr = psr )
			print(f'The PSR is: {psr:.6f}, '
				  f'\n deepfool: {acc_df:.4f}, '
				  f'\n UAP: {acc_Uni:.4f} '
				  f'\n FGSM: {acc_fgsm:.4f} '
				  f'\n Guassian noise: {acc_awgn:.4f}',)
			print('===============================================================================================================')
			fgsm_all.append(acc_fgsm)
			df_all.append(acc_df)
			psr_all.append(psr)
			UAP_all.append(acc_Uni)
			awgn_all.append(acc_awgn)

		else:
			acc_whole = {
					'fgsm_acc': fgsm_all,
					'deepfool': df_all,
					'UAP': UAP_all,
					'awgn_acc':awgn_all,
					'psr'     : psr_all
					}
		savemat( results_path, acc_whole )

	label = ['one-step FGSM','DeepFool','UAP','Guassian noise']
	accuracy = [fgsm_all,df_all,UAP_all,awgn_all]
	marker = ['s','v','P','o']
	s = 7
	plt.figure()
	idx = np.arange(0,psr_all.__len__(),2)
	for i in range(accuracy.__len__( )):

		plt.plot(psr_all[idx],accuracy[i][idx],label = label[i],marker =marker[i],ms = s,fillstyle=Line2D.fillStyles[
			-1] )
	# plt.plot( psr_all, fgsm_all, label = 'one-step FGSM', marker = 's',ms=s ,fillstyle
    #     =Line2D.fillStyles[-1])
	# plt.plot( psr_all, df_all, label = 'DeepFool', marker = 'o' ,ms=s,fillstyle
    #     =Line2D.fillStyles[-1])
	# plt.plot( psr_all, UAP_all, label = 'UAP', marker = '4' ,ms=s,fillstyle
    #     =Line2D.fillStyles[-1])
	# plt.plot( psr_all, awgn_all, label = 'Guassian noise', marker = 'v',ms=s ,fillstyle
    #     =Line2D.fillStyles[-1])
	plt.ylabel( 'Accuracy' )
	plt.xlabel( 'PSR' )
	plt.grid( alpha = 0.4 )
	plt.legend( )
	plt.show()
def cross_domain_UAP_test_widar():
	acc_all = { }
	name = [ 'Widar U/P, envir 2', 'signfi U/P', 'Widar U/P, same envir' ]
	results_path = os.path.join(config.results_dir,'compare_UAP_signfi_widar_envir12.mat')
	# if os.path.exists( results_path ):
	if False:
		psr_range = np.arange( 0.00, 0.055, 0.002 )
		for i in range( name.__len__( ) ):
			acc_all[name[i]] = loadmat( results_path , squeeze_me = 1 )[name[i]]
	else:
		'''Prepare data'''
		config.D_range = 1
		procOBJ = gestureDataLoader.preprocessing( )
		config.data_dir = [
				'E:\\SensingDataset\\Widar\\20181118\\user2'
				]
		config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
				config, 'widar'
				)
		test_data = copy.deepcopy( procOBJ.scale( config.test_data, config.D_range ) )
		train_data = copy.deepcopy( procOBJ.scale( config.train_data, config.D_range ) )
		test_label = copy.deepcopy( config.test_label )
		train_label = copy.deepcopy( config.train_label )
		config.attacker_model_Root = 'SavedModel\\victim_model'
		# runTrain( config = config, dataset_name = 'widar' )
		Attack_model = tf.keras.models.load_model(
				os.path.join(
						config.attacker_model_Root,
						'widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5'
						)
				)
		# f = Model(Attack_model.input,Attack_model.get_layer('FC_2').output)
		# uni_per_widar_in_domain = universal_perturbation(dataset = config.test_data,f = f,overshoot=0.002)
		'''Cross domain universal perturbation testing'''
		config.pert_Mat_Root = 'utils\\perturbation'
		uni_per_widar_in_domain = loadmat(os.path.join(config.pert_Mat_Root,
            'uni_per_widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.mat' ))['universal_perturbation']
		uni_per_widar = loadmat(
				os.path.join(
						config.pert_Mat_Root,
						'uni_per_widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.mat'
						)
				)[ 'universal_perturbation' ]
		uni_per_signfi = loadmat( os.path.join( config.pert_Mat_Root, 'signfi_lab_276_universal_perturbation.mat' ) )[
			'universal_perturbation' ]
		# model = tf.keras.models.load_model( os.path.join(model_path,
		#         'widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.h5'))
		for i, perturb in enumerate( [ uni_per_widar, uni_per_signfi, uni_per_widar_in_domain] ):
			acc_all[ name[ i ] ] = [ ]
			v = perturb
			psr_range = np.arange( 0.00, 0.055, 0.002 )
			print( f'Testing the attack performance of the {name[ i ]} generated universal perturbation' )
			for psr in psr_range:
				# Perturbation calibration
				scale_factor = np.sqrt( psr * test_data.var( ) * ((v.max( ) - v.min( )) ** 2) / v.var( ) )
				scaled_uni_per = procOBJ.scale( v, scale_factor )
				adv_data = test_data + np.repeat( scaled_uni_per, test_data.__len__( ), axis = 0 )
				adapted_adv_data = adv_data - np.abs( np.mean( adv_data ) - np.mean( test_data ) )
				# Attack_model.evaluate( test_data, test_label )
				_, acc = Attack_model.evaluate( adapted_adv_data, test_label, verbose = 0 )
				print( f'The PSR is {scaled_uni_per.var( ) / test_data.var( ):.4f}, accuracy is {acc:.4f}' )
				acc_all[ name[ i ] ].append( acc )
		# else:
	marker = [ 's', 'v', 'P', 'o' ]
	plt.figure( )
	for k in range( name.__len__( )):
		plt.plot( psr_range, acc_all[ name[ k ] ], label = name[ k ], marker = marker[k], markersize = 7,
				fillstyle=Line2D.fillStyles[
			-1] )

	plt.legend( )
	plt.ylabel( 'Accuracy' )
	plt.xlabel( 'PSR' )
	plt.grid( alpha = 0.4 )
	plt.show( )
def compare_DNN_difference(
		# victim_model_path = 'SavedModel\\PSR\\' + 'signfi_model_lab_276_scale_1.h5',
		# victim_model_path = config.attacker_model_Root  + '\\wiar_model_defult_scale_1numuser_10.h5',
		victim_model_path = config.victim_model_Root +
		                    '\\widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5',
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
	psr_range = np.arange( 0.000, 0.011, 0.005 )
	acc_all['Guassian_noise'] = []
	# for name in tqdm(file_name_list,desc = 'Testing...',position = 0):
	for name in file_name_list:
		DNN_name = UAP_file_names[name].split('_')[3]
		data_name = UAP_file_names[ name ].split( '_' )[ 1 ]
		if 'seed' in name:
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
	# acc_all['psr_range'] = psr_range
	# savemat(results_path,acc_all)
	# plotting(psr_range,acc_all)

	return acc_all

@saveToPath
def impact_domain_on_environment1_widar():
	import re
	config.pert_Mat_Root = 'utils\\perturbation\\UAP_H5'
	config.data_dir = ['E:\\SensingDataset\\Widar\\20181118\\user2']
	UAP_PATH = os.listdir(config.pert_Mat_Root)
	acc_all = {}
	config.location = [ 2 ]
	config.orientation = [ 1, 2, 3, 4, 5, 6 ]
	config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
			config,
			dataset_name = 'widar'
			)
	config.D_range = 1
	test_data = procOBJ.scale( config.test_data, config.D_range )
	# train_data = procOBJ.scale( config.train_data, config.D_range )
	config.attacker_model_Root = 'SavedModel\\victim_model'
	attack_model = tf.keras.models.load_model(
			os.path.join(
					config.attacker_model_Root,
					'widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5'
					)
			)
	for name in UAP_PATH:
		if "UAP" not in name:
			continue
		acc_all[name] = []
		# config.location = [int(id) for id in re.findall( r'\d+',  name.split('_')[3])[0]]
		# config.orientation = [int(id) for id in re.findall( r'\d+',  name.split('_')[4])[0]]
		config.pert_Mat_Root = 'utils\\perturbation\\UAP_H5'
		per_PATH = os.path.join(config.pert_Mat_Root,name)
		with h5py.File( per_PATH, "r" ) as f:
			a_group_key = list( f.keys( ) )[ 0 ]
			# Get the data
			perturbation = np.asarray(list( f[ a_group_key ] ))
		psr_range = np.arange( 0.00, 0.055, 0.002 )
		for psr in psr_range:
			# Perturbation calibration
			scale_factor = np.sqrt( psr * test_data.var( ) * ((perturbation.max( ) - perturbation.min( )) ** 2) / perturbation.var( ) )
			scaled_uni_per = procOBJ.scale( perturbation, scale_factor )
			adv_data = test_data + scaled_uni_per
			# adapted_adv_data = adv_data - np.abs( np.mean( adv_data ) - np.mean( test_data ) )
			_,acc = attack_model.evaluate(adv_data,config.test_label)
			acc_all[ name ].append(acc)
	return acc_all
if __name__ == '__main__':
	# heatmap()
	# test()
	# compare_deepFool_and_FGSM_UAP()
	'''Compare task factors'''
	# config.data_dir = [
	#                       config.sensingDataset_Root + 'Widar\\' + '20181109',
	#                     config.sensingDataset_Root + 'Widar\\' + '20181115'
	#                     # config.sensingDataset_Root + 'Widar\\' + '20181118'
	#                     ]
	# config.location = [ 2,]
	# config.orientation = [ 1,2,3,4,5,6 ]
	# config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
	# 		config, 'wiar', ifscale = True
	# 		)
	# config.source = 'lab_276'
	# config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
	# 	config, 'signfi', ifscale=True
	# )
	# UAP_files = os.listdir(config.pert_Mat_Root)
	# acc_all = { }
	# for d_set in ['wiar','widar','signfi']:
	#
	# 	for vic_model_name in ['defult']:#,'alex1','alex2','alex3','cnn','cnnlstm']:
	# 		t_buffer = {}
	#
	# 		acc_buf = []
	# 		for file_name in UAP_files:
	# 			if 'UAP' not in file_name:
	# 				continue
	# 			if d_set not in file_name:
	# 				continue
	# 			if vic_model_name != file_name.split('_')[3]:
	# 				continue
	# 			print(file_name)
	# 		# print('===============================================================================================')
	# 			t_buffer[file_name] = file_name
	# 		accpgd_attack = compare_DNN_difference(**t_buffer)
	# 		accpgd_attack.pop('Guassian_noise',None)
	# 		for key in list(accpgd_attack.keys()):
	# 			acc_buf.append(accpgd_attack[key])
	# 		acc_all[d_set + '_' + vic_model_name] = np.asarray(acc_buf).mean(axis = 0)
	# 		savemat(f'acc_{d_set}_{vic_model_name}', mdict = acc_all)
	'''Compare model factors'''
	if 0:
		UAP_files = os.listdir( config.pert_Mat_Root )
		acc_all = { }
		d_set = 'signfi'
		config.source = 'lab_276'
		_, config.test_data, _, config.test_label = gestureDataLoader.getData(
			config, 'signfi', ifscale=True
		)
		mat_Path = 'architecture_compare.mat'
		try:
			old_rec = loadmat(mat_Path,squeeze_me = 1)
		except:
			old_rec = {}
			print("There is no records")
		architecture = ['vgg16','vgg19','resnet','resnet6','resnet10','resnet12']
		for vic_model_name in architecture:
			for atk_model_name in architecture:
				perm_name = d_set + '_vic_' + vic_model_name+'_atk_' + atk_model_name
				if perm_name in list(old_rec.keys()):
					continue
				acc_buf = [ ]
				t_buffer = { }
				print('===============================================================================================')
				for file_name in UAP_files:
					if 'UAP' not in file_name:
						continue
					if d_set not in file_name:
						continue
					if atk_model_name != file_name.split( '_' )[ 3 ]:
						continue
					t_buffer[ file_name ] = file_name
				acc = compare_DNN_difference(
						victim_model_path = config.victim_model_Root + f'\\signfi_model_'
						                                                 f'{vic_model_name}_lab_276_scale_1.h5',
						**t_buffer
						)
				acc.pop('Guassian_noise',None)
				for key in list(acc.keys()):
					acc_buf.append(acc[key])
				min_idx = np.asarray(acc_buf)[:,-1].argmin()
				acc_all[perm_name] = np.asarray(acc_buf)[min_idx]
				old_rec.update(acc_all)
				savemat( mat_Path, old_rec )
				savemat(perm_name+'.mat',acc)

	acc_all = loadmat( 'resultsMat\\architecture_compare.mat',squeeze_me=1 )
	acc_all.pop('__header__', None)
	acc_all.pop('__version__', None)
	acc_all.pop('__globals__', None)
	# heatmap( acc_dict = acc_all, vic_model = ['vgg19','resnet','resnet6','resnet10','resnet12'], idx = -1 )
	# victim_model = tf.keras.models.load_model( config.victim_model_Root + f'\\signfi_model_'
	# 					                                                 f'{vic_model_name}_lab_276_scale_1.h5' )
	# loss,acc_t = victim_model.evaluate( config.test_data, config.test_label, batch_size = 32, verbose = 0 )
	# UAP_path ='perturbation\\UAP_signfi_model_resnet_lab_276_scale_1_seed_7.h5'
	# victim_model = tf.keras.models.load_model( config.victim_model_Root + f'\\signfi_model_'
	# 					                                                 f'{vic_model_name}_lab_276_scale_1.h5' )
	# with h5py.File( UAP_path, 'r' ) as f:
	# 	a_group_key = 'universal_perturbation'
	# 	# Get the data
	# 	UAP_data = np.asarray( list( f[ a_group_key ] ) )
	# # scaled_uni_per = scaleDeepfool( psr = psr, test_data = config.test_data, perturbation = UAP_data )
	# adv_data = config.test_data + UAP_data - UAP_data.mean( )
	# _, accpgd_attack = victim_model.evaluate( config.test_data, config.test_label, batch_size = 32, verbose = 0 )



	# vic_model_name = 'resnet'
	# accpgd_attack = compare_DNN_difference(
	# 		victim_model_path = config.victim_model_Root + f'\\signfi_model_'
	# 		                                               f'{vic_model_name}_lab_276_scale_1.h5',
	# 		p1 ='UAP_signfi_model_resnet_lab_276_scale_1_seed_7.h5'
	# 		)


	'''Average performance over 10 UAP for Widar, wiar, signfi'''
	# a = loadmat( 'acc_mean_task_signfi.mat',squeeze_me=1 )
	# plotting_bar_chart(copy.deepcopy(a),psr_idx = 3)
	# '\\widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5'


	# config.data_dir = [
	#                     config.sensingDataset_Root + 'Widar\\' + '20181109',
	#                     config.sensingDataset_Root + 'Widar\\' + '20181115'
	#                     # config.sensingDataset_Root + 'Widar\\' + '20181118'
	#                     ]
	# config.location = [ 2,]
	# config.orientation = [ 1,2,3,4,5,6 ]
	# config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
	# 		config, 'widar', ifscale = True
	# 		)
	# acc_all = compare_DNN_difference(
	# 		victim_model_path = config.victim_model_Root +
	# 		                    '\\widar_model_loc2_ori123456_scale_1_user_2_envir_1_20181109_20181115.h5',
	# 		use_Guas = 1,
	# 		p = 'UAP_surrogateModel_widar_model_None_loc2_ori1_Rx1_scale_1_user_2_envir_1.h5',
	# 		p7 = 'UAP_widar_model_defult_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	# 		p1 = 'UAP_signfi_model_defult_home_276_scale_1.h5'
	# 		)


	# acc_all = compare_DNN_difference(
	# 	p1 = 'UAP_signfi_model_defult_home_276_scale_1.h5',
	# 	# p2 = 'UAP_signfi_model_alex1_home_276_scale_1.h5',
	# 	# p3 = 'UAP_signfi_model_alex2_home_276_scale_1.h5',
	# 	# p4 = 'UAP_signfi_model_alex3_home_276_scale_1.h5',
	# 	# p5 = 'UAP_signfi_model_cnn_home_276_scale_1.h5',
	# 	# p6 = 'UAP_signfi_model_cnnlstm_home_276_scale_1.h5',
	#
	# 	p7 = 'UAP_widar_model_defult_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	# 	p8='UAP_widar_model_alex1_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	# 	p9='UAP_widar_model_alex2_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	# 	p10='UAP_widar_model_alex3_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	# 	p11='UAP_widar_model_cnn_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	# 	p12='UAP_widar_model_cnnlstm_loc123456_ori2_Rx123456_scale_1_user_2_envir_1.h5',
	#
	# 	p13='UAP_wiar_model_defult_scale_1numuser_10.h5',
	# 	p14='UAP_wiar_model_alex1_scale_1numuser_10.h5',
	# 	p15='UAP_wiar_model_alex2_scale_1numuser_10.h5',
	# 	p16='UAP_wiar_model_alex3_scale_1numuser_10.h5',
	# 	p17='UAP_wiar_model_cnn_scale_1numuser_10.h5',
	# 	p18='UAP_wiar_model_cnnlstm_scale_1numuser_10.h5',
	# )
	# results_path = 'utils/resultsMat/guassian.mat'
	# results_path = 'utils/resultsMat/cross_model_test.mat'
	# acc_atk = loadmat(results_path, squeeze_me = 1)
	#
	# plotting(np.arange( 0.000, 0.105, 0.005 ), acc_all = copy.deepcopy(all_data))
	# psr_range = np.arange( 0.00, 0.007, 0.0005 )





