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
from Universal_pert import universal_perturbation
from matplotlib.lines import Line2D
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
	noise = np.random.normal( 0, np.sqrt(psr*test_data.var( )), size = (test_data.__len__(),200, 60, 3) )
	noisy_data = test_data+noise
	_, acc = Attack_model.evaluate( noisy_data, test_label, verbose = 0 )
	return acc
'''Experiments'''
def compare_deepFool_and_FGSM_UAP():
	config = Config.getconfig( )

	results_path = os.path.join( config.results_dir, 'compare_deepFool_and_FGSM.mat' )
	if os.path.exists(results_path):
	# if False:
		psr_all = loadmat(results_path,squeeze_me = 1)['psr']
		fgsm_all = loadmat( results_path,squeeze_me = 1 )[ 'fgsm_acc' ]
		awgn_all = loadmat( results_path,squeeze_me = 1 )[ 'awgn_acc' ]
		df_all = loadmat( results_path, squeeze_me = 1)[ 'deepfool' ]
		UAP_all = loadmat( results_path, squeeze_me = 1 )[ 'UAP' ]
	else:
		uni_per = loadmat( os.path.join(config.pert_Mat_Root,'signfi_lab_276_universal_perturbation_0.02.mat') )[ 'universal_perturbation' ]
		deepfool_per = loadmat( os.path.join(config.pert_Mat_Root,'signfi_lab_276_deepfool.mat'))[ 'perturbation' ]
		# uni_per_signfi = loadmat( os.path.join( config.pert_Mat_Root, 'signfi_lab_276_universal_perturbation.mat' ) )['universal_perturbation' ]
		'''Preparing dataset and model'''
		# model_name = 'widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118'+'.h5'
		model_name = 'signfi_model_lab_276_scale_50'+'.h5'
		Attack_model = tf.keras.models.load_model(
				os.path.join(
						config.attack_model_Root,
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
			print(f'The PSR is: {psr:.6f}, '
				  f'\n deepfool: {acc_df:.4f}, '
				  f'\n UAP: {acc_Uni:.4f} '
				  f'\n FGSM: {acc_fgsm:.4f} '
				  f'\n Guassian noise: {acc_awgn:.4f}',)
			print('--------------------------------')
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
	if os.path.exists( results_path ):
		psr_range = np.arange( 0.00, 0.055, 0.002 )
		for i in range( name.__len__( ) ):
			acc_all[name[i]] = loadmat( results_path , squeeze_me = 1 )[name[i]]
	else:
		'''Prepare data'''
		config.D_range = 1
		procOBJ = gestureDataLoader.preprocessing( )
		config.train_data, config.test_data, config.train_label, config.test_label = gestureDataLoader.getData(
				config, 'widar'
				)
		test_data = copy.deepcopy( procOBJ.scale( config.test_data, config.D_range ) )
		train_data = copy.deepcopy( procOBJ.scale( config.train_data, config.D_range ) )
		test_label = copy.deepcopy( config.test_label )
		train_label = copy.deepcopy( config.train_label )
		config.attack_model_Root = 'SavedModel\\Attack_target_model'
		# runTrain( config = config, dataset_name = 'widar' )
		Attack_model = tf.keras.models.load_model(
				os.path.join(
						config.attack_model_Root,
						'widar_model_loc2_ori123456_scale_1_user_2_envir_2_20181118.h5'
						)
				)
		# f = Model(Attack_model.input,Attack_model.get_layer('FC_2').output)
		# uni_per_widar_in_domain = universal_perturbation(dataset = config.test_data,f = f,overshoot=0.002)
		'''Cross domain universal perturbation testing'''
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


		for i, perturb in enumerate( [ uni_per_widar, uni_per_signfi, ] ):
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

# compare_deepFool_and_FGSM_UAP( )
cross_domain_UAP_test_widar()