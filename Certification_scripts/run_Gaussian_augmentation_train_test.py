#%%
import sys
import os
sys.path.append(os.getcwd())
from Adversarial_training import get_adv_data,test_loop
from ATKMethods import *
from Run_UAP_Test import save_UAP,genereate_UAP,UAPTest
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split
import numpy as np
import utils.gestureDataLoader as gestureDataLoader
import tensorflow as tf
import os, random,time
import utils.Config as Config
from Adversarial_training import train_epoch
from core_fn.test import whitebox_smoothed_testing_loop

config = Config.getconfig( )
config.source = 'lab_276'
tf.config.experimental_run_functions_eagerly(True)
assert config.source != None, 'source should not be None'
train_data, test_data, train_label, test_label = gestureDataLoader.getData(
		config, 'signfi'
		)
##
X_train, X_test, y_train, y_test = train_test_split( train_data, train_label, test_size=0.1, random_state=42)
batch_size = config.batch_size
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
test_dataset = test_dataset.batch(batch_size)

mode = 'train'
# %% Training with Noise
if mode == 'train':
	from core_fn.train import train_loop

	std = 1
	# for psr in [5e-3]:
	for psr in [0.25,0.3,0.35,0.4,0.45]:
     
			config.DNN_name = 'defult'
			config.epoch = 1000 
			config.lr = 1e-4
			
			method = 'noise'
			model_name = f'robust_adv_training_{method}_psr_{psr}_std_{std}_{config.DNN_name}_{config.source}.h5'
			print(model_name)
			config.model_path['adv_robust_model_path'] = os.path.join( 'SavedModel/Adversarial_robust_model/random_smooth', model_name)
			net = AlexNetTF( config )
			model = net.buildModel( choice = config.DNN_name)
			if os.path.exists(config.model_path['adv_robust_model_path']):
					model.load_weights(config.model_path['adv_robust_model_path'])
					print(f'The model existance? ',os.path.exists(config.model_path['adv_robust_model_path']))
					print('load the model',config.model_path['adv_robust_model_path'])
			model = train_loop(config,model,train_dataset,val_dataset,psr,method,std = std,k = 1)
					
			test_atk_method = "fgsm"
			print(f"The accuracy of model {config.model_path['adv_robust_model_path']} is",f'{test_loop(config,psr,model,test_dataset,test_atk_method).numpy()*100}%',sep= '\t')
			test_atk_method = "pgd"
			print(f"The accuracy of model {config.model_path['adv_robust_model_path']} is",f'{test_loop(config,psr,model,test_dataset,test_atk_method,n_iter = 3).numpy()*100}%',sep= '\t')
			test_atk_method = "noise"
			print(f"The accuracy of model {config.model_path['adv_robust_model_path']} is",f'{test_loop(config,psr,model,test_dataset,test_atk_method).numpy()*100}%',sep= '\t')
   
			# model.load_weights(config.model_path['adv_robust_model_path'])
			# acc = whitebox_smoothed_testing_loop(model,dataset = val_dataset,attack_method = method,test_model_name = model_name.replace('.h5',''),sigma = 0.05)
			# print('The attack accuracies: ',acc)
# %% Testing
if mode == 'test':
	from core_fn.test import whitebox_smoothed_testing_loop
	# method = 'pgd'
	std = 1
	net = AlexNetTF( config )
	model = net.buildModel( choice = 'defult')
	for method in ['pgd']:
		# for psr in [0.01e-1,0.03e-1,0.05e-1,2e-1,1.0,1.25,1.5]:
		for psr in [0.5]:
			model_name = f'robust_adv_training_noise_psr_{psr}_std_{std}_defult_{config.source}.h5'
			model_path = os.path.join( 'SavedModel/Adversarial_robust_model/random_smooth', model_name)
			print("Testing model: ",model_path)
			model.load_weights(model_path)
			val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
			val_dataset = val_dataset.batch(1)
			acc = whitebox_smoothed_testing_loop(model,dataset = val_dataset,attack_method = method,test_model_name = model_name.replace('.h5',''),
                                        sigma = 0.2, # testing std
                                        pred_N = 1000,
                                        n_iter = 10
                                        )
			print(acc)