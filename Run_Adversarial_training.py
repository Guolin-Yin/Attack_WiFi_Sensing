#%%
from Adversarial_training import get_adv_data,train_loop,test_loop
from ATKMethods import *
from Run_UAP_Test import save_UAP,genereate_UAP,UAPTest
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split
import numpy as np
import utils.gestureDataLoader as gestureDataLoader
import tensorflow as tf
import os
import utils.Config as Config
import random
#%%
# random.seed(42)
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
#%% Regular training the surrogate model
# print('surrogate model training')
# config.DNN_name = 'defult'
# config.epoch = 1000
# config.lr = 1e-4
# method = None
# model_name = 'Surrogate_model_home.h5'
# config.model_path['adv_robust_model_path'] = os.path.join( 'SavedModel/Adversarial_robust_model' , model_name)
# net = AlexNetTF( config )
# model = net.buildModel( choice = config.DNN_name)
# model = train_loop(config,model,train_dataset,val_dataset,None,None)
# print(f"The accuracy of model {config.model_path['adv_robust_model_path']} is",f'{test_loop(None,None,model,test_dataset,None).numpy()*100}%',sep= '\t')

#%% Generate UAP
print('UAP generation')
# data          = copy.deepcopy( np.concatenate( (X_train,X_test), axis = 0 ) )
# test_label    = copy.deepcopy( np.concatenate( (y_train,y_test), axis = 0 ) )
UAP_path_dic = {}
for seed in [2, 3, 4, 5, 6, 7, 8, 9, 10, 42]:
    np.random.seed( seed )
    # per_idx = np.random.permutation( data.shape[ 0 ] )
    # for model_path in os.listdir('SavedModel/Adversarial_robust_model'):
    model_name = 'Surrogate_model_lab.h5'
    config.model_path['adv_robust_model_path'] = os.path.join( 'SavedModel/Adversarial_robust_model' , model_name)
    UAP_save_path       = 'perturbation/UAP_AT_model/' + 'UAP_' + f'Seed_{seed}_' + config.model_path['adv_robust_model_path'].split( '/' )[ -1 ]
    

    
    if os.path.exists(UAP_save_path):
        print(f'The UAP existance? ',UAP_save_path,os.path.exists(UAP_save_path))
        # os.remove(UAP_save_path)
    
    # net = AlexNetTF( config )
    # model = net.buildModel( choice = 'defult')
    # model.load_weights(config.model_path['adv_robust_model_path'])
    # UAP_data     = genereate_UAP( dataset = data[per_idx], model = model, config = config )
    UAP_path_dic[f'Norm_{seed}_path'] = UAP_save_path
    # save_UAP(UAP_save_path,UAP_data)

#%% Test the UAP
print('UAP testing')
config = Config.getconfig( )
config.source = 'home_276'
train_data, test_data, train_label, test_label = gestureDataLoader.getData(
        config, 'signfi'
        )
X_train, X_test, y_train, y_test = train_test_split( train_data, train_label, test_size=0.1, random_state=42)
UAP_results_folder = 'resultsMat/Adversarial_training_results'
UAP_results_fileName = 'UAP_Cross_domain_results_normal_surro_train_lab_to_home.mat'
UAP_perturbation_folder = 'perturbation/UAP_AT_model'
Victim_model_folder = 'SavedModel/Adversarial_robust_model/'
path_to_results = os.path.join( UAP_results_folder, UAP_results_fileName )
if os.path.exists(path_to_results):
    result_dic = loadmat(path_to_results)
else:
    result_dic = {}
for v_model in os.listdir(Victim_model_folder):
    if 'home' in v_model and 'pgd' in v_model and 'resnet' not in v_model:
            v_psr = v_model.split('_psr_')[1].split('_')[0]
            v_iter = v_model.split('_niter_')[1].split('_')[0]
            name = f'Surrogate_Normal_lab_atk_home_PSR_{v_psr}_iter_{v_iter}'
            if name in result_dic.keys():
                continue
            print('Testing the performance of UAP: ',name)
            
            # net = AlexNetTF( config )
            # model = net.buildModel( choice = 'defult')
            # model.load_weights(Victim_model_folder + v_model)
            # print(f'The victim model is {v_model}')
            # acc = UAPTest(
            #         X = X_test,
            #         y = y_test,
            #         victim_model = model,
            #         psr_range = np.linspace(0,2e-2,21),
            #         **UAP_path_dic
            #         )
            # result_dic.update( {
            #     name:acc
            # } )
            # savemat(path_to_results,result_dic)

#%% Adversarial training the surrogate model
config = Config.getconfig( )
config.source = 'home_276'
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
for psr in [3e-3]:
    for n_iter in [4,8,]:
        config.DNN_name = 'defult'
        config.epoch = 1000
        method = 'pgd'
        model_name = f'robust_adv_training_{method}_psr_{psr}_{config.DNN_name}_{config.source}_niter_{n_iter}_new.h5'
        config.model_path['adv_robust_model_path'] = os.path.join( 'SavedModel/Adversarial_robust_model' , model_name)
        net = AlexNetTF( config )
        model = net.buildModel( choice = config.DNN_name)
        if os.path.exists(config.model_path['adv_robust_model_path']):
            model.load_weights(config.model_path['adv_robust_model_path'])
            print(f'The model existance? ',os.path.exists(config.model_path['adv_robust_model_path']))
            print('load the model',config.model_path['adv_robust_model_path'])
        else:
            print('Model not exist, create new model: \n',config.model_path['adv_robust_model_path'])
        config.lr = 1e-4
        model = train_loop(config,model,train_dataset,val_dataset,psr,method,n_iter = n_iter)
        print(f"The accuracy of model {config.model_path['adv_robust_model_path']} is",f'{test_loop(None,None,model,test_dataset,None).numpy()*100}%',sep= '\t')
        
# %%
