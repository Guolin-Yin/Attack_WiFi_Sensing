# add parent directory to path
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
from core_fn.randomised_smooth import Smooth
from core_fn.utils import compute_psr_from_distance
import pandas as pd
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


net = AlexNetTF( config )
model = net.buildModel( choice = 'defult')
std = 1
psr_boundary = 1e-4 # next: test
whitebox_psr_list = np.linspace(0, psr_boundary, 9)
from tqdm import tqdm
radius_dict = {}
# for psr in [0.01e-1,0.03e-1,0.05e-1,2e-1,1.0,1.25,1.5]:
# method = 'pgd'
for method in ['fgsm','pgd']:
    for psr in [0.5]:
        key = f'psr_{psr}_std_{std}'
        if key not in radius_dict:
            radius_dict[key] = []
        model_name = f'robust_adv_training_noise_psr_{psr}_std_{std}_defult_lab_276.h5'
        model_path = os.path.join( 'SavedModel/Adversarial_robust_model/random_smooth', model_name)
        model.load_weights(model_path)
        smooth = Smooth(model, 276, sigma=0.2)
        print("Testing model: ",model_path)

        try:
            df = pd.read_excel(f'certification_{model_name.replace("h5","")}.xlsx')
        except:
            continue
        # filter the df
        df = df[df['prediction'] != -1] # remove abstain
        # df = df[df['y'] == df['prediction']] # select the correct prediction
        test_index_list = df[df['PSR']>psr_boundary]['index'].values # select the index of the test data with certificated PSR boundary
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test[test_index_list], y_test[test_index_list]))
        val_dataset = val_dataset.batch(1)
        acc = whitebox_smoothed_testing_loop(model,
                                            dataset = val_dataset,
                                            attack_method = method,
                                            psr_list = whitebox_psr_list,
                                            test_model_name = "TTT_certified_" + model_name.replace('.h5','')+ f"psr_boundary_{psr_boundary}",
                                            sigma = 0.2, # testing std
                                            pred_N = 1000,
                                            )
