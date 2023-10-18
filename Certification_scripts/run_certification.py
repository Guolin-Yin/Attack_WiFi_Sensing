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
from core_fn.attacks import *
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
from tqdm import tqdm
radius_dict = {}
# for psr in [0.01e-1,0.03e-1,0.05e-1,2e-1,1.0,1.25,1.5]:
for psr in [0.25,0.3,0.35,0.4,0.45]:
    key = f'psr_{psr}_std_{std}'
    if key not in radius_dict:
        radius_dict[key] = []
    model_name = f'robust_adv_training_noise_psr_{psr}_std_{std}_defult_lab_276.h5'
    model_path = os.path.join( 'SavedModel/Adversarial_robust_model/random_smooth', model_name)
    
    results_path = f'certification_{model_name.replace("h5","")}.xlsx'
    if os.path.exists(results_path):
        df = pd.read_excel(results_path)
        tested_index = df['index'].values
    else:
        df = pd.DataFrame(columns=['prediction', 'radius', 'PSR', 'y'])
        tested_index = [-1]
        
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(1)
    
    model.load_weights(model_path)
    smooth = Smooth(model, 276, sigma=0.2)
    print("Testing model: ",model_path)
    for i, (data, label) in enumerate(val_dataset):
        if i in tested_index:
            continue
        # introduce attack
        # delta = atk_fgsm(data,label,model,1e-4)
        # delta = atk_pgd(data, label, model, psr, n_iter=3)
        # data = data + delta
        prediction, radius = smooth.certify(data, n0 = 100, n = 10000, alpha = 0.001, batch_size = 128)
        psr_certified = compute_psr_from_distance(radius, data)
        radius_dict[key].append(radius)
        
        # print and save to excel
        output = f"prediction: {prediction}, radius: {radius:.4f}, PSR: {psr_certified} y: {int(np.argmax(label,axis=-1))}"
        if prediction == -1:
            output = "\033[91m" + output + "\033[0m"
        print(output, flush=True)
        # save to excel
        df = pd.concat([
                        df,
                        pd.DataFrame({'index':i,'prediction': prediction, 'radius': radius, 'PSR': psr_certified, 'y': int(np.argmax(label,axis=-1))}, index=[i])
                        ])
        df.to_excel(results_path, index=False)
    print(f"Average radius for {key}: {np.mean(radius_dict[key])}")
    print('-----------------------------------'*3)