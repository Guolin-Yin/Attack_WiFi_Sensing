#%%
import h5py
import numpy as np
import utils.gestureDataLoader as gestureDataLoader
import copy
from utils.Universal_pert import universal_perturbation
from utils.TOOLS import scaleDeepfool
from ATKMethods import *
from sklearn.model_selection import train_test_split
import utils.Config as Config
from Adversarial_training import test_loop
#import savemat
from scipy.io import savemat,loadmat
def save_UAP(UAP_save_path,UAP_data):
    
    with h5py.File(  UAP_save_path, 'w' ) as hdf:
        hdf.create_dataset( 'universal_perturbation', data = UAP_data )
def genereate_UAP(dataset,model,config):
    '''
    :param dataset: the dataset to loop over
    :param model_path: the attack model path
    :return: the UAP
    '''
    f = tf.keras.Model( model.input, model.layers[ -2 ].output )
    if f.output_shape[ 1 ] != config.N_classes:
        raise Exception(
                f'The output of the feed forward function is wrong, the output should be {config.N_classes}, '
                f'but it is {f.output_shape[ 1 ]}'
                )
    UAP = universal_perturbation( dataset = dataset, f = f, overshoot = 0.002 )
    return UAP
def UAPTest( X, y, victim_model = None, psr_range = None, **UAP_file_names):
    
    if psr_range is None:
        raise Exception( 'psr_range should not be None' )
    acc_all_seed = []
    for name in tqdm(UAP_file_names.keys(),position = 0):
        with h5py.File(UAP_file_names[name],'r') as f:
            UAP_data = np.asarray(list( f[ 'universal_perturbation' ] ))
        acc_all = []
        for psr in psr_range:
            # Perturbation calibration
            scaled_uni_per = scaleDeepfool(psr = psr,x = X, perturbation = UAP_data)
            adv_data = X + scaled_uni_per - scaled_uni_per.mean()
            # Testing
            test_ds = tf.data.Dataset.from_tensor_slices((adv_data, y))
            test_ds = test_ds.shuffle(buffer_size=1024).batch(config.batch_size)
            acc_all.append(test_loop(None,psr,victim_model,test_ds,method = None).numpy())
        acc_all_seed.append(acc_all)
    return acc_all_seed

#%% Generate UAP
# data          = copy.deepcopy( np.concatenate( (X_train,X_test), axis = 0 ) )
# test_label    = copy.deepcopy( np.concatenate( (y_train,y_test), axis = 0 ) )
# for seed in [2, 3, 4, 5, 6, 7, 8, 9, 10, 42]:
#     np.random.seed( seed )
#     per_idx = np.random.permutation( data.shape[ 0 ] )
#     for model_path in os.listdir('SavedModel/Adversarial_robust_model'):
#         config.model_path['adv_robust_model_path'] = 'SavedModel/Adversarial_robust_model/' + model_path
#         UAP_save_path       = 'perturbation/UAP_AT_model/' + 'UAP_' + f'Seed_{seed}_' + config.model_path['adv_robust_model_path'].split( '/' )[ -1 ]
        
#         if '.h5' not in model_path or 'resnet' in model_path:
#             continue
#         if os.path.exists(UAP_save_path):
#             print(f'The UAP existance? ',UAP_save_path,os.path.exists(UAP_save_path))
#             continue
#         if 'home' in config.model_path['adv_robust_model_path']:
#             net = AlexNetTF( config )
#             model = net.buildModel( choice = 'defult')
#             model.load_weights(config.model_path['adv_robust_model_path'])
            
#             UAP_data     = genereate_UAP( dataset = data[per_idx], model = model, config = config )
#             save_UAP(UAP_save_path,UAP_data)
if __name__ == '__main__':
    config = Config.getconfig( )
    config.source = 'lab_276'
    train_data, test_data, train_label, test_label = gestureDataLoader.getData(
            config, 'signfi'
            )
    X_train, X_test, y_train, y_test = train_test_split( train_data, train_label, test_size=0.1, random_state=42)
    #%% Test the UAP
    UAP_results_folder = 'resultsMat/Adversarial_training_results'
    UAP_results_fileName = 'UAP_Cross_domain_results.mat'
    UAP_perturbation_folder = 'perturbation/UAP_AT_model'
    Victim_model_folder = 'SavedModel/Adversarial_robust_model/'
    ##############
    all_path = []
    psr_list = []
    iteration_list = []
    path_group = []
    UAP_file_names = {}
    for path in os.listdir('perturbation/UAP_AT_model'):
        if 'home' in path:
            psr_list.append(path.split('_')[8])
            iteration_list.append(path.split('_')[13])
            all_path.append(path)
    psr_list = np.unique(psr_list)
    iteration_list = np.unique(iteration_list)
    for current_psr in psr_list:
        for current_iteration in iteration_list:
            pathBuff = {}
            for path in all_path:
                if current_psr == path.split('_')[8] and current_iteration == path.split('_')[13]:
                    # print(path)
                    seed = path.split('_')[2]
                    pathBuff[f'PSR_{current_psr}_iter_{current_iteration}_seed_{seed}'] = os.path.join( UAP_perturbation_folder, path )
            if len(pathBuff) != 0:
                UAP_file_names[ f'PSR_{current_psr}_iter_{current_iteration}' ] = pathBuff
    ###################
    psr_range = np.linspace(0,2e-2,21)

    path_to_results = os.path.join( UAP_results_folder, UAP_results_fileName )
    if os.path.exists(path_to_results):
        result_dic = loadmat(path_to_results)
    else:
        result_dic = {}
        
    for surrogate_model_name, UAP_path_dic in UAP_file_names.items():
        for v_model in os.listdir(Victim_model_folder):

            if 'lab' in v_model and 'pgd' in v_model and 'resnet' not in v_model:
                
                v_psr = v_model.split('_psr_')[1].split('_')[0]
                v_iter = v_model.split('_niter_')[1].split('_')[0]
                name = f'{surrogate_model_name}_atk_lab_PSR_{v_psr}_iter_{v_iter}'
                if name != 'PSR_0.00075_iter_16_atk_lab_PSR_0.00075_iter_16':
                    continue
                net = AlexNetTF( config )
                model = net.buildModel( choice = 'defult')
                model.load_weights(Victim_model_folder + v_model)
                # if name in result_dic.keys():
                #     print(f'{name} already exist')
                #     continue
                print(name)
                acc = UAPTest(
                        X = X_test,
                        y = y_test,
                        victim_model = model,
                        psr_range = psr_range,
                        **UAP_path_dic
                        )
                result_dic.update( {
                    name:acc
                } )
                # savemat(path_to_results,result_dic)
                print('=====================================================================================================')
                            
    print('Done')

    # %%
