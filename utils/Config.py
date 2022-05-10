import numpy as np
import tensorflow as tf

class getconfig:
    def __init__(self,):
        self.epoch =  1000
        self.data_dir = None
        self.lr = 1e-4
        self.batch_size = 32
        self.N_classes = None
        self.input_shape = [200,60,3]
        self.D_range = 1
        self.pert_Mat_Root = 'perturbation'
        self.set_seed = None
        self.attacker_model_Root = 'SavedModel\\Impact_of_model_structure'
        self.victim_model_Root = 'SavedModel\\victim_model'
        # self.attacker_model_Root = 'SavedModel\\PSR'
        self.results_dir = 'utils\\resultsMat'
        self.sensingDataset_Root = 'E:\\SensingDataset\\'
        self.pretrained_model_path = None
        self.test_data = None
        self.test_label = None
        self.train_data = None
        self.train_label = None
        self.DNN_name = None

        self.source = 'home_276'

        self.orientation = [2]
        self.location = [1,2,3,4,5,6]
        self.receiver = ['r1', 'r2', 'r3','r4','r5', 'r6']

        self.n_iter = None
        # if if_Restore_Samp_idx:
        #     self.getSampleIdx()


