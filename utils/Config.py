import numpy as np
import tensorflow as tf

class getconfig:
    def __init__(self,):
        self.epoch =  150
        self.data_dir = '/content/drive/MyDrive/Colab Notebooks/SensingDataset/SignFi/Dataset'
        self.lr = 1e-4
        self.batch_size = 32
        self.N_classes = None
        self.input_shape = [200,60,3]
        self.source = None
        self.orientation = None
        self.location = None
        self.receiver = ['r1', 'r2', 'r3','r4','r5', 'r6']
        self.pretrained_model_path = None
        self.test_data = None
        self.test_label = None
        self.train_data = None
        self.train_label = None
        self.D_range = None
        self.pert_Mat_Root = 'utils\\perturbationMatFiles'
        self.attack_model_Root = 'SavedModel\\PSR'
        self.results_dir = 'utils\\resultsMat'
        # if if_Restore_Samp_idx:
        #     self.getSampleIdx()


