import os
import random
import re
import copy
import scipy.io as sio
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy import stats
import time
class preprocessing:
    def __init__( self ):
        pass
    def norm( self, data ):
        # interval_min = -1
        # interval_max = 1
        # data = (data - np.min( data )) / (np.max( data ) - np.min( data )) * (
        #         interval_max - interval_min) + interval_min
        shape_of_data = data.shape
        data = data.reshape(-1)
        data = (data - np.mean( data))/np.std(data)
        data = data.reshape(shape_of_data)
        # data = stats.zscore(data, axis=1)
        return data
    def scale( self, data,D_range):
        # buf = []
        n = data.shape[0]
        re_data = data.reshape(n,-1)
        scale_data = (re_data - np.min(re_data,axis=1,keepdims = True)) / ( np.max(re_data,axis=1,keepdims = True) -
                                                                           np.min(re_data,axis=1,keepdims = True) )
        scale_data = scale_data.reshape(data.shape)
        # for i in range(len(data)):
        #     buf.append((data[i] - np.min(data[i]))/(np.max(data[i]) - np.min(data[i])))
        return scale_data * D_range
class signDataLoader:
    ''':returns
        filename: [0] home-276 -> user 5, 2760 samples,csid_home and csiu_home
        filename: [1] lab-150 -> user 1 to 5, 1500 samples/user
        filename: [2] lab-276 -> user 5, 5520 samples,downlink*
        filename: [3] lab-276 -> user 5, 5520 samples,uplink*
    '''
    def __init__( self, config = None ):
        self.config = config
        # self.dataDir = config.data_dir
    def loadData( self, ):
        source = self.config.source
        def _reformat( ori_data ):
            reformatData = np.zeros(
                    (ori_data.shape[ 3 ], ori_data.shape[ 0 ], ori_data.shape[ 1 ], ori_data.shape[ 2 ]),
                    dtype='complex_' )
            for i in range( ori_data.shape[ -1 ] ):
                reformatData[ i, :, :, : ] = ori_data[ :, :, :, i ]
            return reformatData
        print("Loading data................")
        data = [ ]
        label = [ ]
        self.filename = os.listdir( self.config.data_dir )
        if source == 'home':
            name_of_loading = 'dataset_home_276.mat'
        elif source == 'lab_276':
            name_of_loading = 'dataset_lab_276_dl.mat'
        elif source == 'lab_150':
            name_of_loading = 'dataset_lab_150.mat'
        # for name in self.filename:
        path = os.path.join(self.config.data_dir,name_of_loading)
        buf = sio.loadmat(path)
        buf.pop( '__header__', None )
        buf.pop('__version__',None)
        buf.pop( '__globals__', None )
        for i in range(buf.__len__()):
            key = list( buf.keys( ) )[ i ]
            if 'csid' in key:
                data =  _reformat(buf[key])
            elif 'label' in key:
                label =buf[key]
            elif 'csi' in key and 'csid' not in key and 'csiu' not in key:
                data.append(_reformat(buf[key]))
                if i == buf.__len__() - 1:
                    data = np.concatenate( data, axis=0 )
        # data = np.asarray( data, axis=0 )
        x_amp = np.abs( data )
        x_phase = np.angle( data )
        data = np.concatenate( (x_amp, x_phase), axis=2 )
        self.config.N_classes = int( np.max( label ) )
        self.config.input_shape = data.shape[1:]
        label = to_categorical( label - 1, num_classes=self.config.N_classes )
        X_train, X_test, y_train, y_test = train_test_split( data, label, test_size=0.2, random_state=42 )
        return [X_train, X_test, y_train, y_test]
class widarDataLoader:
    def __init__(self,config):
        self.config =config
    def loadData( self, motion_sel, ):
        data = [ ]
        label = [ ]
        start = time.time()
        for path in self.config.data_dir:
            for data_root, data_dirs, data_files in os.walk( path ):
                print( f'loading directory {data_root} \n locations:{self.config.location} orientations: {self.config.orientation} receivers: {self.config.receiver}' )
                for data_file_name in data_files:
                    file_path = os.path.join( data_root, data_file_name )
                    try:
                        data_amp = sio.loadmat( file_path )['csiAmplitude']
                        data_phase = sio.loadmat( file_path )[ 'csiPhase' ]
                        data_buf = np.concatenate((data_amp, data_phase),axis=1 )
                        label_buf = int( data_file_name.split( '-' )[ 1 ] )
                        location = int( data_file_name.split( '-' )[ 2 ] )
                        orientation = int( data_file_name.split( '-' )[ 3 ] )
                        repetition = int( data_file_name.split( '-' )[ 4 ] )
                        receiver =  data_file_name.split( '-' )[ 5 ].split('.')[0]
                        # Select Motion
                        if (label_buf not in motion_sel):
                            continue
                        if location not in self.config.location:
                            continue
                        if orientation not in self.config.orientation:
                            continue
                        if receiver not in self.config.receiver:
                            continue
                    except Exception:
                        continue
                    # Save to List
                    data.append( np.expand_dims(data_buf,axis=0) )
                    label.append( label_buf )
        label = np.expand_dims( np.array( label ), axis = 1)
        data = np.concatenate(data,axis=0)

        self.config.N_classes = int( np.max( label ) )
        self.config.input_shape = data.shape[1:]
        label = to_categorical( label - 1, num_classes=self.config.N_classes )
        X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42, test_size=0.2)
        end = time.time( )
        print(f'Time cost: {end-start:.2f}')
        return X_train, X_test, y_train, y_test
class BVPDataLoader:
    def __init__(self,config):
        self.T_MAX = 0
        self.config = config

    def normalize_data( self, data_1 ):
        # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
        data_1_max = np.concatenate( (data_1.max( axis = 0 ), data_1.max( axis = 1 )), axis = 0 ).max( axis = 0 )
        data_1_min = np.concatenate( (data_1.min( axis = 0 ), data_1.min( axis = 1 )), axis = 0 ).min( axis = 0 )
        if (len( np.where( (data_1_max - data_1_min) == 0 )[ 0 ] ) > 0):
            return data_1
        data_1_max_rep = np.tile( data_1_max, (data_1.shape[ 0 ], data_1.shape[ 1 ], 1) )
        data_1_min_rep = np.tile( data_1_min, (data_1.shape[ 0 ], data_1.shape[ 1 ], 1) )
        data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
        return data_1_norm
    def zero_padding( self, data, T_MAX ):
        # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
        data_pad = [ ]
        for i in range( len( data ) ):
            t = np.array( data[ i ] ).shape[ 2 ]
            data_pad.append(
                    np.pad( data[ i ], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values = 0 ).tolist( )
                    )
        return np.array( data_pad )
    def loadData( self,  motion_sel = [1,2,3,4,5,6], l = [ 1, 2, 3, 4, 5, ], o = [ 1, 2, 3, 4, 5 ] ):
        T_MAX = self.T_MAX
        data = [ ]
        label = [ ]
        for data_root, data_dirs, data_files in os.walk( self.config.data_dir ):
            for data_file_name in data_files:
                file_path = os.path.join( data_root, data_file_name )
                try:
                    data_1 = sio.loadmat( file_path )[ 'velocity_spectrum_ro' ]
                    label_1 = int( data_file_name.split( '-' )[ 2 ] )
                    location = int( data_file_name.split( '-' )[ 3 ] )
                    orientation = int( data_file_name.split( '-' )[ 4 ] )
                    repetition = int( data_file_name.split( '-' )[ 5 ] )

                    # Select Motion
                    if (label_1 not in motion_sel):
                        continue
                    # if location not in l:
                    #     continue
                    # if orientation not in o:
                    #     continue
                    # Select Location
                    # if (location not in [1,2,3,5]):
                    #     continue
                    # Select Orientation
                    # if (orientation not in [1,2,4,5]):
                    #     continue
                    # Normalization
                    data_normed_1 = self.normalize_data( data_1 )
                    # Update T_MAX
                    if np.array( data_1 ).shape[ 2 ] > 28:
                        continue
                    if T_MAX < np.array( data_1 ).shape[ 2 ]:
                        T_MAX = np.array( data_1 ).shape[ 2 ]
                except Exception:
                    continue
                # Save List
                data.append( data_normed_1.tolist( ) )
                label.append( label_1 )
        # Zero-padding
        data = self.zero_padding( data, T_MAX )
        # Swap axes
        data = np.swapaxes( np.swapaxes( data, 1, 3 ), 2, 3 )  # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
        data = np.expand_dims( data, axis = -1 )  # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1]
        # Convert label to ndarray
        label = np.array( label )
        self.config.N_classes = int( np.max( label ) )
        label = to_categorical( label - 1, num_classes = self.config.N_classes )
        [ data_train, data_test, label_train, label_test ] = train_test_split(
                data, label, test_size = 0.2, random_state = 42
                )
        # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
        return [data_train, data_test, label_train, label_test]
def getData(config, dataset_name:str,ifzscore:bool=False,ifscale:bool=False):
    procObj = preprocessing()
    if dataset_name == 'widar':
        config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109\\User1',
                            'E:\\SensingDataset\\Widar\\20181115\\User1' ]
        config.receiver = [
                'r1',
                'r2',
                'r3',
                'r4', 'r5', 'r6'
                ]
        config.location = [ 2 ]
        config.orientation = [ 2 ]
        load_widar_obj = widarDataLoader( config )
        X_train, X_test, y_train, y_test = load_widar_obj.loadData(
                motion_sel = [ 1, 2, 3, 4, 5, 6 ],
                # l = [ 2 ], o = [ 2 ], r = [ 'r1', 'r2', 'r3', 'r4', 'r5', 'r6' ]
                )
        config.pretrained_model_path = dataset_name + '_model_' + f'loc{config.location}_' + f'ori{config.orientation}' + f'Rx{config.receiver}'
    elif dataset_name =='signfi':
        # config.data_dir = '/Users/guolinyin/Library/Mobile Documents/com~apple~CloudDocs/PhD Research Files/Dataset/SignFi/Dataset'
        config.data_dir = 'E:\SensingDataset\SignFi\Dataset'
        config.source = 'lab_276'
        signFiLoader = signDataLoader( config=config )
        X_train, X_test, y_train, y_test = signFiLoader.loadData( )
        config.pretrained_model_path = 'SavedModel\\' + dataset_name + '_model_' + f'{config.source}'+f'_scale' \
                                                                                                   f'_' \
                                                                                                      f'' \
                                                                                                      f'' \
                                                                                                      f'' \
                                                                                                      f'{config.D_range}' +'.h5'
    elif dataset_name == 'BVP':
        '''test BVP loader'''
        config.data_dir = 'E:\SensingDataset\BVP_attack'
        BVP_Obj = BVPDataLoader( config )
        X_train, X_test, y_train, y_test = BVP_Obj.loadData( )
    if ifzscore:
        return [procObj.norm(X_train), procObj.norm(X_test), y_train, y_test]
    elif ifscale:
        return [procObj.scale(X_train,D_range = config.D_range), procObj.scale(X_test,D_range = config.D_range), y_train,
                y_test]
    else:
        return [ X_train ,  X_test , y_train, y_test ]
if __name__ == '__main__':
    import sys,os
    current_dir = os.getcwd( )
    sys.path.append( current_dir )
    sys.path.append( current_dir + '/utils' )
    import Config
    config = Config.getconfig()
    '''test widar loader'''
    config.data_dir = ['E:\\SensingDataset\\Widar\\20181109\\User1',
                       'E:\\SensingDataset\\Widar\\20181115\\User1' ]
    load_widar_obj =widarDataLoader(config)
    X_train, X_test, y_train, y_test = load_widar_obj.loadData(motion_sel = [1,2,3,4,5,6],)
    '''test BVP loader'''
    # config.data_dir = 'E:\SensingDataset\BVP_attack'
    # BVP_Obj = BVPDataLoader(config)
    # data_train, data_test, label_train, label_test = BVP_Obj.loadData()