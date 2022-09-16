from ATKMethods import *
import numpy as np
from tensorflow.keras import backend as K

from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from gestureDataLoader import *
# from Experiments import *
from utils.TOOLS import *

for  rx in [  'r1','r2','r3','r6']:
	'''=======================Prepare data======================='''
	config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109',
	                    # 'E:\\SensingDataset\\Widar\\20181115'
	                    ]
	config.D_range = 1
	config.receiver = rx
	config.location = [ 5 ]
	config.orientation = [ 2,3,4 ]
	config.DNN_name = "defult"
	X_train ,  X_test , y_train, y_test  = getData(config, 'widar' )
	data = np.concatenate((X_train,X_test),axis=0)
	label = np.concatenate((y_train,y_test),axis=0)
	# label = label.argmax(axis = 1).max( )
	# n_classes = label.max( )
	'''=======================Find the pseudo-labels for training======================='''
	test_num_clusters = [3, 4, 5,6, 7, 8, 9, 10,]
	for num_clusters in test_num_clusters:
	# num_clusters = label.argmax(axis = 1).max( )
	# num_clusters = label.argmax(axis = 1).max( )
		model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors',
		                           assign_labels='kmeans')
		# kmeans = KMeans(n_clusters=n_classes, random_state=0)
		clusters = model.fit_predict(data.reshape(data.shape[0],-1))
		label_pred = to_categorical( clusters, num_classes =  clusters.max()+1)
		config.N_classes = clusters.max() + 1
		# labels_pred = np.zeros_like( clusters )
		# for i in range( n_classes ):
		#     mask = (clusters == i)
		#     labels_pred[ mask ] = mode( label[ mask ] )[ 0 ]
		# print(accuracy_score( label, labels_pred ) )
		'''=======================Build and train the Nerual Network======================='''
		Net = AlexNetTF(config)
		network = Net.buildModel(choice = config.DNN_name)
		earlyStop = tf.keras.callbacks.EarlyStopping( monitor = 'val_acc', patience = 35, restore_best_weights = True )
		lrScheduler = ReduceLROnPlateau(
				monitor = 'val_loss', factor = 0.1,
				patience = 25,
				)
		optimizer = tf.keras.optimizers.Adamax(
				learning_rate = config.lr, beta_1 = 0.95, beta_2 = 0.99, epsilon = 1e-09,
				name = 'Adamax'
				)
		network.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
		model_name = f'surrogateModel_NClusters_{num_clusters}_'+config.pretrained_model_path.split('\\')[2]

		model_path = os.path.join('SavedModel','Eavesdropping', model_name)
		if not os.path.exists( model_path ):
			history = network.fit(
					data, label_pred,
					validation_split = 0.05,
					batch_size = config.batch_size,
					epochs = 200,
					callbacks = [ earlyStop, lrScheduler ],
					shuffle = True,
					verbose = False,
					)
			network.evaluate( data, label_pred )
			network.save( model_path )
		seed_container = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42 ]
		for seed in seed_container:
			config.set_seed = seed
			per_name = f'UAP_NClusters_{num_clusters}_' + model_name.split( '.' )[ 0 ] + f'_seed_{config.set_seed}' + '.h5'
			# config.pert_Mat_Root = os.path.join(config.pert_Mat_Root,'Eavesdropping')
			path = os.path.join( config.pert_Mat_Root, per_name )
			if not os.path.exists(path):
				print( f'================Generating UAP for receiver: {rx} seed: {seed} N Clusters: {num_clusters}================' )
				X_train , X_test , y_train, y_test  = getData(config, 'widar' )
				data = np.concatenate((X_train,X_test),axis=0)
				label = np.concatenate((y_train,y_test),axis=0)
				config.N_classes = clusters.max( ) + 1
				UAP = genereate_UAP(data,model_path,config)
				with h5py.File( path, 'w' ) as hdf:
				    hdf.create_dataset( 'universal_perturbation', data = UAP )
