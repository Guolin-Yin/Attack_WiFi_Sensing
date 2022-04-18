from DeepNet import *
import numpy as np
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from gestureDataLoader import *
from Experiments import *

'''=======================Prepare data======================='''
config.data_dir = [ 'E:\\SensingDataset\\Widar\\20181109',
                    'E:\\SensingDataset\\Widar\\20181115' ]
config.D_range = 1
config.receiver = ['r1',]
config.location = [ 2 ]
config.orientation = [ 1, ]
X_train ,  X_test , y_train, y_test  = getData(config, 'widar' )
data = np.concatenate((X_train,X_test),axis=0)
label = np.concatenate((y_train,y_test),axis=0)
# label = label.argmax(axis = 1).max( )
# n_classes = label.max( )
'''=======================Find the pseudo-labels for training======================='''
model = SpectralClustering(n_clusters=label.argmax(axis = 1).max( ), affinity='nearest_neighbors',
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
config.DNN_name = "defult"
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
# optimizer = tf.keras.optimizers.SGD(learning_rate=config.lr)
# optimizer = tf.keras.optimizers.Adam(learning_rate = config.lr)
network.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
# Net.summary( )
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
model_name = 'surrogateModel_'+config.pretrained_model_path.split('\\')[2]
model_path = os.path.join(config.attacker_model_Root, model_name)
network.save( model_path )

UAP = genereate_UAP(data,model_path)
per_name = 'UAP_' + model_name
path = os.path.join( config.pert_Mat_Root, per_name )
with h5py.File( path, 'w' ) as hdf:
    hdf.create_dataset( 'universal_perturbation', data = UAP )
