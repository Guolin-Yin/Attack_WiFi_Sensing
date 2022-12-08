import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.lines import Line2D
import gestureDataLoader,Config
import tensorflow as tf
from tensorflow.keras import backend as K
import plotSig
current_dir = os.getcwd( )
sys.path.append( 'G:\我的云端硬盘\Colab Notebooks\AdvAttackandDefense' )
import ATKMethods
config = Config.getconfig( )
def visualize_scatter_domain( data_2d, label_ids, perplexity,n_iter,figsize = (10, 10) ):
	plt.figure( figsize = figsize )
	# plt.grid( )
	# nb_classes = len( np.unique( label_ids ) )
	# id_to_label_dict = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(Vertical)','Draw-N(Vertical)']
	# id_to_label_dict = [f'domain_{i + 1}' for i in range(nb_classes)]
	# marker = ['o','v','^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
	# for label_id in np.unique( label_ids ):
		# plt.scatter(
		# 		data_2d[ np.where( label_ids == label_id ), 0 ],
		# 		data_2d[ np.where( label_ids == label_id ), 1 ],
		# 		marker = marker[label_id],
		# 		color = plt.cm.Set1( label_id / float( nb_classes + 1 ) ),
		# 		linewidth = 1,
		# 		alpha = 0.8,
		# 		label = id_to_label_dict[ label_id ]
		# 		)
	domain_labels = []
	for i in range(len(label_ids)):
		domain_labels.append(f'domain_{label_ids[i]+1}')
	tsne_df = pd.DataFrame(
			{
					't-SNE_1'    : data_2d[ :, 0 ],
					't-SNE_2'    : data_2d[ :, 1 ],
					'labels_pred': domain_labels
					}
			)
	sns.scatterplot(
			x = "t-SNE_1", y = "t-SNE_2",
			hue = "labels_pred",
			style = 'labels_pred',
			data = tsne_df
			)
	plt.legend( loc = 'best' )
	plt.title(f'perplexity is {perplexity}, number of iterations is {n_iter}', fontsize=18)
	plt.xlabel("t-SNE_1", fontsize=15)
	plt.ylabel( "t-SNE_2", fontsize = 15 )
def visualize_scatter_classes( data, label_id, label, perplexity,n_iter,figsize = (12,10) ,domain = None,):
	plt.figure( figsize = figsize )
	plt.tick_params( axis = 'x', label1On = False )
	plt.tick_params( axis = 'y', label1On = False )
	# plt.grid( )
	# nb_classes = len( np.unique( label_ids ) )
	# id_to_label_dict = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(Vertical)','Draw-N(Vertical)']
	# id_to_label_dict = list(str(np.unique( label_ids )))
	marker = ['x','v',  '.', 'h', 'H', 'D', 'P', 'o','v','^', '<', '>', '8', 's', 'p', '*',  ]
	color = np.random.choice([1,2,3],3,replace = False )
	color = ['r','b','g']
	domains = ['Attack free','Attacked','XXXX']
	facecolors=['r','none','none','b','none','none','g','none','none']
	# facecolors = [ 'none','r', 'none', 'none','r', 'none', ]
	count = 0
	for i in range(len(data)):
		data_2d = data[i]
		label_ids = label_id[ i ]
		domain = domains[i]
		print( i )
		for idx, id in enumerate(np.unique( label_ids ) ):

			p = np.where( label_ids == id )[0]
			# if id == 5:
			# 	o = 3
			# elif id == 3:
			# 	o = 2
			# elif id == 1:
			# 	o = 1
			plt.scatter(
					data_2d[ p, 0 ],
					data_2d[ p, 1 ],
					marker = marker[idx],
					s = 200,facecolors=facecolors[count],
					# color = plt.cm.Set1( id / float( nb_classes + 1 ) ),
					# color = plt.cm.Set1( color[i] ),
					color=color[i],
					linewidth = 2,
					alpha = 0.7,
					label = 'sign ' + str(id) + f' ({domain})',
					# fillstyle = Line2D.fillStyles[ -1 ]
					)
			count = count+1
	plt.legend( fontsize = 17,loc = 'best' )
	out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
		  f'learning/Results/results_figs/Paperfigure/' + 't_SNE'
	plt.savefig( out + '.pdf', bbox_inches = 'tight' )
def domain_t_sne(data,n_components:int = 2,random_state = 0,perplexity:int = 6,n_iter:int = 5000):
	# data = data.reshape(len(data),-1)
	n = len(data)
	domain_label = []
	data_con = []
	for i in range(n):
		buff = data[ i ].reshape( len( data[ i ] ), -1 )
		data_con.append( buff )
		domain_label.append( [i for _ in range( len( buff ) )] )
	data_con = np.concatenate(data_con,axis = 0)
	domain_label = np.concatenate(domain_label,axis = 0)
	model = TSNE(n_components=n_components, random_state=random_state,perplexity = perplexity,n_iter=n_iter)
	tsne_data = model.fit_transform(data_con)
	visualize_scatter_domain( data_2d = tsne_data, label_ids = domain_label, perplexity = perplexity, n_iter = n_iter )
def class_t_sne(data,label_id,label,n_components:int = 2,random_state = 0,perplexity:int = 6,n_iter:int = 5000):
	data = [d.reshape( len( d ), -1 ) for d in data]
	model = TSNE(n_components=n_components, random_state=random_state,perplexity = perplexity,n_iter=n_iter)
	tsne_data = [model.fit_transform(t_d) for t_d in data]
	visualize_scatter_classes( data = tsne_data,label_id = label_id, label = label, perplexity = perplexity, n_iter = n_iter,
			domain = 'lab' )
if __name__ == '__main__':
	dataset_name = 'widar'
	# config.D_range = 1
	_, data_test, _, label_test = gestureDataLoader.getData( config, dataset_name, ifzscore = True )
	config.pretrained_model_path = 'SavedModel/widar_model_loc[2]_ori[2]Rx123456_zscore'
	pretrained_model = tf.keras.models.load_model( 'G:\\我的云端硬盘\\Colab '
												   'Notebooks\\AdvAttackandDefense\\'+config.pretrained_model_path )
	# advData = DeepNet.generatePerturbData(psr=0.2,data=data,current_label=label,pretrained_model = pretrained_model,
	# 		t_label = None)

	advData = [ ]
	for i, test_data in enumerate( data_test ):
		test_data, current_label = np.expand_dims( test_data, axis = 0 ), np.expand_dims(label_test[ i ], axis = 0)
		advData.append(
				ATKMethods.generatePerturbData(
						psr = 0.01, data = test_data, current_label = current_label, pretrained_model =
						pretrained_model, t_label = None
						)
				)
	advData = np.concatenate( advData, axis = 0 )
	label_test_buf = np.argmax(label_test,axis=1)
	selection = [2,3,4]
	idx = np.where( np.expand_dims(label_test_buf,axis=1) == selection )[ 0 ]
	data = data_test[idx]
	label = label_test_buf[idx]
	adv_Data = advData[idx]
	# data = [data]
	# label_id = [label_home,label_lab,label_user4]
	# label = []
	class_t_sne( data = [data,adv_Data], label_id = [label,label], label = label, perplexity = 7, n_iter = 2000 )
	show_idx = np.where( np.expand_dims( label, axis = 1 ) == 3 )[ 0 ][3]
	plotSig.showSignal(data[show_idx,:,0,0],[adv_Data[show_idx,:,0,0]],[303030])

	import TOOLS
	x = 10
	p = advData[x] - data_test[x]
	d = data_test[x]
	print(TOOLS.PSRCompute(p,d))