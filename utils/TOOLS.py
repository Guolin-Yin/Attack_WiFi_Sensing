import matplotlib.pyplot as plt
import numpy as np
import h5py
'''Loading tools'''
def load_h5(path, keys, mode = 'r'):
	if mode=='r':
		with h5py.File(path,'w') as hdf:
			hdf.create_dataset('universal_perturbation',data = uni_per_widar_in_domain)
	else:
		with h5py.File( path, "r" ) as f:
			# List all groups
			print( "Keys: %s" % f.keys( ) )
			a_group_key = list( f.keys( ) )[ 0 ]
			# Get the data
			data = list( f[ a_group_key ] )
def saveToPath(func):
	def wrapper(*args, **kwargs):
		value_to_save = func()
		# if :
		# assert type( value_to_save ) == dict, 'Value cannot be a dictionary'
		try:
			with h5py.File(kwargs['path'],'w') as hdf:
				hdf.create_dataset('accuracy',data = value_to_save)
		except Exception as e:
			print(e)
			print('The return value is a dictionary, this value were not been saved')
		return value_to_save
	return wrapper
def logger(func):
	def wrapper(*args,**kwargs):
		accuracy, name = func(*args, **kwargs)
		return accuracy,name
	return wrapper
'''plotting tools'''
def plotting(psr_range = None, acc_all = None):
	'''
	:param psr_range: testing psr range
	:param acc_all: a dictionary type, include all the accuracy values,the key is the victim model type
	'''
	try:

		psr_range = acc_all['psr_range']
		acc_all.pop('psr_range')
		acc_all.pop('__header__', None)
		acc_all.pop('__version__', None)
		acc_all.pop('__globals__', None)
	except:
		print(f'psr_range is not in acc_all')
	plt.figure()
	for idx,key in enumerate(acc_all):
		if 'widar' in key:
			# continue
			mk = 'o'
		elif 'signfi' in key:

			mk = 'x'
		else:
			mk = 's'
		if 'alex1' in key:
			c = 'b'
		elif 'alex2' in key:
			c = 'k'
		elif 'alex3' in key:
			c = 'y'
		elif 'cnn' in key and 'lstm' not in key:
			c = 'm'
		elif 'cnnlstm' in key:
			c = 'lime'
		elif 'Guassian_noise' in key:
			c = 'steelblue'
		else:
			c= 'r'
		try:
			if 'psr_range' not in key:
				plt.plot(psr_range,acc_all[key],marker = mk,color = c,label = key)
				plt.ylabel('Accuracy')
				plt.xlabel('PSR')
				plt.grid(True)
				plt.legend()
				# plt.ylim(0.25, 0.95)
				plt.show()
		except:
			print('error occurred:')
			print(f'The key of the dictionary is{list(acc_all.keys())}')
def plotting_bar_chart(acc_all,psr_idx = 5):
	'''
	:param acc_all: a dictionary type, include all the accuracy values,the key is the victim model type
	'''
	signFi = [ ]
	widar = [ ]
	wiar = []
	awgn = [ ]
	psr = acc_all[ 'psr_range' ][ psr_idx ]
	try:
		acc_all.pop( 'psr_range' )
		acc_all.pop('__header__', None)
		acc_all.pop('__version__', None)
		acc_all.pop('__globals__', None)
	except:
		print('psr_range not in the dictionary......')
	# net_name = ['alex1','alex2','alex3','cnn','cnnlstm','defult','Guassian noise']
	net_name = [ ]
	ori_acc = acc_all[ 'Guassian_noise' ][ 0 ]

	for key in sorted( acc_all.keys( ) ):
		print( key )
		if 'signfi' in key:
			signFi.append( 1 - acc_all[ key ][ psr_idx ] )
		if 'widar' in key:
			widar.append( 1 - acc_all[ key ][ psr_idx ] )
		if 'wiar' in key:
			wiar.append( 1 - acc_all[ key ][ psr_idx ] )
		if 'Guassian_noise' not in key:
			if key.split( '_' )[ 1 ] not in net_name:
				net_name.append( key.split( '_' )[ 1 ] )
	else:
		if 'Guassian_noise' in list(acc_all.keys()):
			awgn.append( 1 - acc_all[ 'Guassian_noise' ][ psr_idx ] )
			net_name.append( 'Guassian' )
	N = max(widar.__len__( ),signFi.__len__( ))
	id = np.arange( N )
	width = 0.25
	plt.figure( )
	plt.bar( id - 0.5*width, signFi, width, label = 'SignFi' )
	plt.bar( id + 0.5*width, widar, width, label = 'Widar' )
	plt.bar( id + 1.5*width, wiar, width, label = 'Wiar' )
	plt.bar( N + width/2, awgn, width, label = 'Guassian noise' )
	plt.ylabel( "Fooling rate" )
	plt.grid( False )
	plt.legend( )

	plt.title( f'PSR = { psr }' )
	id = np.arange( N + 1 )
	# plt.xticks( id + width / 2, ('alex1','alex2','alex3','cnn','cnnlstm','defult','Guassian noise'))
	plt.xticks( id + width / 2, net_name )
