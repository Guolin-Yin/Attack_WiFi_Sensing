from ATKMethods import generatePerturbData,gen_adv_data,compute_psr
import numpy as np
import os,time
import tensorflow as tf
from core_fn.attacks import get_adv_data
from copy import deepcopy as dc
@tf.function
def test_step(x, y,model,val_acc_metric,loss_fn):
	val_logits = model(x, training=False)
	val_acc_metric.update_state(y, val_logits)
	return loss_fn(y, val_logits)

@tf.function
def train_step(x, y,model,optimizer,train_acc_metric,loss_fn):
	with tf.GradientTape() as tape:
		logits = model(x, training=True)
		loss_value = loss_fn(y, logits)
		# Add any extra losses created during the forward pass.
		# loss_value += sum(model.losses)
	grads = tape.gradient(loss_value, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	train_acc_metric.update_state(y, logits)
	return loss_value 
def train_epoch(config,
				model,
				train_dataset,
				val_dataset,
				optimizer,
				loss_fn,
				# loss,
				train_acc_metric,
				val_acc_metric,
				psr,
				method,
				**kwargs
				):

	t_loss = []
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):#,desc='Training',total=len(train_dataset):
		# print(f'\rTraining step {step}',end='')
		if method:
			x = x_batch_train
			y = y_batch_train
			for _ in range(kwargs.get('k',1)):
				x_batch_train_adv, y_batch_train_adv = get_adv_data(psr,model,x, y, method = method,batch_size=config.batch_size,to_tf_dataset = False,**kwargs)
				x_batch_train, y_batch_train = np.concatenate((x_batch_train,x_batch_train_adv),axis = 0),np.concatenate((y_batch_train,y_batch_train_adv),axis = 0)
		loss_value = train_step(x_batch_train, y_batch_train,model,optimizer,train_acc_metric,loss_fn)
		t_loss.append(loss_value)

	train_acc = train_acc_metric.result()
	train_acc_metric.reset_states()
	t_loss_val = []
	for x_batch_val, y_batch_val in val_dataset:
		if method:
			x_batch_val, y_batch_val = get_adv_data(psr,model,x_batch_val, y_batch_val,method = method,
													to_tf_dataset = False,
													batch_size=config.batch_size,**kwargs)
		t_loss_val.append(test_step(x_batch_val, y_batch_val,model,val_acc_metric,loss_fn))
	val_acc = val_acc_metric.result()
	val_acc_metric.reset_states()
	return val_acc,train_acc,np.mean(t_loss_val),np.mean(t_loss)
def train_loop(config,model,train_ds,val_ds,psr,method,**kwargs):
	# Instantiate an optimizer to train the model.
	optimizer = tf.keras.optimizers.Adamax(learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,name='Adamax')
	loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	# Prepare the metrics.
	train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
	val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
	loss_record_min = np.inf
	if not os.path.exists(config.model_path['adv_robust_model_path'].split('/')[0]):
		os.makedirs(config.model_path['adv_robust_model_path'].split('/')[0])
	if os.path.exists(config.model_path['adv_robust_model_path']):
		print("Loading model from disk")
		model.load_weights(config.model_path['adv_robust_model_path'])
		print("Model loaded")

	idx_min = 0
	idx_lr = 0
	for i in range(config.epoch):
		print(f"\rEpoch:{i}",end='')
		val_acc,train_acc,t_loss_val,t_loss = train_epoch(config,model,train_ds,val_ds,
														  optimizer,loss_fn,train_acc_metric,
														  val_acc_metric,psr,method,**kwargs)
		
		if loss_record_min > t_loss_val:
			loss_record_min = t_loss_val
			acc_record = val_acc
			idx_min = i
			model.save_weights(config.model_path['adv_robust_model_path'])
			# print(f'                                      \rEpoch {i} : saving model with loss {loss_record_min}, accuracy {acc_record}',end='')
		if i - idx_min > 5:
			idx_lr = i
			config.lr = config.lr * 0.1
			print(f'\nEpoch {i} : lr decay to {config.lr}, {time.asctime(time.localtime())}')
		if i - idx_min > 10:
			model.load_weights(config.model_path['adv_robust_model_path'])
			print(f'\nEarly stopping at epoch {i}, {time.asctime(time.localtime())}')
			break                                                    
		print(f"\rEpoch {i}, Validation acc: {val_acc:.4f},Training acc: {train_acc:.4f}, Validation loss: {t_loss_val:.4f}, Training loss over epoch: {t_loss:.4f}, Model saved at epoch {idx_min}, accuracy {val_acc:.4f}, loss {loss_record_min:.4f}", end='')

	return model
# def train_loop(config,model,train_ds,val_ds,psr,method,**kwargs):
# 	# Instantiate an optimizer to train the model.
# 	optimizer = tf.keras.optimizers.Adamax(learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,name='Adamax')
# 	loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 	# Prepare the metrics.
# 	train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
# 	val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
# 	loss_record_min = np.inf

# 	idx_min = 0
# 	idx_lr = 0
# 	for i in range(config.epoch):
# 		print(f"\rEpoch:{i}",end='')
# 		val_acc,train_acc,t_loss_val,t_loss = train_epoch(config,model,train_ds,val_ds,
# 								optimizer,loss_fn,train_acc_metric,
# 								val_acc_metric,psr,method,**kwargs)
# 		if loss_record_min > t_loss_val:
# 			loss_record_min = t_loss_val
# 			acc_record = val_acc
# 			idx_min = i
# 			model.save_weights(config.model_path['adv_robust_model_path'])
# 		if i - idx_min > 25 and i - idx_lr > 20:
# 			idx_lr = i
# 			config.lr = config.lr * 0.1
# 			print(f'\nEpoch {i} : lr decay to {config.lr}, {time.asctime(time.localtime())}')
# 		if i - idx_min > 25:
# 			model.load_weights(config.model_path['adv_robust_model_path'])
# 			print(f'\nEarly stopping at epoch {i}, {time.asctime(time.localtime())}')
# 			break
# 		if idx_min:                                                          
# 			print(f"\rEpoch {i}, Validation acc: {val_acc:.4f},Training acc: {train_acc:.4f} Validation loss: {t_loss_val:.4f} Training loss over epoch: {t_loss:.4f} Model saved at epoch {idx_min}, accuracy {acc_record:.4f}, loss {loss_record_min:.4f}",end='')
# 		else:
# 			print(f"\rEpoch {i}, Validation acc: {val_acc:.4f},Training acc: {train_acc:.4f} Validation loss: {t_loss_val:.4f} Training loss over epoch: {t_loss:.4f}, {time.asctime(time.localtime())}",end='')

# 	return model