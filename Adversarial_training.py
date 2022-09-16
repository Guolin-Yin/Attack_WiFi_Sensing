from ATKMethods import generatePerturbData,AlexNetTF
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import os
import tensorflow as tf
import time

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
        loss_value += sum(model.losses)
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
    # Iterate over the batches of the dataset.

    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset),desc='Training',total=len(train_dataset)):
        x_batch_train, y_batch_train = get_adv_data(psr,model,x_batch_train, y_batch_train,method = method,config = config,to_tf_dataset = False,**kwargs)
        loss_value = train_step(x_batch_train, y_batch_train,model,optimizer,train_acc_metric,loss_fn)
        t_loss.append(loss_value)

    train_acc = train_acc_metric.result()

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    t_loss_val = []
    # print("Validation")
    for x_batch_val, y_batch_val in val_dataset:
        x_batch_val, y_batch_val = get_adv_data(psr,model,x_batch_val, y_batch_val,method = method,
                                                to_tf_dataset = False,
                                                config=config,**kwargs)
        t_loss_val.append(test_step(x_batch_val, y_batch_val,model,val_acc_metric,loss_fn))

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    
    return val_acc,train_acc,np.mean(t_loss_val),np.mean(t_loss)
def train_loop(config,model,train_ds,val_ds,psr,method,**kwargs):
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adamax(learning_rate=config.lr, beta_1=0.95, beta_2=0.99, epsilon=1e-09,name='Adamax')
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    loss_record_min = np.inf
    if os.path.exists(config.model_path['adv_robust_model_path']):
        print("Loading model from disk")
        model.load_weights(config.model_path['adv_robust_model_path'])
        print("Model loaded")
    for i in range(config.epoch):
        # compute the adversarial examples
        # train_ds = get_adv_data(psr,model,train_ds,method = method,config = config,**kwargs)
        # val_ds = get_adv_data(psr,model,val_ds,method = method,config=config,**kwargs)
        val_acc,train_acc,t_loss_val,t_loss = train_epoch(config,model,train_ds,val_ds,
                                                          optimizer,loss_fn,train_acc_metric,
                                                          val_acc_metric,psr,method,**kwargs)
        if loss_record_min > t_loss_val:
            loss_record_min = t_loss_val
            idx_min = i
            model.save_weights(config.model_path['adv_robust_model_path'])
            # print(f'Epoch {i} : saving model with loss {loss_record_min}')
        if i - idx_min > 10:
            config.lr = config.lr * 0.1
            print(f'Epoch {i} : lr decay to {config.lr}')
        if i - idx_min > 25:
            model.load_weights(config.model_path['adv_robust_model_path'])
            print(f'Early stopping at epoch {i}')
            break
        if i % 10 == 0:
            print(f"Epoch {i}, Validation acc: {val_acc:.4f},Training acc: {train_acc:.4f} Validation loss: {t_loss_val:.4f} Training loss over epoch: {t_loss:.4f}")
        else:
            print(f"Epoch {i}, Validation acc: {val_acc:.4f},Training acc: {train_acc:.4f}", end="\r")
    return model
def test_loop(config,psr,model,test_ds,method,**kwargs):
    if method == 'pgd':
        assert 'n_iter' in kwargs.keys(), 'n_iter must be specified for pgd'
    t_loss_val = []
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # print("Validation")
    for x_batch_val, y_batch_val in test_ds:
        if method:
            x_batch_val, y_batch_val = get_adv_data(psr,model,x_batch_val, y_batch_val,method = method,
                                                    to_tf_dataset = False,
                                                    config=config,**kwargs)
        
        t_loss_val.append(test_step(x_batch_val, y_batch_val,model,val_acc_metric,loss_fn))
    val_acc = val_acc_metric.result()
    return val_acc
def get_adv_data(psr,model,x_batch,y_batch,method = 'fgsm',config = None,to_tf_dataset = True,**kwargs):
    # generate testing adversarial data
    delta = []
    x_all = []
    y_all = []
    # for x_batch,y_batch in dataset:
    for x,y in zip(x_batch,y_batch):
        x_all.append(np.asarray(x))
        y_all.append(np.asarray(y))

    x_all = np.asarray(x_all)
    y_all = np.asarray(y_all)
    
    delta = generatePerturbData(psr                 = psr,
                        data                = x_all,
                        current_label       = y_all,
                        pretrained_model    = model,
                        t_label             = None,
                        method              = method,
                        **kwargs)[1]  
    
    # for x,y in zip(x_all,y_all):
    #     delta.append(generatePerturbData(psr                = psr,
    #                                     data                = np.expand_dims(x,0),
    #                                     current_label       = np.expand_dims(y,0),
    #                                     pretrained_model    = model,
    #                                     t_label             = None,
    #                                     method              = method,
    #                                     **kwargs)[1]                           
    #     )

    x_adv  = x_all + delta
    if to_tf_dataset:
        adv_dataset = tf.data.Dataset.from_tensor_slices((x_adv, y_all))
        adv_dataset = adv_dataset.batch(config.batch_size)
        return adv_dataset
    else:
        return x_adv,y_all