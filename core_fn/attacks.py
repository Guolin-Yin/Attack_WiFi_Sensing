import tensorflow as tf
from core_fn.utils import *
import numpy as np
from scipy.io import savemat, loadmat
import os,copy

from tqdm import tqdm
from tensorflow import keras


@tf.function
def compute_gradient(model_fn, loss_fn, x, y, targeted = None):
    """
    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(y, model_fn(x))
        if (targeted):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad
def atk_fgsm(x,y,model,psr = 0.3,targeted = False,loss_object = tf.keras.losses.categorical_crossentropy):
    """
    Fast Gradient Sign Method (FGSM) attack.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param model: model
    :param eps: epsilon (input variation parameter)
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :param loss_object: loss function that takes (labels, logits) as arguments and returns loss.
    :return: a tensor for the adversarial example
    """
    # Convert to tensors
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)


    # Define gradient of loss wrt input
    grad = compute_gradient(model_fn=model, loss_fn=loss_object, x=x, y=y, targeted=targeted)

    # Take sign of gradient
    delta = tf.sign(grad)

    delta = psr_limiter(psr,delta, x)

    # Add perturbation to original example to obtain adversarial example
    # adv_x = x + delta
    return delta
def atk_pgd(x,y,model,psr = 0.3,targeted = False,loss_object = tf.keras.losses.categorical_crossentropy,n_iter = None):
    """
    Projected Gradient Descent (PGD) attack.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param model: model
    :param eps: epsilon (input variation parameter)
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :param loss_object: loss function that takes (labels, logits) as arguments and returns loss.
    :return: a tensor for the adversarial example
    """
    # Convert to tensors
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    delta = np.zeros_like(x)
    # Define gradient of loss wrt input
    assert n_iter is not None, "n_iter is None"
    for i in range(n_iter):
        grad = compute_gradient(model_fn=model, loss_fn=loss_object, x=x, y=y, targeted=targeted)
        delta = delta + psr_limiter(psr/n_iter,grad,x)
        # if np.any(np.reshape(grad,(grad.shape[0],-1)).sum(axis = 1) == 0):
        #     print((np.reshape(grad,(grad.shape[0],-1)).sum(axis = 1) == 0).sum())
        # else:
        #     print("all grad is not zero")
    delta = psr_limiter(psr,delta, x)

    # Add perturbation to original example to obtain adversarial example
    # adv_x = x + delta
    return delta
def atk_noise(psr,x,std = 1.0):
    shape = x.shape
    delta = np.random.normal(0,std,shape)
    
    return psr_limiter(psr,delta,x)
def deepfool(input_CSI, pretrained_model, overshoot=0.002, max_iter=50,):
    '''
    :param input_CSI: One sample (n,input_shape)
    :param pretrained_model: feedforward function
    '''
    model = pretrained_model
    num_classes = pretrained_model.output_shape[1]
    image_norm = tf.cast(input_CSI, tf.float32 )
    f_image = model(image_norm).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    # print(label, "label")
    input_shape = np.shape(np.array(image_norm))

    pert_image = copy.deepcopy(image_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    x = tf.Variable(pert_image)
    # fs = model(x)
    k_i = label
    def loss_func(logits, I, k):
        # return tf.nn.softmax_cross_entropy_with_logits(labels_pred=labels_pred, logits=logits)
        return logits[0, I[k]]
    while k_i == label and loop_i < max_iter:
        # print(loop_i)
        pert = np.inf
        # one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            # loss_value = loss_func(one_hot_label_0, fs)
            loss_value = loss_func(fs, I, 0)
        # grad_orig = tape.gradient(fs[0, I[0]], x)
        grad_orig = tape.gradient(loss_value, x)
        for k in range(1, num_classes):
            # one_hot_label_k = tf.one_hot(I[k], num_classes)
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                # loss_value = loss_func(one_hot_label_k, fs)
                loss_value = loss_func(fs, I, k)
            # cur_grad = tape.gradient(fs[0, I[k]], x)
            cur_grad = tape.gradient(loss_value, x)
            w_k = cur_grad - grad_orig # type: ignore
            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()
            pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))
            if pert_k < pert:
                pert = pert_k
                w = w_k
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        pert_image = image_norm + (1 + overshoot) * r_tot # type: ignore
        x = tf.Variable(pert_image)
        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())
        loop_i += 1
    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop_i, label, k_i, np.asarray(pert_image)
def gen_adv_data(x,y,model,atk_type = 'fgsm',psr = None,targeted = False,loss_object = tf.keras.losses.categorical_crossentropy,**kwargs):
    assert isinstance(psr,float), "psr should with length 1"
    if psr == 0.0:
        delta = np.zeros(x.shape)
    else:
        if atk_type == 'fgsm':
            delta = atk_fgsm(x,y,model,psr,targeted,loss_object)
        elif atk_type == 'pgd':
            delta = atk_pgd(x,y,model,psr,targeted,loss_object,n_iter = kwargs['n_iter'])
        elif atk_type == 'noise':
            delta = atk_noise(psr,x,std=kwargs['std'])
        elif atk_type == 'deepfool':
            model_path = kwargs['model_path']
            if os.path.exists(f'perturbation/deepfool/{model_path}_df.mat',):
                delta = loadmat(f'perturbation/deepfool/{model_path}_df.mat',squeeze_me=True)['delta']
                # delta = scaleDeepfool(psr,x,delta['delta'])
            else:
                model_df = keras.models.Model( inputs = model.input, outputs = model.layers[ -2 ].output )
                delta = []
                for i in tqdm(range(len(x)),desc = 'generating deepfool perturbation'):
                    delta_df, _, _, _, _ = deepfool( x[i:i+1], model_df )
                    delta.append(delta_df)
                delta = np.concatenate(delta,axis=0)
                savemat(f'perturbation/deepfool/{model_path}_df.mat', 
                        {'delta': delta,'x': x,'y': y})
                # delta = scaleDeepfool(psr,x,np.concatenate(delta,axis=0))
        # elif atk_type == 'UAP':
        # 	scaled_uni_per = scaleDeepfool(psr = psr,test_data = config.test_data, perturbation = UAP_data)
        # 	adv_data = config.test_data + scaled_uni_per - scaled_uni_per.mean()
        else:
            raise ValueError('atk_type must be fgsm or pgd')
    return delta
def get_adv_data(psr,model,x_batch,y_batch,method = 'fgsm',batch_size = None,to_tf_dataset = True,**kwargs):
	# generate testing adversarial data
	delta = []
	x_all = []
	y_all = []
	# for x_batch,y_batch in dataset:
	for x_buf,y_buf in zip(x_batch,y_batch):
		x_all.append(np.asarray(x_buf))
		y_all.append(np.asarray(y_buf))

	x_all = np.asarray(x_all)
	y_all = np.asarray(y_all)
	
	delta = gen_adv_data(x = x_all,
						y = y_all,
						model = model,
						atk_type = method,
						psr = psr,
						targeted = False,
						**kwargs,
						)

	if 'ant_flag' in kwargs.keys() and kwargs['ant_flag']:
		delta = np.expand_dims(np.mean(delta,axis=3),axis=3)
		delta = np.tile(delta,[1,1,1,3])
	# else:
	x_adv = x_all + delta
	if to_tf_dataset:
		adv_dataset = tf.data.Dataset.from_tensor_slices((x_adv, y_all))
		adv_dataset = adv_dataset.batch(batch_size)
		return adv_dataset
	else:
		return x_adv,y_all
	print('done')