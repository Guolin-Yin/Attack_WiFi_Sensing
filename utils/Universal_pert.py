import numpy as np
from DeepFool import deepfool
from tqdm import tqdm
import tensorflow as tf
from TOOLS import *
# from DeepNet import generateAdvExsPGD
def proj_lp(v, xi, p, data = None):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        # eps = np.sqrt( xi / np.mean( np.var( v ) / np.var( data ) ) )
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_perturbation(dataset,f,overshoot, delta=0.1, max_iter_uni = 100, xi=50, p=np.inf, max_iter_df=50):
    """
    :param dataset: Images of size (M,H,W,C) (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    v = 0
    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    itr = 0
    print(f'Number of classes is {f.output_shape}')
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)
        # print ('Starting pass number ', itr)
        # Go through the data set and compute the perturbation increments sequentially
        for k in tqdm(range(0, num_images),desc = f'Pass {itr}'):
            cur_img = dataset[k:(k+1), :, :, :]
            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img+v)).flatten())):
                # print('>> k = ', k, ', pass #', itr)
                # Compute adversarial perturbation
                dr,iter,_,_,_ = deepfool( cur_img + v, f, overshoot = overshoot, max_iter = max_iter_df )
                # dr = generateAdvExsPGD(input_CSI = cur_img + v, labels = labels,pretrained_model = f,psr =psr,
                #         n_iter = 5)
                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    # Project on l_p ball
                    v = proj_lp(v, xi, p,data = cur_img)
        itr = itr + 1
        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v
        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))
        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
        # Compute the estimated labels_pred in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()
        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)
    return v
def universal_perturbation_PGD(dataset,f,labels=None, delta=0.1, max_iter_uni = 100, xi=10, p=np.inf,psr=None):
    """
    :param dataset: Images of size (M,H,W,C) (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """

    # def generateAdvExsPGD( input_CSI, labels, pretrained_model, psr, n_iter: int = 0 ):
    #     # print( f'The number of iterations are {n_iter}' )
    #     loss_object = tf.keras.losses.CategoricalCrossentropy( )
    #     input_CSI = tf.convert_to_tensor( input_CSI, dtype = tf.float32 )
    #     labels = tf.convert_to_tensor( labels, dtype = tf.float32 )
    #     gradient = np.zeros( input_CSI.shape )
    #     for i in range( n_iter ):
    #         model_input = input_CSI + gradient
    #         with tf.GradientTape( persistent = True ) as tape:
    #             tape.watch( model_input )
    #             prediction = pretrained_model( model_input )
    #             loss = loss_object( labels, prediction )
    #         # gradient = gradient + (alpha/(i+1))*tf.sign(tape.gradient( loss, model_input ))
    #         g = tape.gradient( loss, model_input )
    #         g_norm = g / np.linalg.norm( g )
    #         gradient = gradient + (
    #                     np.sqrt( (psr / n_iter) / (np.mean( g_norm ** 2 ) / np.mean( input_CSI ** 2 )) ) * g_norm)
    #     return np.asarray( gradient )
    v = 0
    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    itr = 0
    print(f'Number of classes is {f.output_shape}')
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)
        # print ('Starting pass number ', itr)
        # Go through the data set and compute the perturbation increments sequentially
        for k in tqdm(range(0, num_images),desc = f'Pass {itr}'):
            cur_img = dataset[k:(k+1), :, :, :]
            cur_label = labels[k:(k+1)]
            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img+v)).flatten())):
                # print('>> k = ', k, ', pass #', itr)
                # Compute adversarial perturbation
                # dr,iter,_,_,_ = deepfool( cur_img + v, f, overshoot = overshoot, max_iter = max_iter_df )
                # dr = generateAdvExsPGD(input_CSI = cur_img + v, labels = cur_label,pretrained_model = f,psr =psr,
                #         n_iter = 5)


                # Make sure it converged...
                # if iter < max_iter_df-1:
                v = v + dr
                # Project on l_p ball
                # v = proj_lp(v, xi, p,data = cur_img)
                v = psr_limiter( psr = psr, perturbation = v, data = cur_img )
        itr = itr + 1
        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v
        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))
        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
        # Compute the estimated labels_pred in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()
        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)
    return v