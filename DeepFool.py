import numpy as np
from tensorflow.keras.models import Model
import copy
import tensorflow as tf

def deepfool(input_CSI, pretrained_model,  overshoot=0.002, max_iter=50,):
    '''
    :param input_CSI: Whole dataset (n,input_shape)
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
    input_shape = np.shape(image_norm)
    pert_image = copy.deepcopy(image_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    x = tf.Variable(pert_image)
    # fs = model(x)
    k_i = label
    def loss_func(logits, I, k):
        # return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return logits[0, I[k]]
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            # loss_value = loss_func(one_hot_label_0, fs)
            loss_value = loss_func(fs, I, 0)
        # grad_orig = tape.gradient(fs[0, I[0]], x)
        grad_orig = tape.gradient(loss_value, x)
        for k in range(1, num_classes):
            one_hot_label_k = tf.one_hot(I[k], num_classes)
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                # loss_value = loss_func(one_hot_label_k, fs)
                loss_value = loss_func(fs, I, k)
            # cur_grad = tape.gradient(fs[0, I[k]], x)
            cur_grad = tape.gradient(loss_value, x)
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()
            pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))
            if pert_k < pert:
                pert = pert_k
                w = w_k
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        pert_image = image_norm + (1 + overshoot) * r_tot
        x = tf.Variable(pert_image)
        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, np.asarray(pert_image)
# def deepfool(image, f, grads, num_classes=10, overshoot=0.02, max_iter=50):
#
#     """
#        :param image: Image of size HxWx3
#        :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
#        :param grads: gradient functions with respect to input (as many gradients as classes).
#        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
#        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
#        :param max_iter: maximum number of iterations for deepfool (default = 10)
#        :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
#     """
#
#     f_image = np.array(f(image)).flatten()
#     I = (np.array(f_image)).flatten().argsort()[::-1]
#
#     I = I[0:num_classes]
#     label = I[0]
#
#     input_shape = image.shape
#     pert_image = image
#
#     f_i = np.array(f(pert_image)).flatten()
#     k_i = int(np.argmax(f_i))
#
#     w = np.zeros(input_shape)
#     r_tot = np.zeros(input_shape)
#
#     loop_i = 0
#
#     while k_i == label and loop_i < max_iter:
#
#         pert = np.inf
#         gradients = np.asarray(grads(pert_image,I))
#
#         for k in range(1, num_classes):
#
#             # set new w_k and new f_k
#             w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
#             f_k = f_i[I[k]] - f_i[I[0]]
#             pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
#
#             # determine which w_k to use
#             if pert_k < pert:
#                 pert = pert_k
#                 w = w_k
#
#         # compute r_i and r_tot
#         r_i =  pert * w / np.linalg.norm(w)
#         r_tot = r_tot + r_i
#
#         # compute new perturbed image
#         pert_image = image + (1+overshoot)*r_tot
#         loop_i += 1
#
#         # compute new label
#         f_i = np.array(f(pert_image)).flatten()
#         k_i = int(np.argmax(f_i))
#
#     r_tot = (1+overshoot)*r_tot
#
#     return r_tot, loop_i, k_i,
#


