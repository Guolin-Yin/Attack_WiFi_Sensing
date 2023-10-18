import numpy as np
import tensorflow as tf
import os
def l2_limiter(eps,delta,x):
    return eps*(delta / tf.norm(delta,axis = 1).reshape(-1,1,1,1))
def psr_limiter(psr,perturbation,data,if_l2_distance = None):
    perturbation = np.array(perturbation)
    data = np.array(data)
    # normalise by the norm of the perturbation
    per_norm = perturbation / np.linalg.norm( perturbation.reshape(perturbation.shape[0],-1) ,axis = 1 ).reshape(-1,1,1,1)
    per_norm_factor = per_norm.reshape( per_norm.shape[0],-1 )
    data = data.reshape( data.shape[0],-1 )
    delta = (np.sqrt( psr / ( np.mean( per_norm_factor ** 2,axis = 1 ) / np.mean( data ** 2,axis = 1 ) ) )).reshape(-1,1,1,1) * (per_norm)
    if if_l2_distance:
        return delta, np.linalg.norm(delta)
    return delta
def is_within_certified_radius(delta, r1):
    # Compute the l2 norm of delta
    l2_norm = np.linalg.norm(delta)
    print(l2_norm)
    # Check if it's within the certified radius
    return l2_norm <= r1

def compute_psr(delta,data):
    delta = delta.reshape(delta.shape[0],-1)
    data = data.reshape(data.shape[0],-1)
    psr = np.mean( delta ** 2,axis = 1 ) / np.mean( data**2,axis = 1 )
    return psr

def compute_psr_from_distance(r1, data) -> float:
    data = np.array(data)
    data_norm = np.linalg.norm(data.reshape(data.shape[0],-1), axis=1)
    return float((r1 / data_norm) ** 2)

# def psr_limiter_2(psr, perturbation, data, if_l2_distance):

#     perturbation = np.array(perturbation)
#     data = np.array(data)
    
#     per_norm = perturbation / np.linalg.norm(perturbation.reshape(perturbation.shape[0], -1), axis=1).reshape(-1, 1, 1, 1)
#     per_norm_factor = per_norm.reshape(per_norm.shape[0], -1)
#     data = data.reshape(data.shape[0], -1)
    
#     delta = np.sqrt(psr / (np.mean(per_norm_factor ** 2, axis=1) / np.mean(data ** 2, axis=1))).reshape(-1, 1, 1, 1) * per_norm
    
#     if if_l2_distance:
#         l2_distance = np.linalg.norm(delta)
#         return delta, l2_distance
    
#     return delta