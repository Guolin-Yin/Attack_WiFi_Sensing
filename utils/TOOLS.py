import numpy as np
def PSRCompute(perturbation_with_eps,data):
	data = np.squeeze(data)
	perturbation_with_eps = np.squeeze(perturbation_with_eps )

	p_perturbation =np.mean( perturbation_with_eps ** 2,axis=0 )
	p_data = np.mean( data ** 2 ,axis=0)

	return np.mean(p_perturbation/p_data)