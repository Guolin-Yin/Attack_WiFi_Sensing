import pandas as pd
from core_fn.randomised_smooth import Smooth
from core_fn.attacks import atk_fgsm, atk_pgd, atk_noise
import numpy as np
from tqdm import tqdm
whitebox_psr_list = np.linspace(0, 5e-4, 9)
perturbation_generators = {
    'fgsm': atk_fgsm,
    'pgd': atk_pgd,
    'noise': atk_noise,
}

def whitebox_smoothed_testing_loop(model, dataset, attack_method=None, psr_list=whitebox_psr_list,**kwargs):
    sigma = kwargs.get('sigma', 0.05)
    pred_N = kwargs.get('pred_N', 1000)
    smooth = Smooth(model, 276, sigma=sigma)
    n_iter = kwargs.get('n_iter', 3)
    misclassified_list = []
    # Create an empty DataFrame to store the results
    df = pd.DataFrame(columns=['psr', 'acc'])
    for i,psr in enumerate(psr_list):
        correct_counter = 0
        pred_counter = 0

        for x, y in tqdm(dataset):
            if psr > 0.0:
                if attack_method == 'fgsm':
                    delta = atk_fgsm(x, y, model, psr)
                elif attack_method == 'pgd':
                    delta = atk_pgd(x, y, model, psr, n_iter=n_iter)
                elif attack_method == 'noise':
                    delta = atk_noise(psr, x)
                else:
                    raise ValueError(f'Unknown attack method: {attack_method}')
            else:
                delta = 0
            pred = smooth.predict(x + delta, pred_N, alpha=0.001, batch_size=128)
            if pred != Smooth.ABSTAIN:
                correct_counter += np.sum(pred == np.argmax(y, axis=-1))
                pred_counter += 1
                if psr == 0.0 and pred != np.argmax(y, axis=-1):
                    misclassified_list.append(i)
        acc = correct_counter / pred_counter
        print(f'attack_method: {attack_method} psr: {psr}, acc: {acc}')
        
        # Append the results to the DataFrame
        df = pd.concat([df,pd.DataFrame({'psr': psr, 'acc': acc},  index=[i])], )
        # Save the DataFrame to an Excel file
        test_model_name = kwargs.get('test_model_name', '')
        if attack_method == 'pgd':
            df.to_excel(f'attack-method_{attack_method}-{n_iter}_test-std_{sigma}_model-name_{test_model_name}-ABSTAIN.xlsx', index=False)
        else:
            df.to_excel(f'attack-method_{attack_method}_test-std_{sigma}_model-name_{test_model_name}-ABSTAIN.xlsx', index=False)

    return df
