from WhiteBoxATK import *

'''Plot target attacks'''
# plot(pdf_name='Targeted_attack_compare',
# 		label_dict = {
#
# 				'FGSM'          : 'Targeted FGSM',
# 				'PGD'         : 'Targeted PGD (4 iterations)',
#
# 				'Guassian_Noise': 'Guassian Noise'
# 				},
# 		PGD = 'target_PGD_Rx_2_TL_3.mat',
# 		FGSM ='target_FGSM_Rx_2_TL_3.mat',
# 		Guassian_Noise ='target_GAUSSIAN_Rx_2_TL_3.mat'
#
# 		)
'''Plot time of the FGSM PGD DeepFool'''

xaxis = np.linspace(1,20,20,dtype = int)
PGD = np.asarray([ 0.006686071770778601,0.01941068, 0.03610705, 0.0532613 , 0.06856978,
       0.08590898, 0.09996847, 0.11718116, 0.12671299, 0.14637493,
       0.16647467, 0.17595797, 0.1948571 , 0.2148675 , 0.21665183,
       0.231657  , 0.24760692, 0.2650548 , 0.28970636, 0.3045767 ])
Deepfool = 4.62
FGSM = 0.02
plt.plot(xaxis,PGD,marker = 'o',label = 'PGD')
# plt.axhline(y=Deepfool, label = 'DeepFool',color ="red", linestyle ="--")
# plt.axhline(y=FGSM,label = 'FGSM',color ="green", linestyle ="--")
plt.legend()
plt.grid(True)
plt.xlabel('Number of iterations')
plt.ylabel('Time cost (s)')
plt.savefig('time_cost.pdf',bbox_inches='tight')


'''Plots the FGSM vs DeepFool vs PGD vs Gaussian noise'''
# plot(
# 		pdf_name = 'compare_deepfool_PGD_FGSM',
# 		label_dict = {
# 				'Deepfool': 'Deepfool',
# 				'FGSM': 'FGSM',
# 				'PGD_1': 'PGD (1 iteration)',
# 				'PGD_2': 'PGD (2 iterations)',
# 				'PGD_3': 'PGD (3 iterations)',
# 				'Guassian_Noise': 'Guassian Noise'
# 				},
# 		FGSM = 'fgsm_signfi_lab_PSR0to0.0005.mat',
# 		Deepfool = 'deepfool_signfi_lab_PSR0to0.0005.mat',
# 		PGD_1 = 'pgd_1_signfi_lab_PSR0to0.0005.mat',
# 		PGD_2 = 'pgd_2_signfi_lab_PSR0to0.0005.mat',
# 		PGD_3 = 'pgd_3_signfi_lab_PSR0to0.0005.mat',
# 		Guassian_Noise = 'gaussian_signfi.mat'
# 		)
'''Plot compare of deep fool and UAP'''
# plot(
# 		# pdf_name = None,
# 		pdf_name = 'compare_deepFool_and_UAP_indomain_cross_domain',
# 		label_dict = {
# 				'UAP_lab_to_lab'      : 'UAP',
# 				'Deepfool':'Deepfool',
# 				'Guassian_Noise': 'Guassian Noise'
# 				},
# 		Deepfool = 'deepfool_signfi_lab_PSR0to0.004.mat',
# 		UAP_lab_to_lab = 'UAP_signfi_lab_scale_1.mat',
# 		Guassian_Noise = 'gaussian_signfi_PSR0to0.004.mat'
# 		)
'''Plot PGD vs n_iter'''
# PGD_files = {}
# for i in range(1,21):
# 	PGD_files[f'PGD_{i}'] = str(f'pgd_{i}_signfi_lab_PSR0to0.0005.mat')
# plt_PGD(
# 		'PGD_with_Diiferent_iters',
# 		3,
# 		**PGD_files
# 		# PGD_1 = 'pgd_1_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_2 = 'pgd_2_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_3 = 'pgd_3_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_5 = 'pgd_5_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_7 = 'pgd_7_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_10 = 'pgd_10_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_15 = 'pgd_15_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_17 = 'pgd_17_signfi_lab_PSR0to0.0005.mat',
# 		# PGD_20 = 'pgd_20_signfi_lab_PSR0to0.0005.mat',
# 		)
'''Guassian noise atk lab'''
# a = loadmat(
# 		'resultsMat/Pub_results/cross_model_test/eleven_model_test/signfi_vic_lab_276_atk_guassian_noise.mat',
# 		squeeze_me = 1
# 		)
# plotGuassian_noiseForModel( fname = 'Guassian_noise_lab', **a )
'''Plot model compare'''
# plot_model_compare(
# 		psr_val = 0.0158,
# 		ifsave = True,
# 		vic_model = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19', ],
# 		home_to_home = 'cross_model_atk_home_276_vic_home_276_2',
# 		home_to_lab = 'cross_model_atk_home_276_vic_lab_276_2',
# 		lab_to_home = 'cross_model_atk_lab_276_vic_home_276_2',
# 		lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276_2'
# 		)
'''Plot cross domain compare'''
# plot(
# 		pdf_name = 'Cross_domain_atk_compare',
# 		label_dict = {
# 				'UAP_home_to_Lab': 'UAP ( $Home \Rightarrow Lab$ )',
# 				'UAP_lab_to_Home': 'UAP ( $Lab  \Rightarrow Home$ )',
# 				},
# 		pltGuassian = 1,
# 		UAP_home_to_Lab = 'UAP_signfi_atk_home_vic_lab_scale_1_method_2.mat',
# 		UAP_lab_to_Home = 'UAP_signfi_atk_lab_vic_home_scale_1_method_2.mat',
# 		# UAP_lab_to_lab = 'UAP_signfi_lab_scale_1.mat',
# 		)
'''Compare pseudo label'''
atk_r = 5
vic_r = 4
plot(
		pdf_name = 'Oringinal_label_vs_pseudo_label',
		pltGuassian = 0,
		Guassian_noise = f'Widar_atk_Guassian_victim_Rx{vic_r}_method_2.mat',
		marker_dict = {
				'full_label'    : '*',
				'pseudo_label'  : 'v',
				'Guassian_noise': 'X',
				},
		label_dict = {
				'full_label'    : 'True label',
				'pseudo_label'  : 'Pseudo label',
				'Guassian_noise': 'Guassian noise',
				},
		full_label = f'Unsupervised_labelForm_full_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
		pseudo_label = f'Unsupervised_labelForm_pseudo_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
		)
'''cross models matrix plot'''
plot_model_compare(
		psr_val = 0.0158,
		ifsave = True,
		vic_model = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19', ],
		home_to_home = 'cross_model_atk_home_276_vic_home_276_2',
		home_to_lab = 'cross_model_atk_home_276_vic_lab_276_2',
		lab_to_home = 'cross_model_atk_lab_276_vic_home_276_2',
		lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276_2'
		)