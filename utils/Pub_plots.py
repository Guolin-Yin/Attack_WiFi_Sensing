# from ATKTEST import plot
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from utils.TOOLS import comp_atk_success_rate
from collections import OrderedDict
from matplotlib.colors import LightSource
import os
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from ATKMethods import compute_gradient, l2_limiter, compute_psr
def draw_loss(model, X,Y, psr,):
    assert X.shape[0] == 1, 'X should be a single sample'
    def psr_to_eps(psr,pert,data):
        return np.sqrt( psr / ( np.mean( pert ** 2 ) / np.mean( data ** 2 ) ) )
    loss_fn = tf.keras.losses.categorical_crossentropy
    g = compute_gradient(model_fn=model, 
                         loss_fn=loss_fn, 
                         x=X, y=Y)
    per1 = tf.sign(g)
    per1 = tf.reshape(per1,-1)
    # psr to epsilon
    eps = psr_to_eps(psr,per1,X)
    Xi, Yi = np.meshgrid(np.linspace(0, eps,100), np.linspace(0,eps,100))
 
 
 
    np.random.seed(0)
    per2 = np.sign(np.random.randn(per1.shape[0]))
    
    all_deltas = np.array([Xi.flatten(), Yi.flatten()]).T @ np.array([per2, per1])
    x_adv = np.reshape(all_deltas,(-1,200,60,3)) + X
 
    yp = []
    b_size = 100
    for i in range(x_adv.shape[0]//b_size):
        yp.append(model(x_adv[i*b_size:(i+1)*b_size]))
    yp = np.concatenate(yp,axis=0)
    # yp = model(x_adv)
    Zi = loss_fn(yp, Y[0:1].repeat(yp.shape[0],0))
    a,b = Xi.shape
    Zi = np.reshape(Zi,(a,b))
    #Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.ticklabel_format( style='sci', scilimits=(0,0), )
    ls = LightSource(azdeg=0, altdeg=200)
    rgb = ls.shade(Zi, plt.cm.coolwarm)
    Xi, Yi = np.meshgrid(np.linspace(0, psr,100), np.linspace(0,psr,100))
    surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, linewidth=0,
                       antialiased=True, facecolors=rgb) 
    ax.set_xlabel('PSR (Guassian noise)')
    ax.set_ylabel('PSR (FGSM)')
    ax.set_zlabel('loss') 
def extract_info(filename):
    filename = filename.replace('.h5','')
    if 'robust_adv_training' in filename:
        adv_training_method = filename.split('robust_adv_training_')[1].split('_')[0]
        psr = float(filename.split('psr_')[1].split('_')[0])
        model_name = filename.split('_')[6]
        n_iters = 'None' if 'fgsm' in filename else int(filename.split('niter_')[1].split('_')[0])
        return adv_training_method,psr,model_name,n_iters
    else:
        return 'Normal',None,model_name,None
def get_test_info_from_dict(result,psr_idx):

    '''
    result should be a dictionary, 
    the key of result is the name of the test: [defense_method]_defense_against_[attacker_method]
    each key accociates with a list of accuracy of the test
    
    :param psr_idx: the index of the psr
        each test has N psr, the index of psr is from 0 to N-1
        this function will return the psr_idx-th psr for each test
        
    :return: decoded attacker method, decoded defense method, attack success rate, accuracy
    '''
    def get_iter(key):
        # for key in result:
        if 'pgd' in key:
            return int(key.split('_')[1])
        elif 'Normal' in key:
            return 0
        elif 'fgsm' in key:
            return 1
        else:
            assert False, 'unknown key'
            

    Attacker_methods = []
    Defense_methods = []
    acc_all_adv = []
    asr_all = []
    for key in result.keys():
        Attacker_methods.append(key.split('against_')[1])
        Defense_methods.append(key.split('_defense')[0])
    
    Defense_methods = list(dict.fromkeys(Defense_methods))
    Attacker_methods = list(dict.fromkeys(Attacker_methods))
    
    Attacker_methods.sort(key = get_iter)
    Defense_methods.sort(key = get_iter)
    
    ordered_result = {}
    for dfs in Defense_methods:
        for atk in Attacker_methods:
            key = dfs+'_defense_against_'+atk
            ordered_result[key] = result[key]
    
    
            asr_all.append(comp_atk_success_rate(result[key])[psr_idx])
            acc_all_adv.append(result[key][psr_idx])
    return Attacker_methods,Defense_methods,asr_all,acc_all_adv
def heatmap_defense_vs_attacker(result,Attacker_methods,Defense_methods):
    fig, ax = plt.subplots()
    
    # We want to show all ticks...
    ax.set_yticks(np.arange(len(Attacker_methods)))
    ax.set_xticks(np.arange(len(Defense_methods)))
    Attacker_methods.reverse()
    # ... and label them with the respective list entries
    ax.set_yticklabels(Attacker_methods)
    ax.set_xticklabels(Defense_methods)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Defense_methods)):
        for j in range(len(Attacker_methods)):
            text = ax.text(i, j,  result[ j,i],ha="center", va="center", color="black" )
    im = ax.imshow(result,cmap = 'Oranges')
def hp_plotter_defense(acc_dic,psr_idx, matric = 'acc'):
    psr = acc_dic['psr']
    acc_dic.pop('psr')
    print('Showing PSR = ',psr[psr_idx], 'with matrix = ',matric)
    # acc_dic = dict(OrderedDict(sorted(acc_dic.items())))
    Attacker_method,Defense_method,asr,acc = get_test_info_from_dict(acc_dic,psr_idx)
    

    if matric == 'acc':
        out_matrix= acc
    elif matric == 'asr':
        out_matrix = asr
    else:
        assert False, 'unknown matric'
    
    out_matrix_reshaped = np.asarray(out_matrix).reshape((len(Defense_method),len(Attacker_method)))
    out_matrix_reshaped = np.round(out_matrix_reshaped*100,2)
    out_matrix_reshaped = np.rot90(out_matrix_reshaped,1)
    heatmap_defense_vs_attacker(out_matrix_reshaped,Attacker_method,Defense_method)
def heatmap(acc_dict,title,vic_model = ['defult', 'alex1', 'alex2', 'alex3', 'vgg19'],idx = -1,ifsave = False):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import copy
    atk_model = copy.deepcopy( vic_model )
    atk_model.reverse( )
    print_name_dict = {
            'defult': '$C_{D}$',
            'alex1': '$C_{D_1}$',
            'alex2': '$C_{D_2}$',
            'alex3': '$C_{D_2}$',
            'vgg19': 'VGG19'
            }
    atk_model_print = [print_name_dict[key] for key in atk_model]
    vic_model_print = [print_name_dict[key] for key in vic_model]
    def getAccMatrix( acc_dict, idx, vic_model,atk_model ):
        # atk_model = copy.deepcopy( vic_model )
        # atk_model.reverse( )
        keys = list( acc_dict.keys( ) )
        acc_matrix = [ ]
        acc_matrix_ori = []
        for atk in atk_model:
            vic_acc = [ ]
            vic_ori_acc = [ ]
            for vic in vic_model:
                for key in keys:
                    if key =='psr':
                        continue
                    if vic != key.split( '_' )[ 2 ] or atk != key.split( '_' )[ 4 ]:
                        continue
                    vic_acc.append( acc_dict[ key ][ idx ] )
                    vic_ori_acc.append(acc_dict[key][0])
            acc_matrix.append( vic_acc )
            acc_matrix_ori.append(vic_ori_acc)
        return np.asarray(acc_matrix),np.asarray(acc_matrix_ori)

    acc,acc_ori = getAccMatrix(acc_dict = acc_dict,
                                vic_model = vic_model,
                                atk_model = atk_model,
                                idx = idx)
    '''Model specific factors'''
    # atk_model = [ 'cnnlstm', 'cnn', 'alex3', 'alex2', 'alex1', 'default' ]
    # victim = [ 'default', 'alex1', 'alex2', 'alex3', 'cnn', 'cnnlstm' ]

    # accpgd_attack = np.round( 1 - np.array( [
    # 		[ 0.395, 0.355, 0.551, 0.404, 0.298, 0.098, ],
    # 		[ 0.812, 0.766, 0.992, 0.687, 0.55, 0.475, ],
    # 		[ 0.506, 0.4438, 0.669, 0.393, 0.324, 0.21, ],
    # 		[ 0.364, 0.2857, 0.572, 0.416, 0.375, 0.158, ],
    # 		[ 0.33, 0.1896, 0.504, 0.424, 0.273, 0.192, ],
    # 		[ 0.383, 0.274, 0.584, 0.433, 0.332, 0.196, ],] ),2 )
    acc = np.round((acc_ori-acc)/acc_ori,2)
    '''Task specific factors'''
    # target models using default model
    # atk_model= ['wiar','widar','signfi']
    # victim= ['signfi','widar','wiar']
    # accpgd_attack = np.round(1 - np.array([
    # 		[0.38,0.25,0.329],
    # 		[0.5765,0.194,0.972],
    # 		[0.2346,0.42,0.966]
    # 		]),2)
    fig, ax = plt.subplots( )
    im = ax.imshow( acc,cmap = 'Oranges',vmin=0.65, vmax=1 )
    ax.set_xlabel('Victim Model')
    ax.set_ylabel('Attacker Model')
    # im = plt.imshow( acc, cmap = 'magma_r' )
    # Show all ticks and label them with the respective list entries
    plt.yticks(  np.arange( len(atk_model ) ),atk_model_print )
    plt.xticks(  np.arange( len(vic_model ) ),vic_model_print )
    # plt.title(title)
    # Rotate the tick labels_pred and set their alignment.
    plt.setp( ax.get_xticklabels( ), rotation=45, ha="right",
              rotation_mode="anchor" )
    # Loop over data dimensions and create text annotations.
    for i in range( len( vic_model ) ):
        for j in range( len( atk_model ) ):
            text = ax.text( j, i, acc[ i, j ],
                            ha="center", va="center", color="black" )

    # plt.colorbar(im)
    # plt.clim( 0, 1 )
    fig.tight_layout( )
    plt.show(  )
    if ifsave:
        out = os.path.join('RESULTS_FIGS',title)
        plt.savefig( out + '.pdf',bbox_inches='tight',  )
def plot(pdf_name=None,pltGuassian=0,marker_dict = None,label_dict = None,linestyle_dict = None,to_db = False,**mat_names):
    if marker_dict == None:
        marker_dict = {
            'Deepfool': 'o',
            'FGSM': "v",
            'Guassian_Noise': 'X',
            'Guassian_noise': 'X',
            'UAP':'s',
            'UAP_home_to_Lab':'h',
            'UAP_lab_to_lab':'s',
            'UAP_lab_to_Home':'D',
            'defult': 'H',
            'alexnet': 'H',
            'alex1': 'o',
            'alex2': 'v',
            'alex3': '^',
            'cnn': '<',
            'vgg8': '>',
            'vgg10': '8',
            'vgg16': 's',
            'vgg19': 'p',
            'resnet': 'P',
            'resnet6': '*',
            'Rx1': 'P',
            'Rx2': 'p',
            'Rx3': '^',
            'Rx4': '<',
            'Rx5': '>',
            'Rx6': 'v',
            'PGD':'h',
            'PGD_1':'H',
            'PGD_2':'^',
            'PGD_3':'v',
            'PGD_4':'P',
            'PGD_5':'8',
            'PGD_7':'s',
            'PGD_10':'*',
            'PGD_15':'D',
            'PGD_17':'<',
            'PGD_20':'>',

    }

    color_dict = {
            'Deepfool': 'b',
            'FGSM'    : 'm'
    }
    result_dir = 'resultsMat/Pub_results/'
    keys = list( mat_names.keys( ) )
    ax = plt.figure( figsize=(8, 5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer=True ) )
    for key in keys:
        path = os.path.join(result_dir,mat_names[key])
        result = loadmat(path,squeeze_me=True)
        psr = 10*np.log10(result['psr']+0.0000001) if to_db else result['psr']

        # result['acc'][4:7] = [.58,.55,.53]
        acc = (result[ 'acc' ][0] - result[ 'acc' ])/result[ 'acc' ][0]
        if label_dict == None:
            label = key
        else:
            label = label_dict[key]
        ax.plot(psr[0:7],
                acc[0:7],
                label=label,
                marker = marker_dict[key],
                linestyle = linestyle_dict[key],
                fillstyle = Line2D.fillStyles[-1])
        if 'Guassian_noise' in result and pltGuassian:
            ax.plot(
                    psr,(result[ 'Guassian_noise' ][0] - result[ 'Guassian_noise' ])/result[ 'Guassian_noise' ][0],
                    label = 'Guassian noise '+ '(' +key.split('_')[-1] + ')',
                    marker = marker_dict[ key ],
                    fillstyle = Line2D.fillStyles[ -1 ]
                    )
    ax.ticklabel_format( style='sci', scilimits=(0,0), axis='x' )
    fsize = 14
    plt.xticks( fontsize=fsize )
    plt.yticks( fontsize=fsize )
    plt.ylim(-0.03,1)
    plt.grid(True)
    ax.set_xlabel( 'PSR', fontsize=fsize )
    ax.set_ylabel( 'Attack Success Rate (ASR)', fontsize=fsize )
    ax.legend( fontsize=10, ncol=2,loc = 'best',
            # bbox_to_anchor=(1, 0.1)
            )
    plt.show()
    if pdf_name is not None:
        out = os.path.join('RESULTS_FIGS',pdf_name)
        # out = os.path.join('E:\\',pdf_name)
        plt.savefig( out + '.pdf',bbox_inches='tight',  )
def plot_model_compare(psr_val = 0.0005,ifsave = False,vic_model = ['defult','alex1','alex2','alex3','cnn','vgg8','vgg10','vgg19',],**mat_names):
    for key in mat_names:
        acc_all = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test/'+mat_names[key],squeeze_me=1)
        acc_all.pop( '__header__', None )
        acc_all.pop( '__version__', None )
        acc_all.pop( '__globals__', None )
        title = 'Attacker_'+mat_names[key].split('atk_')[1].split('_')[0] +'_Victim_model_'+mat_names[key].split(
                'vic_')[1].split('_')[0]
        if 'psr' in acc_all.keys():
            psr_range = acc_all['psr']
            # idx = np.where( psr_range == psr_val )[ 0 ][ 0 ]
            idx = ((psr_range - psr_val) ** 2).argmin( )
        else:
            idx = -1

        heatmap( acc_dict = acc_all,
                title=title + f'_PSR={psr_range[idx]:.4f}',
                vic_model = vic_model,
                idx = idx,
                ifsave = ifsave)
def plotGuassian_noiseForModel(fname,**model_names):
    marker_dict = {
            'Deepfool': 'o',
            'FGSM': "v",
            'Guassian_Noise': 'X',
            'UAP':'s',
            'UAP_home_to_lab':'h',
            'UAP_lab_to_lab':'s',
            'UAP_lab_to_home':'D',
            'defult': 'H',
            'alexnet': 'H',
            'alex1': 'o',
            'alex2': 'v',
            'alex3': '^',
            'cnn': '<',
            'vgg8': '>',
            'vgg10': '8',
            'vgg16': 's',
            'vgg19': 'p',
            'resnet': 'P',
            'resnet6': '*',
            'signfi_vic_defult_atk_defult': 'H',
            'signfi_vic_alex1_atk_defult' :'o' ,
            'signfi_vic_alex2_atk_defult' : 'v',
            'signfi_vic_alex3_atk_defult' : '^',
            'signfi_vic_vgg19_atk_defult' : 'p',
    }
    legend = {
            'alexnet': '$C_{D}  (Noise)$',
            'alex1' : '$C_{D_1}$ (Noise)',
            'alex2' : '$C_{D_2}$ (Noise)',
            'alex3' : '$C_{D_2}$ (Noise)',
            'vgg19' : 'VGG19 (Noise)',
            'signfi_vic_defult_atk_defult': '$C_{D}(Home) \Rightarrow  C_{D}(Lab)$',
            'signfi_vic_alex1_atk_defult': '$C_{D}(Home) \Rightarrow C_{D_1}(Lab)$',
            'signfi_vic_alex2_atk_defult': '$C_{D}(Home) \Rightarrow C_{D_2}(Lab)$',
            'signfi_vic_alex3_atk_defult': '$C_{D}(Home) \Rightarrow C_{D_2}(Lab)$',
            'signfi_vic_vgg19_atk_defult': '$C_{D}(Home) \Rightarrow VGG19(Lab)$',
            }
    ax = plt.figure( figsize=(8, 5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer=True ) )
    psr = model_names['psr']
    model_names.pop('psr')
    model_names.pop('__header__', None)
    model_names.pop('__version__',None)
    model_names.pop('__globals__',None)
    for i,name in enumerate(['alexnet','alex1',
                             # 'alex2',
                             'alex3','vgg19',
                 'signfi_vic_defult_atk_defult',
                 'signfi_vic_alex1_atk_defult',
                 # 'signfi_vic_alex2_atk_defult',
                 'signfi_vic_alex3_atk_defult',
                 'signfi_vic_vgg19_atk_defult',]
                 ):
        line = ['solid','solid','solid','solid','dashed','dashed','dashed','dashed']
        result = model_names[name]
        acc = (result[ 0 ] - result)/result[ 0 ]
        ax.plot(psr[0:8],
                acc[0:8],
                label=legend[name],
                marker = marker_dict[name],linestyle=line[i],
                fillstyle = Line2D.fillStyles[-1])
    ax.ticklabel_format( style='sci', scilimits=(0, 0), axis='x' )
    plt.legend(fontsize=10, ncol=2,bbox_to_anchor=(0.376, 0.2),labelspacing=.1,handletextpad=0.1)
    plt.grid(True)
    fsize = 14
    plt.xticks( fontsize=fsize )
    plt.yticks( fontsize=fsize )
    plt.ylim(-0.03,1)
    plt.grid(True)
    ax.set_xlabel( 'PSR', fontsize=fsize )
    ax.set_ylabel( 'Attack Success Rate (ASR)', fontsize=fsize )
    out = os.path.join( 'RESULTS_FIGS', fname )
    plt.savefig( out + '.pdf', bbox_inches = 'tight', )
def plot_adv_results(config,plot_info,save_name = None,matric: str = None,to_dB = False, **kwargs):
    #config path

    if not os.path.exists(config.results_dir['adv_pdf']):
        os.makedirs(config.results_dir['adv_pdf'])
    # plot
    fsize = 14
    marker_dict = kwargs['marker_dic']
    color_dict = kwargs['color_dic']
    fig = plt.figure( figsize=(8, 6) )
    # fig.xaxis.set_major_locator( MaxNLocator( integer=True ) )
    if isinstance(plot_info,dict):
        plot_info = [plot_info] 
        
    for idx, result_dic in enumerate(plot_info):
        psr = 10*np.log10(result_dic['psr']) if to_dB else result_dic['psr']
        result_dic.pop('psr',None)
        result_dic.pop( '__header__', None )
        result_dic.pop( '__version__', None )
        result_dic.pop( '__globals__', None )
        
        
        ax = fig.add_subplot(1,len(plot_info),idx + 1)
        for label in sorted(result_dic):
            acc = result_dic[label]
            if matric == 'acc':
                mtc = acc
            elif matric == 'asr':
                mtc = (acc[0] - acc)/acc[0]
            if 'legend' in kwargs.keys():
                legend = kwargs['legend'][label]
            else:
                legend = label
            ax.plot(psr,
                mtc,
                label=legend,
                marker = marker_dict[label],
                color = color_dict[label],
                # linestyle = linestyle_dict[key],
                fillstyle = Line2D.fillStyles[-1]
                )
            # ax.text( 0.001,0.0001,'Iteration: {}'.format(idx + 1), fontsize=12)
            if kwargs['title']:
                ax.set_title(kwargs['title'][idx])
            if isinstance(plot_info,list):
                plot_idx = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
                ax.set_xlabel( f'PSR \n ({plot_idx[idx]})', fontsize=fsize )
            else:
                ax.set_xlabel( 'PSR', fontsize=fsize )
        
        ax.ticklabel_format( style='sci', scilimits=(0,0), axis='x' )
        plt.xticks( fontsize=fsize )
        plt.yticks( fontsize=fsize )
        plt.ylim(-0.03,1.05)
        plt.grid(True)
        
        # ax.set_ylabel( 'ASR' if matric == 'asr' else 'Accuracy', fontsize=fsize )
        
        fig.supylabel('ASR' if matric == 'asr' else 'Accuracy', fontsize=fsize )
        # fig.supxlabel('Iteration')
        plt.legend(fontsize=10, ncol=1,loc = 'best',
                #    bbox_to_anchor=(0.376, 0.2),
                labelspacing=.1,handletextpad=0.1)
    if save_name:
        save_path = os.path.join('RESULTS_FIGS',save_name)
        # save_path = os.path.join('C:\Users/29073\Dropbox\应用\Overleaf',save_name)
        
        plt.savefig(save_path+'.pdf',bbox_inches = 'tight', )
    plt.show()
if __name__ == '__main__':
    
    # plot_model_compare(
    # 		psr_val = 0.0158,
    # 		ifsave = 1,
    # 		vic_model = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19', ],
    # 		# home_to_home = 'cross_model_atk_home_276_vic_home_276_2',
    # 		home_to_lab = 'cross_model_atk_home_276_vic_lab_276_2',
    # 		# lab_to_home = 'cross_model_atk_lab_276_vic_home_276_2',
    # 		# lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276_2'
    # 		)
    # for atk_r in [1,2,3,5,6]:
    # atk_r = 5
    # vic_r = 4
    # plot(
    # 		pdf_name = f'Oringinal_label_vs_pseudo_label_Diferent_number_of_clusters_atk_r_{atk_r}',
    # 		pltGuassian = 0,
    # 		Guassian_noise = f'Widar_atk_Guassian_victim_Rx{vic_r}_method_2.mat',
    # 		marker_dict = {
    # 				'full_label'    : '*',
    # 				'pseudo_label_3': 'v',
    # 				'pseudo_label_4': 'h',
    # 				'pseudo_label_5': 'H',
    # 				'pseudo_label_6': 'p',
    # 				'pseudo_label_7': 'P',
    # 				'pseudo_label_8': 'D',
    # 				'pseudo_label_9': '^',
    # 				'Guassian_noise': 'X',
    # 				},
    # 		label_dict = {
    # 				'full_label'    : '$U_r$',
    # 				'pseudo_label_3': '$U_p$ (3 clusters)',
    # 				'pseudo_label_4': '$U_p$ (4 clusters)',
    # 				'pseudo_label_5': '$U_p$ (5 clusters)',
    # 				'pseudo_label_6': '$U_p$ (6 clusters)',
    # 				'pseudo_label_7': '$U_p$ (7 clusters)',
    # 				'pseudo_label_8': '$U_p$ (8 clusters)',
    # 				'pseudo_label_9': '$U_p$ (9 clusters)',
    # 				'Guassian_noise': 'Guassian noise',
    # 				},
    # 		linestyle_dict = {
    # 				'full_label'    : '-',
    # 				'pseudo_label_3': '--',
    # 				'pseudo_label_4': '--',
    # 				'pseudo_label_5': '--',
    # 				'pseudo_label_6': '--',
    # 				'pseudo_label_7': '--',
    # 				'pseudo_label_8': '--',
    # 				'pseudo_label_9': '--',
    # 				'Guassian_noise': '-',
    # 				},
    # 		full_label = f'Unsupervised_labelForm_full_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_3 = f'Unsupervised_labelForm_pseudo_label_NClusters_3_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_4 = f'Unsupervised_labelForm_pseudo_label_NClusters_4_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_5 = f'Unsupervised_labelForm_pseudo_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_6 = f'Unsupervised_labelForm_pseudo_label_NClusters_6_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_7 = f'Unsupervised_labelForm_pseudo_label_NClusters_7_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_8 = f'Unsupervised_labelForm_pseudo_label_NClusters_8_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label_9 = f'Unsupervised_labelForm_pseudo_label_NClusters_9_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		)
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
    # import numpy as np
    # xaxis = np.linspace(1,20,20,dtype = int)
    # PGD = np.asarray([ 0.006686071770778601,0.01941068, 0.03610705, 0.0532613 , 0.06856978,
    #        0.08590898, 0.09996847, 0.11718116, 0.12671299, 0.14637493,
    #        0.16647467, 0.17595797, 0.1948571 , 0.2148675 , 0.21665183,
    #        0.231657  , 0.24760692, 0.2650548 , 0.28970636, 0.3045767 ])
    # Deepfool = 4.62
    # FGSM = 0.02
    #
    # ax = plt.figure( figsize = (8, 5) ).gca( )
    # plt.plot(xaxis,PGD,marker = 'o',label = 'PGD',fillstyle = Line2D.fillStyles[-1])
    # # plt.axhline(y=Deepfool, label = 'DeepFool',color ="red", linestyle ="--")
    # # plt.axhline(y=FGSM,label = 'FGSM',color ="green", linestyle ="--")
    # plt.legend()
    # plt.grid(True)
    #
    # fsize = 14
    # plt.xticks( fontsize = fsize )
    # plt.yticks( fontsize = fsize )
    #
    # plt.grid( True )
    #
    #
    #
    # plt.xlabel('Number of iterations',fontsize = fsize)
    # plt.ylabel('Time cost (s)',fontsize=fsize)
    # plt.savefig('time_cost.pdf',bbox_inches='tight')
    #%%
    '''Plots the FGSM vs DeepFool vs PGD vs Gaussian noise'''
    plot(
            pdf_name = 'compare_deepfool_PGD_FGSM',
            label_dict = {
                    'Deepfool': 'Deepfool',
                    'FGSM': 'FGSM',
                    'PGD_1': 'PGD (1 iteration)',
                    'PGD_2': 'PGD (2 iterations)',
                    'PGD_3': 'PGD (3 iterations)',
                    'Guassian_Noise': 'Guassian Noise'
                    },
            FGSM = 'fgsm_signfi_lab_PSR0to0.0005.mat',
            Deepfool = 'deepfool_signfi_lab_PSR0to0.0005.mat',
            PGD_1 = 'pgd_1_signfi_lab_PSR0to0.0005.mat',
            PGD_2 = 'pgd_2_signfi_lab_PSR0to0.0005.mat',
            PGD_3 = 'pgd_3_signfi_lab_PSR0to0.0005.mat',
            Guassian_Noise = 'gaussian_signfi.mat'
            )
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
    # acc = loadmat('resultsMat/Pub_results/cross_model_test/eleven_model_test/'+'cross_model_atk_home_276_vic_lab_276_2',squeeze_me=1)
    # out = {}
    # for key,accuracy in acc.items():
    # 	if 'atk_defult' in key:
    # 		out[key] = accuracy

    # a = loadmat(
    # 		'resultsMat/Pub_results/cross_model_test/eleven_model_test/signfi_vic_lab_276_atk_guassian_noise.mat',
    # 		squeeze_me = 1
    # 		)
    # a.update(out)
    # plotGuassian_noiseForModel( fname = 'Guassian_noise_lab', **a )
    '''Plot model compare'''
    # plot_model_compare(
    # 		psr_val = 0.0158,
    # 		ifsave = 1,
    # 		vic_model = [ 'defult', 'alex1', 'alex3', 'vgg19', ],
    # 		# home_to_home = 'cross_model_atk_home_276_vic_home_276_2',
    # 		home_to_lab = 'cross_model_atk_home_276_vic_lab_276_2',
    # 		# lab_to_home = 'cross_model_atk_lab_276_vic_home_276_2',
    # 		# lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276_2'
    # 		)
    '''Plot cross domain compare'''
    # plot(
    # 		pdf_name = 'Cross_domain_atk_compare',
    # 		label_dict = {
    # 				'UAP_home_to_Lab': 'UAP ( $Home \Rightarrow Lab$ )',
    # 				'UAP_lab_to_Home': 'UAP ( $Lab  \Rightarrow Home$ )',
    # 				},
    # 		pltGuassian = 1,
    # 		UAP_home_to_Lab = 'UAP_signfi_atk_home_vic_lab_scale_1_more_psr.mat',
    # 		UAP_lab_to_Home = 'UAP_signfi_atk_lab_vic_home_scale_1_more_psr.mat',
    # 		# UAP_lab_to_lab = 'UAP_signfi_lab_scale_1.mat',
    # 		)

    # home_to_lab_1 = loadmat(os.path.join(result_dir,'UAP_signfi_atk_home_vic_lab_scale_1_method_2_18072022.mat'),squeeze_me = True)
    # home_to_lab_2 = loadmat(os.path.join(result_dir,'UAP_signfi_atk_home_vic_lab_scale_1_method_2.mat'),squeeze_me = True)
    # lab_to_home_1 = loadmat(os.path.join(result_dir,'UAP_signfi_atk_lab_vic_home_scale_1_method_2_18072022'),squeeze_me = True)
    # lab_to_home_2 = loadmat(os.path.join(result_dir,'UAP_signfi_atk_lab_vic_home_scale_1_method_2'),squeeze_me = True)
    # home_to_lab = {'Guassian_noise':np.concatenate((home_to_lab_2['Guassian_noise'],home_to_lab_1['Guassian_noise'])),
    #               'psr':np.concatenate((home_to_lab_2['psr'],home_to_lab_1['psr'] + 0.001)),
    # 				'acc':np.concatenate((home_to_lab_2['acc'],home_to_lab_1['acc']))}
    # lab_to_home = {'Guassian_noise':np.concatenate((lab_to_home_2['Guassian_noise'],lab_to_home_1['Guassian_noise'])),
    #               'psr':np.concatenate((lab_to_home_2['psr'],lab_to_home_1['psr']  + 0.001 )),
    # 				'acc':np.concatenate((lab_to_home_2['acc'],lab_to_home_1['acc']))}
    # savemat(os.path.join(result_dir,'UAP_signfi_atk_home_vic_lab_scale_1_more_psr.mat'),home_to_lab)
    # savemat(os.path.join(result_dir,'UAP_signfi_atk_lab_vic_home_scale_1_more_psr.mat'),lab_to_home)
    '''Compare pseudo label'''
    # atk_r = 5
    # vic_r = 4
    # plot(
    # 		# pdf_name = 'Oringinal_label_vs_pseudo_label',
    # 		pltGuassian = 0,
    # 		marker_dict = {
    # 				'full_label'    : '*',
    # 				'pseudo_label'  : 'v',
    # 				'Guassian_noise': 'X',
    # 				},
    # 		label_dict = {
    # 				'full_label'    : 'True label',
    # 				'pseudo_label'  : 'Pseudo label',
    # 				'Guassian_noise': 'Guassian noise',
    # 				},
    # 		linestyle_dict = {},
    # 		full_label = f'Unsupervised_labelForm_full_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		pseudo_label = f'Unsupervised_labelForm_pseudo_label_atkRx_Rx{atk_r}_vicRx_Rx{vic_r}_method_2.mat',
    # 		Guassian_noise = f'Widar_atk_Guassian_victim_Rx{vic_r}_method_2.mat',
    # 		)
    '''cross models matrix plot'''
    # plot_model_compare(
    # 		psr_val = 0.0158,
    # 		ifsave = True,
    # 		vic_model = [ 'defult', 'alex1', 'alex2', 'alex3', 'vgg19', ],
    # 		home_to_home = 'cross_model_atk_home_276_vic_home_276_2',
    # 		home_to_lab = 'cross_model_atk_home_276_vic_lab_276_2',
    # 		lab_to_home = 'cross_model_atk_lab_276_vic_home_276_2',
    # 		lab_to_lab = 'cross_model_atk_lab_276_vic_lab_276_2'
    # 		)
# %%
