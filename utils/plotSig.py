import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from scipy.io import savemat, loadmat
import scipy.io as sio
from os.path import dirname, join as pjoin
import os
result_dir = os.getcwd() + '\\resultsMat'
class pltConfusionMatrix():
    def __init__( self ):
        pass
    def make_confusion_matrix(self,
            cf,
            group_names = None,
            categories = 'auto',
            count = True,
            percent = False,
            cbar = True,
            xyticks = True,
            xyplotlabels = True,
            sum_stats = True,
            figsize = None,
            cmap = 'Oranges',
            title = None
            ):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = [ '' for i in range( cf.size ) ]

        if group_names and len( group_names ) == cf.size:
            group_labels = [ "{}\n".format( value ) for value in group_names ]
        else:
            group_labels = blanks

        if count:
            group_counts = [ "{0:0.0f}\n".format( value ) for value in cf.flatten( ) ]
        else:
            group_counts = blanks

        if percent:
            group_percentages = [ "{0:.2%}".format( value ) for value in cf.flatten( ) / np.sum( cf ) ]
        else:
            group_percentages = blanks

        box_labels = [ f"{v1}{v2}{v3}".strip( ) for v1, v2, v3 in zip( group_labels, group_counts, group_percentages ) ]
        box_labels = np.asarray( box_labels ).reshape( cf.shape[ 0 ], cf.shape[ 1 ] )

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace( cf ) / float( np.sum( cf ) )

            # if it is a binary confusion matrix, show some more stats
            if len( cf ) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[ 1, 1 ] / sum( cf[ :, 1 ] )
                recall = cf[ 1, 1 ] / sum( cf[ 1, : ] )
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                        accuracy, precision, recall, f1_score
                        )
            else:
                # stats_text = "\n\nAccuracy={:0.3f}".format( accuracy )
                stats_text = ""
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get( 'figure.figsize' )

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure( figsize = figsize )
        g = sns.heatmap(
                cf, annot = box_labels, fmt = "", cmap = cmap, cbar = cbar, xticklabels = categories,
                yticklabels = categories,annot_kws={"size": 18},
                )
        g.set_yticklabels( g.get_yticklabels( ), rotation = 45, fontsize = 22 )
        g.set_xticklabels( g.get_xticklabels( ), rotation = 0, fontsize = 22 )
        if xyplotlabels:
            plt.ylabel( 'True label',fontsize=22  )
            plt.xlabel( 'Predicted label' + stats_text ,fontsize=22 )
        else:
            plt.xlabel( stats_text,fontsize=22 )

        if title:
            plt.title( title,fontsize = 20 )
    def pltCFMatrix( self,y,y_pred,figsize,title ):
        cf_matrix = confusion_matrix(y,y_pred)
        categories = [ 'P&P',
                     'Sweep',
                     'Clap',
                     'O',
                     'Zigzag',
                     'N']
        self.make_confusion_matrix(cf_matrix,categories = categories,figsize = figsize,title=title)
def showSignal(ori_sig,adv_sig,eps):
    # plt.figure( figsize = (8, 4.5) ).gca( )
    # ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    fig = plt.figure( figsize = (12, 12) )
    nrows = len(adv_sig)
    for i in range(nrows):
        ax = fig.add_subplot(nrows,1,i+1)
        ax.plot( ori_sig, label = 'original signal' )
        ax.plot( adv_sig[ i ], label = 'adversarial signal' )
        # ax.set_xlabel('Time',fontsize=12)
        ax.set_ylabel( 'Amplitude',fontsize=12 )
        # answer = round( eps[ i ], 2 )
        ax.set_title( f'ep = {round( eps[ i ], 2 ) }' )
        plt.legend( )
        # ax.set_xticks( 'time', fontsize = 17 )
        # ax.set_yticks( 'time', fontsize = 17 )
    plt.show()

# def showSpectrogram(ori,adv,eps):
def pltAttackPerform(info,label,title):
    ax = plt.figure( ).gca( )

    for i, v in enumerate(info):
        eps, acc = zip(*list(v))
        if label[i] == 'Nontargeted':
            ax.plot(np.asarray(eps),np.asarray(acc),label = label[i],marker = 'o')
            print(i)
        else:
            ax.plot( np.asarray(eps),np.asarray(acc), label = label[ i ] ,marker = 'o')
        ax.set_xlabel('EPS',fontsize=17)
        ax.set_ylabel('Accuracy',fontsize=17)
        plt.title(title)
    ax.legend(fontsize=12)
    plt.grid()
def pltcm(label_test_pred,true_label,title):
    pltcmatrix_Obj = pltConfusionMatrix()

    # label_test_pred = np.argmax( label_test_pred, axis = -1 ) + 1
    # true_label = np.argmax( true_label, axis = -1 ) + 1
    # Confusion Matrix
    cm = confusion_matrix( true_label, label_test_pred )
    # print( cm )
    # cm = cm.astype( 'float' ) / cm.sum( axis = 1 )[ :, np.newaxis ]
    # cm = np.around( cm, decimals = 2 )
    # print( cm )
    # Accuracy

    categories = [ 'P&P','Sweep','Clap','O','Zigzag','N' ]
    pltcmatrix_Obj.make_confusion_matrix(cm,categories = categories,figsize = (11,9),title=title)
def nonTargeted():
    eps = np.array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  , 1.05,
       1.1 , 1.15, 1.2 , 1.25, 1.3 , 1.35, 1.4 , 1.45, 1.5 , 1.55, 1.6 ,
       1.65, 1.7 , 1.75, 1.8 , 1.85, 1.9 , 1.95, 2.  , 2.05, 2.1 , 2.15,
       2.2 , 2.25, 2.3 , 2.35, 2.4 , 2.45, 2.5 , 2.55, 2.6 , 2.65, 2.7 ,
       2.75, 2.8 , 2.85, 2.9 , 2.95])
    eps_zscore = np.arange( 0.01, 0.2, 0.01 )
    # acc = np.array([0.9020979 , 0.88111888, 0.83916084, 0.81118881, 0.77622378,
    #    0.76923077, 0.72027972, 0.70629371, 0.6993007 , 0.68531469,
    #    0.67832168, 0.64335664, 0.62237762, 0.6013986 , 0.57342657,
    #    0.55244755, 0.55244755, 0.53846154, 0.53846154, 0.53846154,
    #    0.53846154, 0.53146853, 0.51748252, 0.51048951, 0.48951049,
    #    0.48251748, 0.46153846, 0.45454545, 0.44755245, 0.44055944,
    #    0.43356643, 0.43356643, 0.42657343, 0.42657343, 0.41258741,
    #    0.39160839, 0.38461538, 0.38461538, 0.38461538, 0.37762238,
    #    0.35664336, 0.35664336, 0.34965035, 0.34265734, 0.33566434,
    #    0.33566434, 0.33566434, 0.31468531, 0.30769231, 0.30769231,
    #    0.3006993 , 0.3006993 , 0.3006993 , 0.29370629, 0.28671329,
    #    0.27972028, 0.27972028, 0.27972028, 0.27972028, 0.27272727]) # widar_model_loc[2]_ori[2]Rx['r1', 'r2', 'r3',
    # 'r4', 'r5', 'r6'] -> user 1
    acc = np.array(
            [ 0.84027778, 0.47916667, 0.3125, 0.20138889, 0.13888889,
              0.09027778, 0.06944444, 0.0625, 0.04861111, 0.03472222,
              0.04166667, 0.03472222, 0.04166667, 0.04861111, 0.04166667,
              0.04166667, 0.04166667, 0.04166667, 0.02777778, 0.02777778 ]
            )
    widar = zip(eps_zscore,acc)
    # acc = np.array(
    #         [ 0.98277425, 0.92293744, 0.90299184, 0.8712602, 0.83771532,
    #           0.78513146, 0.72801451, 0.65548504, 0.57751587, 0.50861287,
    #           0.43427017, 0.37080689, 0.31912965, 0.26654578, 0.22846782,
    #           0.20580236, 0.18041704, 0.15956482, 0.13508613, 0.11604714,
    #           0.10698096, 0.09610154, 0.08068903, 0.06799637, 0.05711695,
    #           0.05258386, 0.04895739, 0.04533092, 0.04261106, 0.03898459,
    #           0.03717135, 0.03535811, 0.0344515, 0.03263826, 0.03082502,
    #           0.02901179, 0.02810517, 0.02447869, 0.02266546, 0.02357208,
    #           0.02357208, 0.02266546, 0.02357208, 0.02357208, 0.01903898,
    #           0.01813237, 0.01722575, 0.01722575, 0.01631913, 0.01631913,
    #           0.01541251, 0.01450589, 0.01359927, 0.01450589, 0.01359927,
    #           0.01359927, 0.01359927, 0.01359927, 0.01269266, 0.01178604 ]
    #         )
    acc = np.array([0.98460145, 0.91485507, 0.77264493, 0.59601449, 0.46376812,
       0.37137681, 0.29800725, 0.24547101, 0.20923913, 0.17481884,
       0.14764493, 0.12952899, 0.11684783, 0.09782609, 0.08152174,
       0.07065217, 0.0634058 , 0.05615942, 0.05344203, 0.04710145])
    signfi = zip(eps_zscore,acc)
    acc = np.array([0.92, 0.10, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
       0.06, 0.06, 0.06, 0.06, 0.06, 0.06,0.06,0.06, ])
    BVP = zip(eps_zscore,acc)
    pltAttackPerform(info = [widar,signfi,BVP],label = ['Widar raw CSI (Alexnet)','SignFi raw CSI (Alexnet)',
                                                        'Widar BVP (Temporal model)'
                                                    ],
            title = None)
def targeted():
    results = loadmat("resultsMat\\targeted-acc.mat")
    eps = np.squeeze(results['eps'])
    widar_target1 = np.squeeze(results['Targeted1'])
    widar_target2 = np.squeeze(results[ 'Targeted2' ])
    widar_target3 = np.squeeze(results[ 'Targeted3' ])
    widar_target4 = np.squeeze(results[ 'Targeted4' ])
    widar_target5 = np.squeeze(results[ 'Targeted5' ])
    widar_target6 = np.squeeze(results[ 'Targeted6' ])
    w1 = zip( eps, widar_target1 )
    w2 = zip( eps, widar_target2 )
    w3 = zip( eps, widar_target3 )
    w4 = zip( eps, widar_target4 )
    w5 = zip( eps, widar_target5 )
    w6 = zip( eps, widar_target6 )
    widar_nontarget = np.array(
            [ 0.84027778, 0.47916667, 0.3125, 0.20138889, 0.13888889,
              0.09027778, 0.06944444, 0.0625, 0.04861111, 0.03472222,
              0.04166667, 0.03472222, 0.04166667, 0.04861111, 0.04166667,
              0.04166667, 0.04166667, 0.04166667, 0.02777778, 0.02777778 ]
            )
    widar = zip( eps,widar_nontarget)
    # targeted
    # eps = np.arange( 0.0, 3, 0.1 )
    # # widar_target1 = np.array(
    # #         [ 0.89583333, 0.88194444, 0.84722222, 0.8125, 0.79166667,
    # #           0.77777778, 0.73611111, 0.70138889, 0.6875, 0.67361111,
    # #           0.65277778, 0.63194444, 0.61805556, 0.60416667, 0.59722222,
    # #           0.54861111, 0.53472222, 0.52777778, 0.52777778, 0.52777778,
    # #           0.52083333, 0.50694444, 0.5, 0.5, 0.5,
    # #           0.48611111, 0.47222222, 0.47222222, 0.45833333, 0.45833333 ]
    # #         )
    # w1 = zip( eps, widar_target1 )
    # widar_target2 = np.array(
    #         [ 0.89583333, 0.86111111, 0.82638889, 0.79861111, 0.77083333,
    #           0.75694444, 0.73611111, 0.70138889, 0.67361111, 0.66666667,
    #           0.65972222, 0.63888889, 0.61805556, 0.60416667, 0.59027778,
    #           0.56944444, 0.5625, 0.55555556, 0.52777778, 0.52083333,
    #           0.5, 0.47916667, 0.47222222, 0.47222222, 0.45833333,
    #           0.45833333, 0.45138889, 0.44444444, 0.44444444, 0.44444444 ]
    #         )
    # w2 = zip( eps, widar_target2 )
    # widar_target3 = np.array(
    #         [ 0.89583333, 0.86805556, 0.84027778, 0.79166667, 0.77083333,
    #           0.75694444, 0.72916667, 0.72222222, 0.70833333, 0.6875,
    #           0.67361111, 0.65972222, 0.65972222, 0.63194444, 0.625,
    #           0.61805556, 0.61111111, 0.61111111, 0.60416667, 0.57638889,
    #           0.5625, 0.5625, 0.54861111, 0.54861111, 0.54861111,
    #           0.54166667, 0.54166667, 0.54166667, 0.51388889, 0.5 ]
    #         )
    # w3 = zip( eps, widar_target3 )
    # widar_target4 = np.array(
    #         [ 0.89583333, 0.85416667, 0.80555556, 0.76388889, 0.72916667,
    #           0.71527778, 0.6875, 0.68055556, 0.67361111, 0.64583333,
    #           0.63194444, 0.61111111, 0.59722222, 0.58333333, 0.58333333,
    #           0.57638889, 0.56944444, 0.5625, 0.54861111, 0.54861111,
    #           0.54861111, 0.54166667, 0.54166667, 0.54166667, 0.53472222,
    #           0.52777778, 0.52083333, 0.51388889, 0.51388889, 0.51388889 ]
    #         )
    # w4 = zip( eps, widar_target4 )
    # widar_target5 = np.array(
    #         [ 0.89583333, 0.85416667, 0.8125, 0.78472222, 0.77083333,
    #           0.74305556, 0.71527778, 0.70138889, 0.68055556, 0.66666667,
    #           0.65972222, 0.64583333, 0.64583333, 0.625, 0.61805556,
    #           0.61111111, 0.61111111, 0.60416667, 0.59027778, 0.58333333,
    #           0.57638889, 0.5625, 0.55555556, 0.54861111, 0.54166667,
    #           0.53472222, 0.52777778, 0.51388889, 0.5, 0.49305556 ]
    #         )
    # w5 = zip( eps, widar_target5 )
    # widar_target6 = np.array(
    #         [ 0.89583333, 0.875, 0.84027778, 0.77777778, 0.75694444,
    #           0.72916667, 0.70833333, 0.70833333, 0.70833333, 0.68055556,
    #           0.66666667, 0.61805556, 0.60416667, 0.59722222, 0.59027778,
    #           0.59027778, 0.5625, 0.5625, 0.55555556, 0.55555556,
    #           0.52777778, 0.52083333, 0.50694444, 0.50694444, 0.50694444,
    #           0.5, 0.49305556, 0.48611111, 0.47916667, 0.46527778 ]
    #         )
    # w6 = zip( eps, widar_target6 )
    # eps = np.array(
    #         [ 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05,
    #           1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
    #           1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2., 2.05, 2.1, 2.15,
    #           2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7,
    #           2.75, 2.8, 2.85, 2.9, 2.95 ]
    #         )
    # acc = np.array(
    #         [ 0.9020979, 0.88111888, 0.83916084, 0.81118881, 0.77622378,
    #           0.76923077, 0.72027972, 0.70629371, 0.6993007, 0.68531469,
    #           0.67832168, 0.64335664, 0.62237762, 0.6013986, 0.57342657,
    #           0.55244755, 0.55244755, 0.53846154, 0.53846154, 0.53846154,
    #           0.53846154, 0.53146853, 0.51748252, 0.51048951, 0.48951049,
    #           0.48251748, 0.46153846, 0.45454545, 0.44755245, 0.44055944,
    #           0.43356643, 0.43356643, 0.42657343, 0.42657343, 0.41258741,
    #           0.39160839, 0.38461538, 0.38461538, 0.38461538, 0.37762238,
    #           0.35664336, 0.35664336, 0.34965035, 0.34265734, 0.33566434,
    #           0.33566434, 0.33566434, 0.31468531, 0.30769231, 0.30769231,
    #           0.3006993, 0.3006993, 0.3006993, 0.29370629, 0.28671329,
    #           0.27972028, 0.27972028, 0.27972028, 0.27972028, 0.27272727 ]
    #         )
    # widar = zip( eps, acc )
    pltAttackPerform(
            info = [ w1, w2, w3, w4, w5, w6, widar ],
            label = [ 'Target 1', 'Target 2', 'Target 3', 'Target 4', 'Target 5',
                      'Target 6', 'Nontargeted' ],
            title = None
            )
def PSRPerform():
    '''The PSR param performance under differnt range of input'''
    a = loadmat(pjoin(result_dir,"PSR_ACC.mat"))
    range5 = np.squeeze(a[list( a.keys( ) )[ 3 ]])
    range10 = np.squeeze(a[list( a.keys( ) )[ 4 ]])
    range20 = np.squeeze(a[list( a.keys( ) )[ 5 ]])
    range30 = np.squeeze(a[list( a.keys( ) )[ 6 ]])
    range40 = np.squeeze(a[list( a.keys( ) )[ 7 ]])
    range50 = np.squeeze(a[list( a.keys( ) )[ 8 ]])
    eps = np.squeeze(a[list( a.keys( ) )[ 9 ]])
    r5 = zip(eps,range5)
    r10 = zip( eps, range10 )
    r20 = zip( eps, range20 )
    r30 = zip( eps, range30 )
    r40 = zip( eps, range40 )
    r50 = zip( eps, range50 )
    pltAttackPerform(info = [r5,r10,r20,r30,r40,r50],label = ['range 0 - 5','range 0 - 10','range 0 - 20','range 0 - '
                                                                                                        '30',
                                                              'range 0 - 40','range 0 - 50'],title = None)
def EPSPerform():
    a = loadmat(pjoin(result_dir,"EPS_ACC.mat"))
    range5 = np.squeeze(a['range5'])
    range10 = np.squeeze(a['range10'])
    range20 = np.squeeze(a['range20'])
    range30 = np.squeeze(a['range30'])
    range40 = np.squeeze(a['range40'])
    range50 = np.squeeze(a['range50'])
    eps = np.squeeze(a['eps'])
    r5 = zip(eps,range5)
    r10 = zip( eps, range10 )
    r20 = zip( eps, range20 )
    r30 = zip( eps, range30 )
    r40 = zip( eps, range40 )
    r50 = zip( eps, range50 )
    pltAttackPerform(info = [r5,r10,r20,r30,r40,r50],label = ['range 0 - 5','range 0 - 10','range 0 - 20','range 0 - '
                                                                                                        '30',
                                                              'range 0 - 40','range 0 - 50'
                                                              ],title = None)
if __name__ == '__main__':
    EPSPerform()