import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from scipy.io import savemat, loadmat
import scipy.io as sio
from os.path import dirname, join as pjoin
import os
result_dir = os.getcwd() + '/resultsMat'
class pltConfusionMatrix():
    def __init__( self ):
        pass
    def make_confusion_matrix(self,
            cf,
            group_names = None,
            categories = 'auto',
            count = False,
            percent = True,
            cbar = False,
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
        group_names:   List of strings that represent the labels_pred row by row to be shown in each square.
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
            # group_percentages = [ "{0:.2f}".format( value ) for value in cf.flatten( ) / np.sum( cf ) ]

            group_percentages = [ "{0:.2f}".format( value ) for value in (cf / np.expand_dims( cf.sum( axis = 1 ),
                    axis = 1 )).flatten() ]
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
        ax.plot( label='original signal' )
        ax.plot( label='adversarial signal' )
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
            ax.plot( label=label[ i ], marker='o' )
            print(i)
        else:
            ax.plot( label=label[ i ], marker='o' )
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
def PSRPerform():
    '''The PSR param performance under differnt range of input'''
    a = loadmat(pjoin(result_dir,"PSR_ACC_signfi.mat"))
    range5 = np.squeeze(a['range5'])
    range10 = np.squeeze(a['range10'])
    range20 = np.squeeze(a['range20'])
    range30 = np.squeeze(a['range30'])
    range40 = np.squeeze(a['range40'])
    range50 = np.squeeze(a['range50'])
    eps = np.squeeze(a['PSR'])
    range500 = np.squeeze(a['range500'])
    r5 = zip(eps,range5)
    r10 = zip( eps, range10 )
    r20 = zip( eps, range20 )
    r30 = zip( eps, range30 )
    r40 = zip( eps, range40 )
    r50 = zip( eps, range50 )
    r500 = zip( eps, range500 )
    pltAttackPerform(info = [r5,r10,r20,r30,r40,r50,r500],label = ['range 0 - 5','range 0 - 10','range 0 - 20','range 0 - '
                                                                                                        '30',
                                                              'range 0 - 40','range 0 - 50','range 0 - 500'],
            title = None)
def EPSPerform():
    a = loadmat(pjoin(result_dir,"EPS_ACC_signfi.mat"))
    range5 = np.squeeze(a['range5'])
    range10 = np.squeeze(a['range10'])
    range20 = np.squeeze(a['range20'])
    range30 = np.squeeze(a['range30'])
    range40 = np.squeeze(a['range40'])
    range50 = np.squeeze(a['range50'])
    range500 = np.squeeze( a[ 'range500' ] )
    eps = np.squeeze(a['eps'])
    r5 = zip(eps,range5)
    r10 = zip( eps, range10 )
    r20 = zip( eps, range20 )
    r30 = zip( eps, range30 )
    r40 = zip( eps, range40 )
    r50 = zip( eps, range50 )
    r500 = zip( eps, range500 )
    pltAttackPerform(info = [r5,r10,r20,r30,r40,r50,r500],label = ['range 0 - 5','range 0 - 10','range 0 - 20','range 0 - '
                                                                                                        '30',
                                                              'range 0 - 40','range 0 - 50','range 0 - 500',
                                                              ],title = None)
def zeroPerform():
    a = loadmat(pjoin(result_dir,'ori_zscore_var.mat'),squeeze_me = True)
    original = a[ 'original' ]
    zscore = a[ 'zscore' ]
    eps = np.arange( 0, 0.02, 0.002 )
    ori = zip(eps,original)
    z = zip(eps,zscore)
    pltAttackPerform(info = [ori,z],label = ['original','zscore'],
            title = None)
if __name__ == '__main__':
    print('')