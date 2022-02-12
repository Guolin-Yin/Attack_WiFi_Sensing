from numpy.fft import fft, fftfreq, ifft
from scipy import signal
from scipy.signal import butter, lfilter,savgol_filter
from sklearn.preprocessing import StandardScaler
# from skimage.restoration import denoise_wavelet
from scipy.ndimage import median_filter
# from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
class Denoiser:
    def __init__(self,):
        pass
    def phaseSanitizer( self, rawPhase ):
        k = (rawPhase[ 29 ] - rawPhase[ 0 ]) / (28 - (-28))
        b = np.mean( rawPhase, axis = 0 )
        m_i = np.array([-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,
                    28],dtype = np.float64)
        lin = k * m_i + b
        caliPhase = rawPhase - lin
        return caliPhase
    def csiRatio( self, csi = None, isWidar:bool = False,data_amp = None,data_phase = None):
        if isWidar:
            real = data_amp * np.cos( data_phase )
            imag = data_amp * np.sin( data_phase )
            csi = real + 1j * imag
            csi = csi - np.expand_dims( np.mean( csi, axis = 1 ), axis = 1 )
            ratio_1 = np.expand_dims( csi[  :, :, 0 ] / csi[ :, :, 1 ], axis = 2 )
            ratio_2 = np.expand_dims( csi[  :, :, 1 ] / csi[ :, :, 2 ], axis = 2 )
            ratio_3 = np.expand_dims( csi[  :, :, 0 ] / csi[ :, :, 2 ], axis = 2 )
            csi_ratio = np.concatenate( (ratio_1, ratio_2, ratio_3), axis = 2 )
            # csi_ratio = stats.zscore( csi_ratio, axis = 1 )
            x_amp = stats.zscore( np.abs( csi_ratio ), axis = 0 )
            # angle = stats.zscore(np.angle( csi_ratio ), axis = 1 )
            angle = np.angle( csi_ratio )
            x_phase = angle - angle[ 0, :, : ]
        else:
            csi = csi - np.expand_dims(np.mean(csi, axis = 1),axis = 1)
            ratio_1 = np.expand_dims(csi[ :, :, :, 0 ] / csi[ :, :, :, 1 ],axis=3)
            ratio_2 = np.expand_dims(csi[ :, :, :, 1 ] / csi[ :, :, :, 2 ],axis=3)
            ratio_3 = np.expand_dims(csi[ :, :, :, 0 ] / csi[ :, :, :, 2 ],axis=3)
            csi_ratio = np.concatenate( (ratio_1, ratio_2, ratio_3), axis = 3 )
            # csi_ratio = stats.zscore( csi_ratio, axis = 1 )
            x_amp = stats.zscore(np.abs( csi_ratio ), axis = 1 )
            # angle = stats.zscore(np.angle( csi_ratio ), axis = 1 )
            angle = np.angle( csi_ratio )
            x_phase = angle - np.expand_dims(angle[:,0,:,:],axis = 1)

        return [x_amp,x_phase]
def get_median_dnData(data,size,mode):
    buf = []
    if mode == 'array':
        for i in range(data.shape[0]):
            dn_data = median_filter(input=data[i],size = size)
            buf.append(dn_data)
    if mode == 'single':
        for i in range(data.shape[0]):
            buf_1 = []
            for j in range(data.shape[1]):
                dn_data = median_filter(input=data[i][j,:],size = size)
                buf_1.append(dn_data)
            buf.append(np.asarray(buf_1))
    return np.asarray(buf)
def denoising(data,order,cf_freq, i_plt_subcarrier, plot,t = 2.0):
    fs = data.shape[1]/t
    sos = signal.butter( order, cf_freq, 'lowpass', fs=fs, output='sos' )
    filtered_signal = signal.sosfilt( sos, data )
    if plot:
        n_samples = int(t * fs)
        x_axis = np.linspace( 0, 2, n_samples )
        # FFT for original Signal
        fft_original = fft( data[i_plt_subcarrier] )
        fft_original = 2.0 * np.abs( fft_original / n_samples )
        freq_original = fftfreq( n_samples, d=2.0 / n_samples )
        mask = freq_original > 0
        # FFT for denoised Signal
        fft_filtered = fft( filtered_signal[i_plt_subcarrier] )
        fft_filtered = 2.0 * np.abs( fft_filtered / n_samples )
        freq_filtered = fftfreq( n_samples, d=2.0 / n_samples )

        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots( 2, 2 )
        # Original signal
        ax1.plot(x_axis,data[i_plt_subcarrier])
        ax1.set_title('Oringinal signal')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency in [Hz]')
        # Spectrum of original Signal
        ax2.plot( freq_original[mask], fft_original[mask] )
        ax2.set_title( 'Spectrum Oringinal signal' )
        ax2.set_xlabel( 'Frequency' )
        ax2.set_ylabel( 'Amplitude' )
        # Filted signal
        ax3.plot( x_axis, filtered_signal[i_plt_subcarrier] )
        ax3.set_title( 'Denoised Signal' )
        ax3.set_xlabel( 'Time' )
        ax3.set_ylabel( 'Frequency in [Hz]' )
        # Spectrum of Filted Signal
        ax4.plot( freq_filtered[mask], fft_filtered[mask] )
        ax4.set_title( 'Spectrum Filted signal' )
        ax4.set_xlabel( 'Frequency' )
        ax4.set_ylabel( 'Amplitude' )
        fig.show()
    return filtered_signal
def get_deonised_data(data,cf_freq:int):
    # denoised_data = []
    # for i in range(len(data)):
    # buf = denoising(data=data,order = 20, cf_freq = cf_freq, i_plt_subcarrier = 2, plot = False)
    # denoised_data.append(buf)
    return np.asarray(denoising(data=data,order = 20, cf_freq = cf_freq, i_plt_subcarrier = 2, plot = False))
def remove_zero(data):
    # out = []
    # for i in range(data.shape[0]):
    buff = []
    for j in range(data.shape[0]):
        to_fft = data[j,:]
        fft_vals = fft( to_fft )
        fft_vals[ 0 ] = 0
        re = ifft( fft_vals )
        buff.append(re)
    # out.append(np.asarray(buff))
    return np.real(np.asarray(buff))
def standardisers( data ):

    # for i in range( len( data ) ):
    scaler = StandardScaler( ).fit( data )
    X_scaler = scaler.transform( data )

    return X_scaler
def pca_denoise(data:np.array, n_comp ):
    processed_data = []
    PCA_object = []
    # for i in range(data.shape[0]):
    pca_data = PCA( n_components=n_comp )
    pca_data.fit( data )
    processed_data.append(pca_data.components_)
    PCA_object.append(pca_data)
    return pca_data.components_,PCA_object
def get_preprocessed_data(data,reduction:bool,n_comp:int = 3):
    sd_Xd = standardisers( data=data )
    zero_re = remove_zero( data=sd_Xd )
    dn_Xd = get_deonised_data( zero_re, cf_freq=100 )
    dn_Xd1 = get_median_dnData( data=dn_Xd, size=7, mode="array" )
    if reduction == True:
        pcad_data1, _ = pca_denoise( data=dn_Xd1, n_comp=n_comp )
        return np.asarray(pcad_data1)
    if reduction == False:
        return np.asarray(dn_Xd1)
if __name__ == "__main__":
    d = Denoiser()