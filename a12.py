import imageIO as io
import os
import cv2
from tqdm import tqdm
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy.fftpack import fft, fftshift
import skimage.filters as sk_filters
import glob
import matplotlib.pyplot as plt

io.baseInputPath = './'

def getPNGsInDir(path):
    '''gets the png's in a folder and puts them in out.'''
    fnames = glob.glob(path+"*.png")
    out=[]
    for f in fnames:
        #print f
        imi = io.imread(f)
        out.append(imi)
    return out

def convertToNPY(path, pathOut):
    '''converts the png images in a path path to a npy file at pathOut'''
    L=getPNGsInDir(path)
    V=np.array(L)
    np.save(pathOut, V)

def writeFrames(video, folder, framerate=30):
    '''writes the frames of video to path.'''
    # make the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    nFrame=video.shape[0]
    for i in tqdm(range(nFrame)):
        image_path = os.path.join(folder, "frame_%03d.png" % i)
        io.imwrite(video[i], image_path)
    print('Saved images to {}'.format(folder))
    # now save the video
    os.system("ffmpeg -y -r {} -i {} -vcodec mpeg4 {}.mp4".format(
        framerate,
        os.path.join(folder, "frame_%03d.png"),
        folder
    ))
    print("Saved video to {}.mp4".format(folder))

def RGB2YUV(video):
    '''Convert an RGB video to YUV.'''
    RGB2YUVmatrix=np.transpose([[0.299,  0.587,  0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]])
    return np.dot(video[:, :, :], RGB2YUVmatrix)


def YUV2RGB(video):
    '''Convert an YUV video to RGB.'''
    YUV2RGBmatrix=np.transpose([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    return np.dot(video[:, :, :], YUV2RGBmatrix)

##################### Write the functions below ##############


def lowPass(video, sigma):
    '''This should low pass the spatial frequencies of your video using a gaussian filter with sigma given as the second input parameter.'''
    # filtered_video = sk_filters.gaussian(video, sigma=sigma)
    # window = signal.gaussian(11, std=sigma)
    # window = window / window.sum()
    # filtered_video = ndimage.convolve1d(filtered_video, window, axis=1)
    # conv_high = ndimage.convolve1d(filtered_video, window_high / window_high.sum(), axis=0)

    # filtered_video = ndimage.gaussian_filter(video, sigma=sigma)
    filtered_video = np.copy(video)
    for i in range(len(video)):
        filtered_video[i] = cv2.GaussianBlur(video[i], (7, 7), sigma)
    return filtered_video


def timeBandPass(video, sigmaTlow, sigmaThigh):
    '''Apply a band pass filter to the time dimension of the video.
    Your band passed signal should be the difference between two gaussian
    filtered versions of the signal.
    '''
    
    output_video = np.copy(video)
    for c in [0,1,2]: # the channels
        filtered_video = video[:,:,:,c]
        t, h, w = filtered_video.shape
        filtered_video = filtered_video.reshape((t, -1))

        window_low = signal.gaussian(20, std=sigmaTlow)
        window_high = signal.gaussian(20, std=sigmaThigh)

        conv_low = ndimage.convolve1d(filtered_video, window_low / window_low.sum(), axis=0).reshape((t, h, w))
        conv_high = ndimage.convolve1d(filtered_video, window_high / window_high.sum(), axis=0).reshape((t, h, w))

        # difference
        filtered_video = conv_high - conv_low

        output_video[:,:,:,c] = filtered_video

    return output_video 

def videoMag(video, sigmaS, sigmaTlow, sigmaThigh, alphaY, alphaUV):
    '''returns the motion magnified video. sigmaS is the sigma for your
    spatial blur. sigmaTlow is the sigma for the larger temporal gaussian,
    sigmaThigh is the sigma for the smaller temporal gaussian. alphaY is
    how much of the bandpassed Y signal you should add back to the video.
    alphaUV is how much of the bandpassed UV signal you should add back to
    the video.
    You should use lowPass() to apply your spatial filter and timeBandPass()
    to apply your time filter.'''
    filtered_video = RGB2YUV(video)
    lowpass_video = lowPass(filtered_video, sigmaS)
    bandpass_video = timeBandPass(lowpass_video, sigmaTlow, sigmaThigh)
    filtered_video[:,:,:,0] = filtered_video[:,:,:,0] + bandpass_video[:,:,:,0] * alphaY
    filtered_video[:,:,:,1:3] = filtered_video[:,:,:,1:3] + bandpass_video[:,:,:,1:3] * alphaUV
    filtered_video = YUV2RGB(filtered_video)
    return filtered_video


def timeBandPassButter(video, low, high, order):
    '''    
    B,A = signal.butter(order, [low, high], 'bandpass')
    gives the coefficients used in the butterworth iir filter.
    for a input signal x, the filtered output signal is given
    by the recursion relationship:
    
    A[0]*y[n]= -A[1]*y[n-1]
               -A[2]*y[n-2]
               -A[3]*y[n-3]
                 ...(up to the number of coefficients, which depends on 'order')
               +B[0]*x[n]
               +B[1]*x[n-1]
               +B[2]*x[n-2]
               +B[3]*x[n-2]
               ...(up to the number of coefficients, which depends on 'order')
    '''

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    # pass

    output_video = np.copy(video)
    for c in [0,1,2]: # the channels
        filtered_video = video[:,:,:,c]
        t, h, w = filtered_video.shape
        filtered_video = filtered_video.reshape((t, -1))

        b, a = signal.butter(order, [low, high], btype="bandpass")
        filtered_video = signal.filtfilt(b, a, filtered_video, axis=0)
        filtered_video = filtered_video.reshape((t, h, w))

        output_video[:,:,:,c] = filtered_video

    return output_video 



def videoMagButter(video, sigmaS, low, high, order, alphaY, alphaUV):
    '''video magnification using the butterworth iir filter to
    bandpass the signal over time instead of 
    '''
    filtered_video = RGB2YUV(video)
    lowpass_video = lowPass(filtered_video, sigmaS)
    bandpass_video = timeBandPassButter(lowpass_video, low, high, order)
    filtered_video[:,:,:,0] = filtered_video[:,:,:,0] + bandpass_video[:,:,:,0] * alphaY
    filtered_video[:,:,:,1:3] = filtered_video[:,:,:,1:3] + bandpass_video[:,:,:,1:3] * alphaUV
    filtered_video = YUV2RGB(filtered_video)
    return filtered_video

def main():
    print('Yay for computational photography!')
	#convertToNPY('face/face', 'face.npy')
	#return
	#v=np.load('Input/face.npy')
	#print '    done loading input file, size: ', v.shape
	#out=videoMag(v, 10, 20, 4, 40, 40)
    #writeFrames(out, 'videoOut/frame')


#### test cases ####
def testLowPass(video):
    # select a frame
    originalframe = video[5]
    sigma = 1.0
    filtered_video = lowPass(video, sigma)
    newframe = filtered_video[5]
    print("before -> after")
    plt.figure(figsize=(10,10))
    plt.imshow(np.hstack([originalframe, newframe]))
    plt.show()

def testTemporalIntensity(videocopy):

    c = 0
    sigmaS = 10.0
    sigmaTlow = 5.0
    sigmaThigh = 1.0

    video = np.copy(videocopy)
    # spatial low pass
    video = lowPass(video, sigmaS)


    output_video = np.copy(video)

    filtered_video = video[:,:,:,c] # only the Y
    t, h, w = filtered_video.shape
    filtered_video = filtered_video.reshape((t, -1))

    window_low = signal.gaussian(20, std=sigmaTlow)
    window_high = signal.gaussian(20, std=sigmaThigh)

    conv_low = ndimage.convolve1d(filtered_video, window_low / window_low.sum(), axis=0).reshape((t, h, w))
    conv_high = ndimage.convolve1d(filtered_video, window_high / window_high.sum(), axis=0).reshape((t, h, w))

    

    # difference
    filtered_video = conv_high - conv_low

    output_video[:,:,:,c] = filtered_video

    # choose a spatial position
    x = 200
    y = 150

    times = np.arange(0, len(filtered_video))[:30]
    plt.plot(times, video[times, x, y, c], label="original")
    plt.plot(times, conv_low[times, x, y], label="low")
    plt.plot(times, conv_high[times, x, y], label="high")
    # plt.plot(times, output_video[times, x, y, c], label="filtered")
    plt.legend()
    plt.ylabel("intensity")
    plt.xlabel("time")
    plt.show()

    plt.plot(times, output_video[times, x, y, c], label="Gaussian difference")
    plt.legend()
    plt.ylabel("intensity")
    plt.xlabel("time")
    plt.show()

    

def testYvsUV(video):
    
    FRAMERATE = 30.0

    sigmaS = 1.0 # spatial low pass
    sigmaTlow = 5.0 # temporal larger gaussian
    sigmaThigh = 1.0 # temporal smaller gaussian

    alphaY = 20.0
    alphaUV = 0.0
    final_video = videoMag(video, sigmaS, sigmaTlow, sigmaThigh, alphaY, alphaUV)
    writeFrames(final_video, os.path.join("Output", "testYvsUV_Yonly", "images"), framerate=FRAMERATE)

    alphaY = 0.0
    alphaUV = 20.0
    final_video = videoMag(video, sigmaS, sigmaTlow, sigmaThigh, alphaY, alphaUV)
    writeFrames(final_video, os.path.join("Output", "testYvsUV_UVonly", "images"), framerate=FRAMERATE)

    alphaY = 20.0
    alphaUV = 20.0
    final_video = videoMag(video, sigmaS, sigmaTlow, sigmaThigh, alphaY, alphaUV)
    writeFrames(final_video, os.path.join("Output", "testYvsUV_both", "images"), framerate=FRAMERATE)

def testDifferentOrders(video):
    FRAMERATE = 30

    sigmaS = 1.0 # spatial low pass
    lowcut = 0.4 # in Hz
    highcut = 5.0
    nyq = 0.5 * FRAMERATE
    low = lowcut / nyq
    high = highcut / nyq

    alphaY = 20.0
    alphaUV = 20.0


    order = 2
    final_video = videoMagButter(video, sigmaS, low, high, order, alphaY, alphaUV)
    writeFrames(final_video, os.path.join("Output", "testDifferentOrders_2", "images"), framerate=FRAMERATE)

    order = 4
    final_video = videoMagButter(video, sigmaS, low, high, order, alphaY, alphaUV)
    writeFrames(final_video, os.path.join("Output", "testDifferentOrders_4", "images"), framerate=FRAMERATE)

def testGaussianvsButter(videocopy):

    c = 0
    sigmaS = 10.0
    sigmaTlow = 5.0
    sigmaThigh = 1.0
    FRAMERATE = 30.0
    order = 4

    video = np.copy(videocopy)
    # spatial low pass
    video = lowPass(video, sigmaS)


    output_video = np.copy(video)

    filtered_video = video[:,:,:,c] # only the Y
    t, h, w = filtered_video.shape
    filtered_video = filtered_video.reshape((t, -1))

    # --- BUTTER
    lowcut = 0.4 # in Hz
    highcut = 5.0
    nyq = 0.5 * FRAMERATE
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    butter_filtered_video = signal.filtfilt(b, a, filtered_video, axis=0).reshape((t, h, w))
    # ---

    window_low = signal.gaussian(20, std=sigmaTlow)
    window_high = signal.gaussian(20, std=sigmaThigh)

    conv_low = ndimage.convolve1d(filtered_video, window_low / window_low.sum(), axis=0).reshape((t, h, w))
    conv_high = ndimage.convolve1d(filtered_video, window_high / window_high.sum(), axis=0).reshape((t, h, w))

    

    # difference
    filtered_video = conv_high - conv_low

    output_video[:,:,:,c] = filtered_video

    output_video[:,:,:,c] = filtered_video



    # choose a spatial position
    x = 200
    y = 150

    times = np.arange(0, len(filtered_video))[:30]
    plt.plot(times, output_video[times, x, y, c], label="Gaussian difference")
    plt.plot(times, butter_filtered_video[times, x, y], label="butter")
    plt.legend()
    plt.ylabel("intensity")
    plt.xlabel("time")
    plt.show()


#the usual Python module business
if __name__ == '__main__':
    main()
