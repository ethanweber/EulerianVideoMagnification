import imageIO as io
import os
from tqdm import tqdm
import numpy as np
from scipy import ndimage
from scipy import signal
import glob

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



def timeBandPass(video, sigmaTlow, sigmaThigh):
    '''Apply a band pass filter to the time dimension of the video.
    Your band passed signal should be the difference between two gaussian
    filtered versions of the signal.
    '''
    

def videoMag(video, sigmaS, sigmaTlow, sigmaThigh, alphaY, alphaUV):
    '''returns the motion magnified video. sigmaS is the sigma for your
    spatial blur. sigmaTlow is the sigma for the larger termporal gaussian,
    sigmaThigh is the sigma for the smaller temporal gaussian. alphaY is
    how much of the bandpassed Y signal you should add back to the video.
    alphaUV is how much of the bandpassed UV signal you should aff back to
    the video.
    You should use lowPass() to apply your spatial filter and timeBandPass()
    to apply your time filter.'''

    


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


def videoMagButter(video, sigmaS, low, high, order, alphaY, alphaUV):
    '''video magnification using the butterworth iir filter to
    bandpass the signal over time instead of 
    '''

def main():
    print('Yay for computational photography!')
	#convertToNPY('face/face', 'face.npy')
	#return
	#v=np.load('Input/face.npy')
	#print '    done loading input file, size: ', v.shape
	#out=videoMag(v, 10, 20, 4, 40, 40)
    #writeFrames(out, 'videoOut/frame')

#the usual Python module business
if __name__ == '__main__':
    main()
