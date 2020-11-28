#a12Script
import a12 as vm
import numpy as np
import glob
import imageIO as io





v=np.load('Input/baby.npy')
#try out different parameters for different videos
#out = vm.videoMag(v, 10, 20, 4, 40, 40)
out = vm.videoMag(v, 10, 20, 4, 80, 2)
vm.writeFrames(out, "baby/frame")
