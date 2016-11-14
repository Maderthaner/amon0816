import matplotlib.pyplot as plt
import numpy as np
from operator import truediv
from pylab import figure, cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

from mpl_toolkits.mplot3d import Axes3D

# parameter stuff
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("runs", help="psana run numer",type=int,nargs='+')
args = parser.parse_args()

#Runss = '5354'
runs = np.array([args.runs],dtype='int').flatten()
Runss = str(runs).replace(' ','')
Nruns = len(runs)
#Nruns = 2
xLength = 839 #rotated camera 798
#xbins = Nruns*xLength
#ybins = 50
#ybinsDelay = 40

div = []

Plot1 = np.load('spectra/RatiosXTCAV_%s.npy' % Runss)
Plot2 = np.load('spectra/TotalPowXTCAV_%s.npy' % Runss)
Plot3 = np.load('spectra/RatiosiTOF_%s.npy' % Runss)
Plot4 = np.load('spectra/TotalPowiTOF_%s.npy' % Runss)

Plot5 = np.load('spectra/RelYieldsXTCAV_%s.npy' % Runss)
Plot6 = np.load('spectra/RelYieldsiTOF_%s.npy' % Runss)

Plot7 = np.load('spectra/CorrelationRatio_%s.npy' % Runss)
Plot8 = np.load('spectra/CorrelationPow_%s.npy' % Runss)

counts = np.load('spectra/NormRuns_%s.npy' % Runss)
PhotonCounts = np.load('spectra/PhotonECounter_%s.npy' % Runss)
XTCAVCounts = np.load('spectra/XTCAVCounter_%s.npy' % Runss)
FeeCounts = np.load('spectra/FEEGasCounter_%s.npy' % Runss)

plt.subplot(2,1,1)
plt.imshow(Plot5,norm=LogNorm())

plt.subplot(2,1,2)
plt.imshow(Plot6,norm=LogNorm())

#plt.subplot(4,1,1)
#plt.imshow(Plot1)

#plt.subplot(4,1,2)
#plt.imshow(Plot2)

#plt.subplot(4,1,3)
#plt.imshow(Plot3)

#plt.subplot(4,1,4)
#plt.imshow(Plot4)

plt.show()
