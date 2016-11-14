import matplotlib.pyplot as plt
import numpy as np
from operator import truediv
from pylab import figure, cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

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

Plot1 = np.load('spectra/PhotonSpectra_%s.npy' % Runss)
Plot2 = np.load('spectra/combinedSpectra_%s.npy' % Runss)
Plot3 = np.load('spectra/XTCAVSpectra_%s.npy' % Runss)
Plot4 = np.load('spectra/FeeGasSpectra_%s.npy' % Runss)

counts = np.load('spectra/NormRuns_%s.npy' % Runss)
PhotonCounts = np.load('spectra/PhotonECounter_%s.npy' % Runss)
XTCAVCounts = np.load('spectra/XTCAVCounter_%s.npy' % Runss)
FeeCounts = np.load('spectra/FEEGasCounter_%s.npy' % Runss)

fig = plt.figure()





################################# Plotting

# 1 PhotonE vs Electron E
div2d = []
for j in PhotonCounts:
    div2d.append(np.full(xLength,j))

divme = np.array(div2d).reshape(Plot1.shape)
divme[divme==0.]=1
dived = map(truediv,Plot1,divme)
plotme = dived/np.amax(dived)
plotme[plotme==0]=0.001

plt.subplot(4,1,1)
plt.title('PhotonE vs Electron Spectrum')
ybins = Plot1.shape[0]
xbins = Plot1.shape[1]

#plt.plot_surface(Plot1)
plt.imshow(Plot1,extent=[0,xbins,0,ybins],aspect='auto',norm=Normalize(vmin=0.,vmax=5.))
#plt.imshow(plotme,extent=[0,xbins,0,ybins],aspect='auto',interpolation='none',norm=LogNorm(vmin=0.001,vmax=1))


# 2  Electron spectrum
for x in counts:
    div.append(np.full(xLength,x))

plt.subplot(4,1,2)
#plt.title('Electron Spectrum')
xbins = Plot2.shape[0]
ax = range(xbins)
ax2 = range(xbins)
#plt.p:lot(ax,map(truediv,Plot2,np.array(div).flatten()))
#plt.plot(ax,map(truediv,Plot2,np.array(div).flatten()))
#plt.plot(ax2,map(truediv,Plot1[122],np.array(div).flatten()))
plt.plot(ax2,Plot2)
plt.plot(np.sum(ax2,Plot1[152:160],axis=1)*50)
plt.xlim([0,xbins])


# 3 Pulse E vs Electron E
div2d3=[]
for k in FeeCounts:
    div2d3.append(np.full(xLength,k))

divme3 = np.array(div2d3).reshape(Plot4.shape)
divme3[divme3==0.]=1
dived3 = map(truediv,Plot4,divme3)
plotme3 = dived3/np.amax(dived3)
plotme3[plotme3==0]=0.001

plt.subplot(4,1,3)
ybins = Plot4.shape[0]
xbins = Plot4.shape[1]
plt.imshow(plotme3,extent=[0,xbins,0,ybins],aspect='auto',interpolation='none',norm=LogNorm(vmin=0.001,vmax=1))


#4 XTCAV
div2d2=[]
for i in XTCAVCounts:
    div2d2.append(np.full(xLength,i))

divme2 = np.array(div2d2).reshape(Plot3.shape)
divme2[divme2==0.]=1
dived2 = map(truediv,Plot3,divme2)
plotme2 = dived2/np.amax(dived2)
plotme2[plotme2==0]=0.001

plt.subplot(4,1,4)
ybins = Plot3.shape[0]
xbins = Plot3.shape[1]
plt.imshow(plotme2,extent=[0,xbins,0,ybins],aspect='auto',interpolation='none',norm=LogNorm(vmin=0.001,vmax=1))

plt.show()
