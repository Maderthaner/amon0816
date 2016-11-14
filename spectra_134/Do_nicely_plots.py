import matplotlib.pyplot as plt
import numpy as np

Nruns = 1
Runss = '134'
xLength = 520
xbins = Nruns*xLength
ybins = 50
ybinsDelay = 40

Plot1 = np.load('PhotonSpectra_[%s].npy' % Runss)
Plot2 = np.load('combinedSpectra_[%s].npy' % Runss)
Plot3 = np.load('XTCAVSpectra_[%s].npy' % Runss)
plt.subplot(2,1,1)
plt.imshow(Plot1,extent=[0,xbins,0,ybins],aspect='auto')
plt.subplot(2,1,2)
plt.plot(Plot2)
#plt.subplot(3,1,3)
#plt.imshow(Plot3,extent=[0,xbins,0,ybinsDelay],aspect='auto')
plt.show()


#t = np.arange(0.0, 2.0, 0.01)
#s = np.sin(2*np.pi*t)
#plt.plot(t, s)

#plt.xlabel('time (s)')
#plt.ylabel('voltage (mV)')
#plt.title('About as simple as it gets, folks')
#plt.grid(True)
#plt.savefig("test.png")
#plt.show()
