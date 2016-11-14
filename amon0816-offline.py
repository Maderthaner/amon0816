##### Useful commands
# source /reg/g/psdm/etc/ana_env.sh
# mpirun -n 40 --host daq-amo-mon02,daq-amo-mon03,daq-amo-mon04,daq-amo-mon05,daq-amo-mon06 amon0816.sh
#
# bsub -n 288 -q psnehhiprioq -o /reg/data/ana13/amo/amon0816/results/mbucher/%J.log mpirun python /reg/neh/operator/amoopr/2016/amon0816/amon0816-offline.py
#
#
#

# Standard PYTHON modules
import cv2
import numpy as np
from skbeam.core.accumulators.histogram import Histogram

# LCLS psana to read data
from psana import *
from xtcav.ShotToShotCharacterization import *

# Custom stuff
import find_blobs

# parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# parameter parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("runs", help="psana run numer",type=int,nargs='+')
args = parser.parse_args()

# Set up event counters
eventCounter = 0
evtGood = 0
evtBad = 0

##################################################################
########################### Histograms & stuff ###################
##################################################################

runs =np.array([args.runs],dtype='int').flatten()

xLength = 839 # rotated camera 798
yLength = 591 # rotated camera 647
photonE = 508. # 1050. for runs 134 - ~150 |||| 508. for 156, 157 |||| 530 for 158+
cutoff = 0.2   # gas detector cutoff
xtcav = True

binsOfHist = xLength * len(runs)

spectraCombined = Histogram((binsOfHist,0.,float(binsOfHist)))

delayhist2d = Histogram((50,0.,50.),(binsOfHist,0.,float(binsOfHist)))
delayCountHist = Histogram((50,0.,50.))

feegashist2d = Histogram((100, 0.,2.0),(binsOfHist,0.,float(binsOfHist)))
feeCountHist = Histogram((100,0.,2.0))

photonhist2d = Histogram((200, photonE*(1.-0.02),photonE*(1.+0.02)),(binsOfHist,0.,float(binsOfHist)))
photonCountHist = Histogram((200,photonE*(1.-0.02),photonE*(1.+0.02)))

RatiosXTCAV = Histogram((50, -5., 5.),(binsOfHist,0.,float(binsOfHist)))
TotalPowXTCAV = Histogram((100, 600., 1200.),(binsOfHist,0.,float(binsOfHist)))
RatiosiTOF = Histogram((50, -5., 5.),(binsOfHist,0.,float(binsOfHist)))
TotalPowiTOF = Histogram((20, -20., 0.),(binsOfHist,0.,float(binsOfHist)))
RelYieldsXTCAV = Histogram((60, 200., 800.),(60, 200., 800.))
RelYieldsiTOF = Histogram((100, -10., 0.),(100, -10., 0.))
Correlation_iTOF_XTCAV_ratio = Histogram((50, -5., 5.),(50, -5., 5.))
Correlation_iTOF_XTCAV_pow = Histogram((20, -20., 0.),(100, 600., 1200.))

NormRuns = Histogram((len(runs),0.,float(len(runs))-.99))
#################################################################


### For perspective transformation and warp
#pts1 = np.float32([[229,273],[706,200],[265,822],[763,812]])
#pts1 = np.float32([[29,73],[906,00],[65,1022],[963,1012]])
#pts1 = np.float32([[53,301],[888,158],[136,821],[934,806]]) #rotated set
pts1 = np.float32([[91,248],[930,193],[91,762],[930,785]])
pts2 = np.float32([[0,0],[xLength,0],[0,yLength],[xLength,yLength]])
M = cv2.getPerspectiveTransform(pts1,pts2)


####################################################################
############################# STARTING ANALYSIS #####################
####################################################################

if rank == 0: print "Starting analysis on ", size," cores."


for run in runs:

    ###### --- Offline analysis
    ds = DataSource("exp=AMO/amon0816:run=%s:smd:dir=/reg/d/ffb/amo/amon0816/xtc:live" % run)
    #ds = DataSource("exp=AMO/amon0816:run=%s:smd" % run)

    XTCAVRetrieval = ShotToShotCharacterization();
    XTCAVRetrieval.SetEnv(ds.env())
    opal_det = Detector('OPAL1')
    #opal4_det = Detector('OPAL4')
    minitof_det = Detector('ACQ1')
    minitof_channel = 0
    ebeam_det = Detector('EBeam')
    feegas_det = Detector('FEEGasDetEnergy')
    eorbits_det = Detector('EOrbits')

    for nevt,evt in enumerate(ds.events()):
            if nevt%size!=rank: continue
            
            ##############################  READING IN DETECTORS #############################
            ### Opal Detector
            opal_raw = opal_det.raw(evt)
            #opal4_raw = opal4_det.raw(evt)


            ### TOF
            minitof_volts_raw = minitof_det.waveform(evt)
            minitof_times_raw = minitof_det.wftime(evt)


            ### Ebeam
            ebeam = ebeam_det.get(evt)

            # fee gas energy
            feegas = feegas_det.get(evt)

            # eorbits
            eorbits = eorbits_det.get(evt)

            # Check all detectors are read in
            eventCounter += 1
            if opal_raw is None or ebeam is None or feegas is None or eorbits is None:
                evtBad += 1
                print "Bad event"
                continue
            else:
                evtGood += 1
            
            #if evtGood > 5000: break

            opal = opal_raw.copy()
            #opal_screen = opal4_raw.copy()
            minitof_volts = minitof_volts_raw[minitof_channel]
            minitof_times = minitof_times_raw[minitof_channel]

            ##############################################################################


            ################################## Runs numbers ##############################
            n = int(np.where(runs==run)[0])
            ###############################################################################


            ################################## OPAL ANALYSIS #############################

            # Perform Perspective Transformation
            opal_pers = cv2.warpPerspective(opal,M,(xLength,yLength))
            #opal_pers = opal #NOTE: pers trans deactivated!!!!
            # threshold perspective
            threshold = 500

            # do blob finding
            # find blobs takes two variables. an Opal image (first) and then a threshold value (second)
            c,w = find_blobs.find_blobs(opal_pers,threshold)

            #find center of gravity. If there is a hit
            #if len(c) != 0:
            #    x2,y2 = zip(*c)
            #    blob_centroid = np.sum(x2)/float(len(x2))
            #    #print blob_centroid

            #    shift = 512-int(blob_centroid)
            #    mu,sig = moments(opal_thresh_xproj)	
            #    index = int(blob_centroid/10)            

                # uncomment bellow for two pulse two color jitter correction only !!!!

                #bot = 0     # first edge of the first energy peak
                #top = 700   # second edge of the first energy peak
                #mu,sig = moments(opal_thresh_xproj[bot:top])	
                #shift = 250 - int(mu) - bot
                #index = int(mu/10)        

            #else:
            #    shift = 0
            #    index = 0

            #print shift
            # Hit Histogram
            #hithist.reset()
            #hitjitter.reset()
            #hitseeded.reset()
            #delayhist2d.reset()
            #for hit in c:
                #hithist.fill(float(hit[0]+shift))
                #hitjitter.fill(float(hit[0]))
                #spectraCombined.fill(float(hit[1]+xLength*n))

            ###############################################################################


            ############################ MINI TOF ANALYSIS ################################

            # find yield of waveform over certain interval
            Thres_baseline = -0.05
            iTOF_yield_Ne = np.sum(minitof_volts[4020:4185]) #choose proper window later 
            iTOF_yield_CO = np.sum(minitof_volts[4770:4925]) #choose proper window later 

 
            # get maximum value of x-ray beam
            #max_hithist = np.amax(hithist.data,axis=0) # might have to change this to x2.data?
            #max_hithist = np.amax(opal_thresh_xproj,axis=0) # might have to change this to x2.data?
            #print max_hithist

            #if max_hithist > 25000: #change value later
            #    ion_yield[index] = ion_yield[index] + wf_yield
            #    for hit in c:
            #        hitseeded.fill(float(hit[0]))        

            #minitof_volts_thresh = minitof_volts
            #minitof_volts_thresh[minitof_volts_thresh > -0.01] = 0

            #print 'number of hits:',len(c)
            #find_blobs.draw_blobs(opal,c,w) # draw the blobs in the opal picture

    #        if 'sig' in locals():
    #            print sig
    #            if sig < 10: #change width condition
    #                ion_yield[index] = ion_yield[index] + wf_yield

            ###############################################################################


            ######################### XTCAV ANALYSIS #####################################
            pulse_separation = -666
            iyieldXTCAV = False
            if xtcav:
                if XTCAVRetrieval.SetCurrentEvent(evt):
                    time,power,ok = XTCAVRetrieval.XRayPower()
                    #print ok
                    if ok:
                        agreement,ok=XTCAVRetrieval.ReconstructionAgreement()
                        #print ok
                        if ok and agreement > 0.5:
                            times_p0 = np.asarray(time[0])
                            power_p0 = np.asarray(power[0])
                            power_p0[power_p0<0]=0
                            #times_p1 = np.asarray(time[1])
                            #power_p1 = np.asarray(power[1])
                            
                            mean_t_p0 = np.sum(times_p0*power_p0/np.sum(power_p0))
                            var_t_p0 = np.sum(times_p0**2*power_p0/np.sum(power_p0))
                            rms_p0 = np.sqrt(var_t_p0 - mean_t_p0**2)
                            pulse_separation = rms_p0
                            
                            #mean_t_p1 = np.sum(times_p1*power_p1/np.sum(power_p1))
                            #var_t_p1 = np.sum(times_p1**2*power_p1/np.sum(power_p1))
                            #rms_p1 = np.sqrt(var_t_p1 - mean_t_p1**2)

                            #pulse_separation = mean_t_p0 - mean_t_p1
                            
                            yieldXTCAV1 = np.sum(power_p0[0:140])
                            yieldXTCAV2 = np.sum(power_p0[141:288])
                            iyieldXTCAV = True
                            #print 'Happy'
                        


            #################################### eBeam & FEE gas det#######################
            L3Energy = ebeam.ebeamL3Energy()
            bla = L3Energy/13720.
            photonEnergy = 8330. * bla**2
            
            fee_1 = (feegas.f_11_ENRC()+feegas.f_12_ENRC())/2.
            fee_2 = (feegas.f_21_ENRC()+feegas.f_22_ENRC())/2.
            #fee_6 = (feegas.f_64_ENRC()+feegas.f_63_ENRC())/2.  # New R&D gas dets likely not calibrated.
            

            ###############################################################################



            ################################ Populate histograms ##########################
            if fee_2 > cutoff and iTOF_yield_CO <-2.:
                for hit in c:
                    count = [hit[1]+xLength*n]
                    spectraCombined.fill(count)
                    delayhist2d.fill([pulse_separation],count)
                    feegashist2d.fill([fee_2],count)
                    photonhist2d.fill([photonEnergy],count)
                    TotalPowiTOF.fill([iTOF_yield_Ne+iTOF_yield_CO],count)
                    RelYieldsiTOF.fill([iTOF_yield_Ne],[iTOF_yield_CO])
                    if iTOF_yield_Ne > iTOF_yield_CO:
                        RatiosiTOF.fill([iTOF_yield_Ne/iTOF_yield_CO],count)
                    if iTOF_yield_Ne < iTOF_yield_CO:
                        RatiosiTOF.fill([-iTOF_yield_CO/iTOF_yield_Ne],count)
                            
                    if iyieldXTCAV == True:
                        TotalPowXTCAV.fill([yieldXTCAV1+yieldXTCAV2],count)
                        RelYieldsXTCAV.fill([yieldXTCAV1],[yieldXTCAV2])
                        Correlation_iTOF_XTCAV_pow.fill([iTOF_yield_Ne+iTOF_yield_CO],[yieldXTCAV1+yieldXTCAV2])
                        Correlation_iTOF_XTCAV_ratio.fill([abs(yieldXTCAV1/yieldXTCAV2)],[abs(iTOF_yield_Ne/iTOF_yield_CO)])
                        if yieldXTCAV1 > yieldXTCAV2:
                            RatiosXTCAV.fill([yieldXTCAV1/yieldXTCAV2],count)
                        if yieldXTCAV2 > yieldXTCAV1:
                            RatiosXTCAV.fill([-yieldXTCAV2/yieldXTCAV1],count)
                        
            
                    NormRuns.fill(n)
                    delayCountHist.fill([pulse_separation])
                    feeCountHist.fill([fee_2])
                    photonCountHist.fill([photonEnergy])


            ############################# sporadic updates #####################
            if rank==0 and evtGood%10 == 0:
                print eventCounter*size
   

############################################################################
########################### Summing all up #################################
############################################################################

spectraCombined_all = np.zeros_like(spectraCombined.values)
comm.Reduce(spectraCombined.values,spectraCombined_all)

delayhist2d_all = np.zeros_like(delayhist2d.values)
comm.Reduce(delayhist2d.values,delayhist2d_all)

feegashist2d_all = np.zeros_like(feegashist2d.values)
comm.Reduce(feegashist2d.values,feegashist2d_all)

photonhist2d_all = np.zeros_like(photonhist2d.values)
comm.Reduce(photonhist2d.values,photonhist2d_all)

RatiosXTCAV_all = np.zeros_like(RatiosXTCAV.values)
comm.Reduce(RatiosXTCAV.values,RatiosXTCAV_all)

TotalPowXTCAV_all = np.zeros_like(TotalPowXTCAV.values)
comm.Reduce(TotalPowXTCAV.values,TotalPowXTCAV_all)

RelYieldsXTCAV_all = np.zeros_like(RelYieldsXTCAV.values)
comm.Reduce(RelYieldsXTCAV.values,RelYieldsXTCAV_all)

RatiosiTOF_all = np.zeros_like(RatiosiTOF.values)
comm.Reduce(RatiosiTOF.values,RatiosiTOF_all)

TotalPowiTOF_all = np.zeros_like(TotalPowiTOF.values)
comm.Reduce(TotalPowiTOF.values,TotalPowiTOF_all)

RelYieldsiTOF_all = np.zeros_like(RelYieldsiTOF.values)
comm.Reduce(RelYieldsiTOF.values,RelYieldsiTOF_all)

Correlation_iTOF_XTCAV_pow_all = np.zeros_like(Correlation_iTOF_XTCAV_pow.values)
comm.Reduce(Correlation_iTOF_XTCAV_pow.values,Correlation_iTOF_XTCAV_pow_all)

Correlation_iTOF_XTCAV_ratio_all = np.zeros_like(Correlation_iTOF_XTCAV_ratio.values)
comm.Reduce(Correlation_iTOF_XTCAV_ratio.values,Correlation_iTOF_XTCAV_ratio_all)

# Norms and counter
evtGood_array = np.array([evtGood])
evtGood_all = np.zeros_like(evtGood_array)
comm.Reduce(evtGood_array,evtGood_all)

NormRuns_all = np.zeros_like(NormRuns.values)
comm.Reduce(NormRuns.values,NormRuns_all)

delayCountHist_all = np.zeros_like(delayCountHist.values)
comm.Reduce(delayCountHist.values,delayCountHist_all)

feeCountHist_all = np.zeros_like(feeCountHist.values)
comm.Reduce(feeCountHist.values,feeCountHist_all)

photonCountHist_all = np.zeros_like(photonCountHist.values)
comm.Reduce(photonCountHist.values,photonCountHist_all)


############################### Saving files #############################

if rank==0:
    datName = str(runs).replace(' ','')
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/combinedSpectra_%s.npy' % datName,spectraCombined_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/XTCAVSpectra_%s.npy' % datName,delayhist2d_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/FeeGasSpectra_%s.npy' % datName,feegashist2d_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/PhotonSpectra_%s.npy' % datName,photonhist2d_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/RatiosXTCAV_%s.npy' % datName,RatiosXTCAV_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/TotalPowXTCAV_%s.npy' % datName,TotalPowXTCAV_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/RatiosiTOF_%s.npy' % datName,RatiosiTOF_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/TotalPowiTOF_%s.npy' % datName,TotalPowiTOF_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/RelYieldsXTCAV_%s.npy' % datName,RelYieldsXTCAV_all) 
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/RelYieldsiTOF_%s.npy' % datName,RelYieldsiTOF_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/CorrelationRatio_%s.npy' % datName,Correlation_iTOF_XTCAV_ratio_all) 
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/CorrelationPow_%s.npy' % datName,Correlation_iTOF_XTCAV_pow_all)
 

    # Norms and counter
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/NormRuns_%s.npy' % datName,NormRuns_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/GoodEvents_%s.npy' % datName,evtGood_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/XTCAVCounter_%s.npy' % datName,delayCountHist_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/FEEGasCounter_%s.npy' % datName,feeCountHist_all)
    np.save('/reg/d/psdm/amo/amon0816/results/mbucher/spectra/PhotonECounter_%s.npy' % datName,photonCountHist_all)

    print 'Analysis done. Files written.'
