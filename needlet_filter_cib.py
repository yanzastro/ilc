# This code filter the Planck maps in needlets and calculate the covariance matrices.
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yanza15/research/Ghinshaw/CLASS/codes/sources')
import healpy_supplement as hps
import gc
import fitsio as fs


Np = hp.nside2npix(2048)
freqs = np.array([100,143,217,353,545,857])
h = np.zeros((10,8192))
maps = np.zeros((6,Np))

alm = np.zeros((6,33558528),dtype = complex)
datapath = '/data/yanza15'

for j in range(6):
    freq = freqs[j]
    maps[j] = hps.fast_read_map(datapath + '/CIB_map/Planck_CIB_map_F'+str(freqs[j])+'_smth10.fits')
    alm[j] =  hp.map2alm(maps[j],lmax = 8191)

del maps
gc.collect()
        
for i in range(10):
    h[i] = np.loadtxt('../results/needlet_'+str(i)+'.txt')[0:8192] ** 2
    maps_filtered = np.zeros((6,Np))
    for j in range(6):
        alm_filtered = hp.almxfl(alm[j],h[i])
        maps_filtered[j] = hp.alm2map(alm_filtered,2048)
        hp.write_map('../maps/Planck_maps/COM_CompMap_CIB-GNILC-F'+str(freqs[j])+'_2048_R2.00_smth10_needlet'+str(i)+'.fits',maps_filtered[j])
    del maps_filtered
    gc.collect()



