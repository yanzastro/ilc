# This code filter the Planck maps in needlets and calculate the covariance matrices.
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import fitsio as fs
import gc

Np = hp.nside2npix(2048)
freqs = np.array([100,143,217,353,545,857])
h = np.zeros((10,8192))
maps = np.zeros((6,Np))
maps_filtered = np.zeros((6,Np))
alm = np.zeros((6,33558528),dtype = complex)

localpath = '/home/yanza15/research/Ghinshaw/ILC/'

for j in range(6):
    freq = freqs[j]
    maps[j] = pf.open('/data/yanza15/sky_map/SkyMap_'+str(freqs[j])+'_2048_full_i_smth10.fits')[1].data['I'].ravel()
    alm[j] = hp.map2alm(maps[j])
del maps
gc.collect()
        
for i in range(10):
    h[i] = np.loadtxt(localpath + 'results/needlet_'+str(i)+'.txt')[0:6142]
    for j in range(6):
        alm_filtered = hp.almxfl(alm[j],h[i])
        maps_filtered[j] = hp.alm2map(alm_filtered,2048)
        hp.write_map(localpath + 'maps/Planck_maps/Sky_maps/HFI_SkyMap_'+str(freqs[j])+'_2048_R2.02_full_i_smth10_needlet'+str(i)+'.fits',maps_filtered[j])


