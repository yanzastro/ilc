# This code defines the needlet windows for ILC

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

fwhm = np.array([300,120,60,45,30,15,10,7.5,5]) 
#fwhm = np.radians(fwhm_deg)

sigma = 1 / np.sqrt(8. * np.log(2.)) * np.pi / 180. / 60. * fwhm

l = np.arange(10000)
b = np.zeros((9,10000))
h = np.zeros((10,10000))
for j in range(9):
    b[j] = np.exp(-l*(l+1)*sigma[j]**2/2.)

h[0] = b[0] ** 2
for j in range(8):
    h[j+1] = b[j+1]**2 - b[j]**2
h[9] = 1 - b[8]**2
plt.figure(figsize = (5,3))
for i in range(10):
    plt.semilogx(l,h[i]**0.5,color = 'k')
    np.savetxt('../results/needlet_'+str(i)+'.txt',h[i]**0.5)


btf = hp.gauss_beam(fwhm[-1] * np.pi / 180. / 60.,lmax = 9999)
#plt.semilogx(l,btf)
plt.xlabel(r'$\ell$')
plt.ylabel('Needlet Windows')
plt.savefig('/home/yanza15/figures/draft1/needlet.pdf',dpi = 1000)
plt.show()
