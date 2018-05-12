
import os, sys
import numpy as np
from scipy.signal import hilbert

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

from astropy.modeling.models import BlackBody1D
from astropy.modeling.blackbody import FLAM
from astropy import units as u

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

filename = "../spectra/G298048_XSH_20170821.dat"
data_out = np.loadtxt(filename)

lambdas = data_out[:,0]
spec = data_out[:,1]

spec_lowpass = butter_lowpass_filter(spec, 0.5, 1.0, order=5)

#analytic_signal = hilbert(spec)
#amplitude_envelope = np.abs(analytic_signal)

T = 2800
F = 8e-13
bb = BlackBody1D(temperature=T*u.K,bolometric_flux=F*u.erg/(u.cm**2 * u.s))
wav = np.arange(1000, 110000) * u.AA
flux = bb(wav).to(FLAM, u.spectral_density(wav))



plotDir = "../plots/whiten"
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

plotname = os.path.join(plotDir,'spec.pdf')

plt.figure()
plt.plot(lambdas,spec,'k-')
plt.plot(wav,flux,'r--')
plt.xlim([3500,25000])
plt.savefig(plotname)
plt.close()

