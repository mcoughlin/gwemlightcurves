
import os, sys
import glob
import numpy as np
import h5py

from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

models = ["bulla_2D_phi0_phi90"]
models = ["bulla_2D"]
baseplotDir = "../plots/color_bulla"
if not os.path.exists(baseplotDir):
	os.mkdir(baseplotDir)
filts = ["u","g","r","i","z","y","J","H","K"]
redshifts = np.linspace(0,2,100)

for model in models:
    files = glob.glob("../output/%s/*"%model)

    magall = {}
    for file in files:
        if "Lbol" in file: continue
        name = file.split("/")[-1].replace(".mod","").replace(".spec","").replace("_AB","").replace(".h5","").replace(".dat","").replace(".txt","")

        mag_ds = np.loadtxt(file)

        ndata, nfilt = mag_ds.shape
        for jj in range(nfilt):
            if jj == 0: continue
            ii = np.where(np.isfinite(mag_ds[:,jj]))[0]
            f = interp.interp1d(mag_ds[ii,0], mag_ds[ii,jj], fill_value='extrapolate')
            maginterp = f(mag_ds[:,0])
            mag_ds[:,jj] = maginterp

        magall[name] = mag_ds

    colors = ['b', 'c', 'g', 'm', 'r', 'k']
    cmap = matplotlib.colors.ListedColormap(colors)
     
    cmap=plt.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm([0,15,30,45,60,75,90], cmap.N)

    plotName = "%s/mag_combo.pdf"%(baseplotDir)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6), sharey='row')
    for name in magall.keys():
        keySplit = name.split("_")
        print(keySplit)
        mej0 = float(keySplit[1].replace("mej",""))
        phi0 = float(keySplit[2].replace("phi",""))
        theta = float(keySplit[3])

        if theta == 0.0:
            marker = 'o'
        elif theta == 90:
            marker = 'v'
        elif theta == 45.6:
            marker = '^'
        else:
            continue

        idx = np.argmin(magall[name][:,-1])
        mag = magall[name][idx,-1]
        magcolor = magall[name][idx,-6] - magall[name][idx,-1]

        tt = magall[name][idx,0]
        ttplus = tt + 7
        idx2 = np.argmin(np.abs(ttplus-magall[name][:,0]))
        dmdt = magall[name][idx2,-1] - magall[name][idx,-1]

        im1 = ax1.scatter(magcolor,mag,marker=marker,c=phi0,cmap=cmap,norm=norm)
        im2 = ax2.scatter(np.log10(dmdt),mag,marker=marker,c=phi0,cmap=cmap,norm=norm)

    ax1.set_xlabel('Color at peak i-band [i-K] [mag]')
    ax2.set_xlabel(r'$\log_{10} \Delta m_{7}$ (K) [mag]')

    ax1.set_ylabel('Peak r-band AB Magnitude')
    #plt.xlim([-3,3])
    ax1.set_ylim([-17.5,-9])
    ax2.set_xlim([-1.5,1.75])
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    fig.subplots_adjust(wspace=0.04, hspace=0)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(r'$T_0$')

    ax1.arrow(1,-16,-1.0,4, color='k', width=0.1, alpha=0.2)
    ax1.text(0.25,-12, r'$M_{\rm ej}$')

    #ax1.arrow(1,-16,-1.0,5, color='k', width=0.1, alpha=0.2)
    #ax1.text(0,-10, r'$M_{\rm ej}$')

    ax1.arrow(-0.75,-17,1,0, color='k', width=0.1, alpha=0.2)
    ax1.text(-0.75,-17, r'$\theta$')

    #ax1.arrow(1,-16,-1,0, color='k', width=0.1, alpha=0.2)
    #ax1.text(1,-12, r'$v_{\rm ej}$')

    ax2.arrow(-1,-15,0.75,4.5, color='k', width=0.05, alpha=0.2)
    ax2.text(-0.25,-10, r'$M_{\rm ej}$')

    #ax2.arrow(0.2,-15.5,0.3,0, color='k', width=0.1, alpha=0.2)
    #ax2.text(1,-15.5, r'$v_{\rm ej}$')

    ax2.arrow(1.25,-17,-1,0, color='k', width=0.1, alpha=0.2)
    ax2.text(1.25,-17, r'$\theta$')

    plt.savefig(plotName, bbox_inches='tight')
    plt.close()

    filename = os.path.join(baseplotDir, "magcolor.dat")
    fid = open(filename, 'w')
    for name in magall.keys():
        keySplit = name.split("_")

        mej0 = float(keySplit[1].replace("mej",""))
        phi0 = float(keySplit[2].replace("phi",""))
        theta = float(keySplit[3])

        idx = np.argmin(magall[name][:,-7])
        mag = magall[name][idx,-7]
        magcolor = magall[name][idx,-6] - magall[name][idx,-7]

        tt = magall[name][idx,0]
        ttplus = tt + 7
        idx2 = np.argmin(np.abs(ttplus-magall[name][:,0]))
        dmdt = magall[name][idx2,-7] - magall[name][idx,-7]

        idx3 = np.argmin(magall[name][:,-6])
        tt = magall[name][idx3,0]
        ttplus = tt + 7
        idx4 = np.argmin(np.abs(ttplus-magall[name][:,0]))
        dmdti = magall[name][idx4,-6] - magall[name][idx3,-6]

        idx4 = np.argmin(magall[name][:,-8])
        magg = magall[name][idx,-8]
        maggcolor = magall[name][idx,-8] - magall[name][idx,-7]

        fid.write('%.5e %.5e %.5e %.5f %.5f %.5f %.5f\n' % (mej0, phi0, theta, magcolor, mag, dmdti, dmdt))

    fid.close()
    print(filename)
    print("stop")
    
