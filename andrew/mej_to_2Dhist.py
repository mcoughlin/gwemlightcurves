import numpy as np
from astropy.table import (Table, Column, vstack)
import sys
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle5 as pickle
from matplotlib import cm 

#cmpas = OrderedDict()






### non-standard libraries
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves import __version__
#from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope



#Types = ['BNS_alsing','BNS_farrow','BNS_equal_alsing','BNS_equal_farrow','BNS_uniform','NSBH_uniform','NSBH_zhu','BNS_chirp_q']

Types = ['BNS_alsing','BNS_farrow']
    
for Type in Types:
    fig, ax = plt.subplots(figsize=(16, 12))
    print(f'Initializing {Type}')



    folder_dir = f'./lightcurves_parallel/phi45_updated/{Type}/'
    #folder_dir = f'./lightcurves2/{Type}/'
    ns_dirs = os.listdir(f'{folder_dir}')
    nsns_dict = {}
    nsbh_dict = {}
    bands = ['t','u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K', 'mej', 'theta', 'phi'] 

  

    for band in bands:
        nsns_dict[band], nsbh_dict[band] = [],[]

    count = 0
    for ns_dir in ns_dirs:
        count+=1
        with open (f'./{folder_dir}/{ns_dir}','rb') as f:
            data = pickle.load(f)
        if count%1000 == 0:
            print(f'{count} samples loaded')
        for ii,band in enumerate(bands):
            nsns_dict[band].append(data[:,ii])
    
    #nsns_dict= nsns_dict[:100]
    u_band_nsns = nsns_dict['u']  
    

    mej_data = np.array(nsns_dict['mej'])  
   
    mej_data = mej_data[:,0] 

    thetas = np.array( nsns_dict['theta']) 
    
    thetas = thetas[:,0] 

    u_band_max = []
    for lc in u_band_nsns: 
        u_band_max.append(np.min(lc))   

    l = len(mej_data)
    print(f'{l} samples loaded' ) 

    mej_bins = np.linspace(-2.8, -.9, 50)
    mag_bins = np.linspace(-15.1, -12.3, 50) 
    theta_bins = np.linspace(np.min(thetas), np.max(thetas)) 
    #pl.hist(mej_bins, bins=np.logspace(np.log10(0.1),np.log10(1.0),50)) 
    #pl.gca().set_xsclae("log") 
    #pl.show() 



     
    print(np.shape(mej_data), np.shape(u_band_max))
    
 
    im  = plt.hist2d(np.log10(mej_data), u_band_max, bins =(mej_bins,mag_bins))
    #plt.title("mej vs peak mag")
    plt.xlabel('mej')
    plt.ylabel('Peak u mag')
    plt.gca().invert_yaxis()
    
    #theta/peakmag 
    #im = plt.hist2d(mej_theta_data, u_band_max, bins = (50,50)) 
    #plt.xlabel('theta')
    #plt.ylabel('Peak u mag')
    #plt.gca().invert_yaxis()
    #ymin, ymax = plt.ylim() 
     
    plt.colorbar(im[3])
    plt.savefig(f'mej_mag_hist2d_{Type}.pdf', bbox_inches='tight')
    plt.close()
   
    #theta/peakmag 
    im = plt.hist2d(thetas, u_band_max, bins = (theta_bins, mag_bins)) 
    plt.xlabel('theta')
    plt.ylabel('Peak u mag')
    plt.gca().invert_yaxis()
    
    plt.colorbar(im[3])
    plt.savefig(f'theta_mag_hist2d_{Type}.pdf', bbox_inches='tight')
    plt.close()



