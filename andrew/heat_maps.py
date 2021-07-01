import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle 

folder_dir = 'lightcurves2'
ns_dirs = os.listdir(f'./{folder_dir}')
#nsbh_dirs = os.listdir('./heatmap_files/bulla_2Component_lnsbh')

nsns_dict = {}
nsbh_dict = {}
bands = ['t','u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

for band in bands:
    nsns_dict[band], nsbh_dict[band] = [],[]

for ns_dir in ns_dirs:   
    with open (f'./{folder_dir}/{ns_dir}','rb') as f:
        data = pickle.load(f)
    for ii,band in enumerate(bands):
        #nsns_dict[band].append(data[:,ii])
        nsns_dict[band] = np.concatenate((nsns_dict[band], data[:,ii]))
    #if 'Lbol' in ns_dir or 'nsns' not in ns_dir: continue
    
    # comment this part out to include all inclinations:
#     if '_90.0.dat' not in ns_dir: continue

    #mag_d = np.loadtxt(f'./heatmap_files/bulla_2Component_lmid/{ns_dir}') 

#mag_d = np.loadtxt('lightcurve_data.txt') 
#print(np.shape(mag_d))
#t = mag_d[:,0] 
#for nsbh_dir in nsbh_dirs:
    #if 'Lbol' in nsbh_dir or 'nsbh' not in nsbh_dir: continue
        
    # comment this part out to include all inclinations:
#     #if '_90.0.dat' not in nsbh_dir: continue
        
    #mag_d = np.loadtxt(f'./heatmap_files/bulla_2Component_lnsbh/{nsbh_dir}')
'''
#pickle  
for i, band in enumerate(sample):
    a = np.array(['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K'] 
    np.vstack(a)
heatmap_name = 'hm_'+str(label)+'.pickle'  
print(np.shape(lightcurve_data)) 
print(lightcurve_data)
with open(heatmap_name,'wb') as filename:
    pickle.dump(lightcurve_data,filename,protocol=pickle.HIGHEST_PROTOCOL)
'''

'''
mag_d = np.loadtxt('lightcurve_data.txt')
mag_d = mag_d[0:100]
t = mag_d[:,0]
for ii,band in enumerate(bands):
    nsbh_dict[band] = mag_d[:,ii+1]
'''
f,axes=plt.subplots(ncols=5,nrows=2,figsize=(35,15),sharey='row')
plt.rcParams['figure.dpi'] = 200
plt.rc('xtick',labelsize=60)
plt.rc('ytick',labelsize=60)

print('Initializing BNS') 

t = nsns_dict['t']


for (i,j,band) in zip([0,0,0,0,0,1,1,1,1],[0,1,2,3,4,0,1,2,3],bands[1:10]):
    nsns = np.array(nsns_dict[band])
    #nsbh = np.array(nsbh_dict[band])
    
    #edit t range visible
    t_bins = t[0:99]
    bins = np.linspace(-20, 1, 50)
    
    #some sort of 0/nan check would be good
    #hist2d_1 = hist2d_1.astype('float')
    #hist2d_1[hist2d_1 == 0] = np.nan


    hist2d_1, xedges, yedges = np.histogram2d(t, nsns, bins = (t_bins,bins))
    X, Y = np.meshgrid(xedges, yedges)
    hist2d_1[hist2d_1 == 0] = np.nan

    #print(np.shape(X))
    #print(np.shape(Y))
    #print(np.shape(hist2d_1)) 
    im = axes[i][j].pcolormesh(X, Y, hist2d_1.T, shading = 'auto', cmap='Reds',alpha=0.7)
    print(str(band)+' complete') 
   
    '''
    Fix percentiles --------------------------------------------
    '''   

    #print (np.shape(nsns))
    #plot 10th, 50th, 90th percentiles
    #axes[i][j].plot(t_bins, np.nanpercentile(nsns,50),c='k',linestyle='--',label='NSNS')
    #axes[i][j].plot(t_bins, np.nanpercentile(nsns,90),'k--')
    #axes[i][j].plot(t_bins, np.nanpercentile(nsns,10),'k--')   
    if band == 'K':
        cb_ax = f.add_axes([0.94, 0.14, 0.023, 0.7])
        cb = f.colorbar(im, cax = cb_ax, ticks=[])
        cb.set_label(label='NSNS',size=30)

#        axes[i][j].text(0.676,0.8,'   High \nDensity',size=10.1)
#        axes[i][j].text(0.676,0.1,'   Low \nDensity',size=10.1)
    '''
    hist2d_2 = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, nsbh)

    hist2d_2 = hist2d_2.astype('float')
    hist2d_2[hist2d_2 == 0] = np.nan

    im = axes[i][j].pcolormesh(X, Y, hist2d_2, shading = 'auto', cmap='hot',alpha=0.6)
    axes[i][j].plot(t, np.nanpercentile(nsbh,50,axis=0),'w--',label='NSBH')
    axes[i][j].plot(t, np.nanpercentile(nsbh,90,axis=0),'w--')
    axes[i][j].plot(t, np.nanpercentile(nsbh,10,axis=0),'w--')

    if band == 'K':
        cb_ax = f.add_axes([1.00, 0.14, 0.023, 0.7])
        cb = f.colorbar(im, cax = cb_ax, ticks=[])
        cb.set_label(label='NSBH',size=30)
    '''
    axes[i][j].set_ylim([0,-20])
    axes[i][j].text(10,-17,f'{band}',size=30)

f.text(0.5,0.05,'Time [days]',size=30)
axes[0][0].set_ylabel('$M_{AB}$',size=30)
axes[1][0].set_ylabel('$M_{AB}$',size=30)

axes[-1, -1].axis('off')

h1, l1 = axes[0][0].get_legend_handles_labels()
h2, l2 = axes[1][1].get_legend_handles_labels()

#Make the legend
legend = axes[-1][-1].legend(h1, l1,  bbox_to_anchor=(0,1,1.0,-0.15), loc=9,
           ncol=1,prop={'size': 30},fancybox=True,frameon=True)

frame = legend.get_frame()
frame.set_color('skyblue')

plt.savefig('./heatmap_test.pdf',bbox_inches='tight')
plt.show()


'''
Types = [....all the types from the othr code....
''' 



