import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle 


Types = ['BNS_alsing','BNS_farrow','BNS_equal_alsing','BNS_equal_farrow','BNS_uniform','NSBH_uniform','NSBH_zhu','BNS_chirp_q']
Types = ['BNS_alsing']

for Type in Types:
    #folder_dir = f'./lightcurves_parallel/{Type}/'
    folder_dir = f'./lightcurves2/alsing/'
    ns_dirs = os.listdir(f'{folder_dir}')

    nsns_dict = {}
    nsbh_dict = {}
    bands = ['t','u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

    for band in bands:
        nsns_dict[band], nsbh_dict[band] = [],[]

    for ns_dir in ns_dirs:   
        with open (f'./{folder_dir}/{ns_dir}','rb') as f:
            data = pickle.load(f)
        for ii,band in enumerate(bands):
            nsns_dict[band] = np.concatenate((nsns_dict[band], data[:,ii]))

    f,axes=plt.subplots(ncols=5,nrows=2,figsize=(35,15),sharey='row')
    plt.rcParams['figure.dpi'] = 200

    print(f'Initializing {Type}') 

    t = nsns_dict['t']


    for (i,j,band) in zip([0,0,0,0,0,1,1,1,1],[0,1,2,3,4,0,1,2,3],bands[1:10]):
        nsns = np.array(nsns_dict[band])
        t_bins = t[0:99]
        bins = np.linspace(-20, 1, 50)

        hist2d_1, xedges, yedges = np.histogram2d(t, nsns, bins = (t_bins,bins))
        X, Y = np.meshgrid(xedges, yedges)
        hist2d_1[hist2d_1 == 0] = np.nan

        im = axes[i][j].pcolormesh(X, Y, hist2d_1.T, shading = 'auto', cmap='viridis',alpha=0.7)
        print(str(band)+' complete') 
       
        p_list = []
        for t_val in t_bins:
            p_list.append(nsns[t == t_val])

        #plot 10th, 50th, 90th percentiles
        axes[i][j].plot(t_bins, np.nanpercentile(p_list, 50, axis=1),c='k',linestyle='--',label=f'{Type}')
        axes[i][j].plot(t_bins, np.nanpercentile(p_list, 90, axis=1),'k--')
        axes[i][j].plot(t_bins, np.nanpercentile(p_list, 10, axis=1),'k--')
    
        if band == 'K':
            cb_ax = f.add_axes([0.94, 0.14, 0.023, 0.7])
            cb = f.colorbar(im, cax = cb_ax, ticks=[])
            cb.set_label(label='NSNS',size=30)

        axes[i][j].set_ylim([0,-20])
        axes[i][j].text(10,-17,f'{band}',size=30)
        axes[i][j].tick_params(axis='x', labelsize=30)
        axes[i][j].tick_params(axis='y', labelsize=30)

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

    plt.savefig(f'./heatmaps_test/heatmap_{Type}.pdf',bbox_inches='tight')
    plt.show()
    print(f'{Type} complete')




