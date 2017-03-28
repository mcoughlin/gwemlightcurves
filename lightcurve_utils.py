
import os, sys
import optparse
import numpy as np
import glob
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def read_files(files,tmin=-100.0,tmax=100.0):

    names = []
    mags = {}
    for filename in files:
        name = filename.replace(".txt","").split("/")[-1]
        mag_d = np.loadtxt(filename)
        mag_d = mag_d[1:,:]

        t = mag_d[:,0]
        g = mag_d[:,1]
        try:
            index = np.nanargmin(g)
            index = 0
        except:
            index = 0
        t0 = t[index]

        mags[name] = {}
        mags[name]["t"] = mag_d[:,0]-t0
        indexes1 = np.where(mags[name]["t"]>=tmin)[0]
        indexes2 = np.where(mags[name]["t"]<=tmax)[0] 
        indexes = np.intersect1d(indexes1,indexes2)

        mags[name]["t"] = mag_d[indexes,0]
        mags[name]["g"] = mag_d[indexes,1]
        mags[name]["r"] = mag_d[indexes,2]
        mags[name]["i"] = mag_d[indexes,3]
        mags[name]["z"] = mag_d[indexes,4]

        names.append(name)

    return mags, names

def xcorr_mags(mags1,mags2):
    nmags1 = len(mags1)
    nmags2 = len(mags2)
    xcorrvals = np.zeros((nmags1,nmags2))
    chisquarevals = np.zeros((nmags1,nmags2))
    for ii,name1 in enumerate(mags1.iterkeys()):
        for jj,name2 in enumerate(mags2.iterkeys()):

            t1 = mags1[name1]["t"]
            t2 = mags2[name2]["t"]
            t = np.unique(np.append(t1,t2))
            t = np.arange(-100,100,0.1)      

            mag1 = np.interp(t, t1, mags1[name1]["g"])
            mag2 = np.interp(t, t2, mags2[name2]["g"])

            indexes1 = np.where(~np.isnan(mag1))[0]
            indexes2 = np.where(~np.isnan(mag2))[0]
            indexes = np.intersect1d(indexes1,indexes2)
            mag1 = mag1[indexes1]
            mag2 = mag2[indexes2]

            indexes1 = np.where(~np.isinf(mag1))[0]
            indexes2 = np.where(~np.isinf(mag2))[0]
            indexes = np.intersect1d(indexes1,indexes2)
            mag1 = mag1[indexes1]
            mag2 = mag2[indexes2]

            if len(indexes) == 0:
                xcorrvals[ii,jj] = 0.0
                continue           

            if len(mag1) < len(mag2):
                mag1vals = (mag1 - np.mean(mag1)) / (np.std(mag1) * len(mag1))
                mag2vals = (mag2 - np.mean(mag2)) / (np.std(mag2))
            else:
                mag1vals = (mag1 - np.mean(mag1)) / (np.std(mag1))
                mag2vals = (mag2 - np.mean(mag2)) / (np.std(mag2) * len(mag2))

            xcorr = np.correlate(mag1vals, mag2vals, mode='full')
            xcorr_corr = np.max(np.abs(xcorr))

            #mag1 = mag1 * 100.0 / np.sum(mag1)
            #mag2 = mag2 * 100.0 / np.sum(mag2)

            nslides = len(mag1) - len(mag1)
            if nslides == 0:
                chisquares = scipy.stats.chisquare(mag1, f_exp=mag1)[0]
            elif nslides > 0:
                chisquares = []
                for kk in xrange(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag1, f_exp=mag2[kk:len(mag1)])[0] 
                    chisquares.append(chisquare)
            elif nslides < 0:
                chisquares = []
                for kk in xrange(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag2, f_exp=mag1[kk:len(mag2)])[0] 
                    chisquares.append(chisquare)

            print name1, name2, xcorr_corr, np.min(np.abs(chisquares)), len(mag1), len(mag2)
            xcorrvals[ii,jj] = xcorr_corr
            chisquarevals[ii,jj] = np.min(np.abs(chisquares))

    return xcorrvals, chisquarevals
