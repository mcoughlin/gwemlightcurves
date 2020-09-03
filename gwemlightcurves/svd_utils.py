# https://arxiv.org/abs/1705.07084

import os, sys, glob
import numpy as np
import scipy.interpolate
from scipy.interpolate import interpolate as interp
from scipy.interpolate import griddata
import scipy.signal

from gwemlightcurves import lightcurve_utils, Global

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, RationalQuadratic

try:
    import torch
    import gpytorch

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x,                                                                    covar_x)
except:
    print('Install gpytorch if you want to use it...')

#import george
#from george import kernels

def calc_svd_lbol(tini,tmax,dt, n_coeff = 100, model = "BaKa2016",
                  gptype="sklearn"):

    print("Calculating SVD model of bolometric luminosity...")

    if model == "BaKa2016":    
        fileDir = "../output/barnes_kilonova_spectra"
    elif model == "Ka2017":
        fileDir = "../output/kasen_kilonova_grid"
    elif model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"
    elif model == "Bu2019":
        fileDir = "../output/bulla_1D"
        fileDir = "../output/bulla_1D_phi0"
        fileDir = "../output/bulla_1D_phi90"
    elif model == "Bu2019inc":
        fileDir = "../output/bulla_2D"
    elif model == "Bu2019lf":
        fileDir = "../output/bulla_2Component_lfree"
    elif model == "Bu2019lr":
        fileDir = "../output/bulla_2Component_lrich"
    elif model == "Bu2019lm":
        fileDir = "../output/bulla_2Component_lmid"
    elif model == "Bu2019lw":
        fileDir = "../output/bulla_2Component_lmid_0p005"
    elif model == "Bu2019bc":
        fileDir = "../output/bulla_blue_cone"
    elif model == "Bu2019re":
        fileDir = "../output/bulla_red_ellipse"
    elif model == "Bu2019op":
        fileDir = "../output/bulla_opacity"
    elif model == "Bu2019ops":
        fileDir = "../output/bulla_opacity_slim"
    elif model == "Bu2019rp":
        fileDir = "../output/bulla_reprocess"
    elif model == "Bu2019rps":
        fileDir = "../output/bulla_reprocess_slim"
    elif model == "Bu2019rpd":
        fileDir = "../output/bulla_reprocess_decimated"
    elif model == "Bu2019nsbh":
        fileDir = "../output/bulla_2Component_lnsbh"
    elif model == "Wo2020dyn":
        fileDir = "../output/bulla_rosswog_dynamical"
    elif model == "Wo2020dw":
        fileDir = "../output/bulla_rosswog_wind"


    filenames = glob.glob('%s/*_Lbol.dat'%fileDir)

    lbols, names = lightcurve_utils.read_files_lbol(filenames)
    lbolkeys = lbols.keys()

    tt = np.arange(tini,tmax+dt,dt)

    for key in lbolkeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            lbols[key]["mej"] = mej0
            lbols[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))

            #if (mej0 == 0.05) and (vej0 == 0.2) and (Xlan0 == 1e-3):
          #    del lbols[key]
            #    continue

            lbols[key]["mej"] = mej0
            lbols[key]["vej"] = vej0
            lbols[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            lbols[key]["mej"], lbols[key]["vej"], lbols[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)
        elif "gamma" in key:
            kappaLF = float(keySplit[2].replace("kappaLF",""))
            gammaLF = float(keySplit[3].replace("gammaLF",""))
            kappaLR = float(keySplit[4].replace("kappaLR",""))
            gammaLR = float(keySplit[5].replace("gammaLR",""))
            theta = float(keySplit[6])

            lbols[key]["kappaLF"] = kappaLF
            lbols[key]["gammaLF"] = gammaLF
            lbols[key]["kappaLR"] = kappaLR
            lbols[key]["gammaLR"] = gammaLR
            lbols[key]["theta"] = theta

        elif "nsns" in key:

            mejdyn = float(keySplit[2].replace("mejdyn",""))
            mejwind = float(keySplit[3].replace("mejwind",""))
            phi0 = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])

            lbols[key]["mej_dyn"] = mejdyn
            lbols[key]["mej_wind"] = mejwind
            lbols[key]["phi"] = phi0
            lbols[key]["theta"] = theta

        elif keySplit[0] == "nsbh":

            mej_dyn = float(keySplit[2].replace("mejdyn",""))
            mej_wind = float(keySplit[3].replace("mejwind",""))
            phi = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])

            lbols[key]["mej_dyn"] = mej_dyn
            lbols[key]["mej_wind"] = mej_wind
            #lbols[key]["phi"] = phi
            lbols[key]["theta"] = theta

        elif "mejdyn" in key:

            mejdyn = float(keySplit[1].replace("mejdyn",""))
            mejwind = float(keySplit[2].replace("mejwind",""))
            phi0 = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])

            lbols[key]["mej_dyn"] = mejdyn
            lbols[key]["mej_wind"] = mejwind
            lbols[key]["phi"] = phi0
            lbols[key]["theta"] = theta

        elif "bluecone" in key:

            mej = float(keySplit[2].replace("mej",""))
            phi0 = float(keySplit[3].replace("th",""))
            theta = float(keySplit[4])

            lbols[key]["mej"] = mej
            lbols[key]["phi"] = 90-phi0
            lbols[key]["theta"] = theta

        elif "redellips" in key:

            mej = float(keySplit[2].replace("mej",""))
            a0 = float(keySplit[3].replace("a",""))
            theta = float(keySplit[4])

            lbols[key]["mej"] = mej
            lbols[key]["a"] = a0
            lbols[key]["theta"] = theta

        elif keySplit[0] == "nph1.0e+06":
 
            mej0 = float(keySplit[1].replace("mej",""))
            phi0 = float(keySplit[2].replace("phi",""))
            theta = float(keySplit[3])

            lbols[key]["mej"] = mej0
            lbols[key]["phi"] = phi0
            #lbols[key]["T"] = T0
            lbols[key]["theta"] = theta

        elif keySplit[0] == "kasenReprocess":

            mej1 = float(keySplit[2].replace("mejcone",""))
            mej2 = float(keySplit[3].replace("mejell",""))
            phi = float(keySplit[4].replace("th",""))
            a = float(keySplit[5].replace("a",""))
            theta = float(keySplit[6])

            lbols[key]["mej_1"] = mej1
            lbols[key]["mej_2"] = mej2
            lbols[key]["phi"] = phi
            lbols[key]["a"] = a
            lbols[key]["theta"] = theta

        elif keySplit[0] == "RGAdyn":
            
            mej = float(keySplit[2].replace("mej",""))
            a = float(keySplit[3].replace("a",""))
            sd = float(keySplit[4].replace("sd",""))
            theta = float(keySplit[5])

            lbols[key]["mej"] = mej
            lbols[key]["a"] = a
            lbols[key]["sd"] = sd
            lbols[key]["theta"] = theta

        elif keySplit[0] == "WoWind":

            mej = float(keySplit[2].replace("mej",""))
            rwind = float(keySplit[3].replace("v",""))
            theta = float(keySplit[4])

            lbols[key]["mej"] = mej
            lbols[key]["rwind"] = rwind
            lbols[key]["theta"] = theta
        
        ii = np.where(np.isfinite(lbols[key]["Lbol"]))[0]
        f = interp.interp1d(lbols[key]["tt"][ii], np.log10(lbols[key]["Lbol"][ii]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbols[key]["Lbol"]= np.log10(lbolinterp)

    lbolkeys = lbols.keys()

    lbol_array = []
    param_array = []
    for key in lbolkeys:
        lbol_array.append(lbols[key]["Lbol"])
        if model == "BaKa2016":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["vej"]])
        elif model == "Ka2017":
            param_array.append([np.log10(lbols[key]["mej"]),np.log10(lbols[key]["vej"]),np.log10(lbols[key]["Xlan"])])
        elif model == "RoFe2017":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["vej"],lbols[key]["Ye"]]) 
        elif model == "Bu2019":
            param_array.append([np.log10(lbols[key]["mej"]),np.log10(lbols[key]["T"])])
        elif model == "Bu2019inc":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["phi"],lbols[key]["theta"]])
        elif model in ["Bu2019lf","Bu2019lr","Bu2019lm"]:
            param_array.append([np.log10(lbols[key]["mej_dyn"]),np.log10(lbols[key]["mej_wind"]),lbols[key]["phi"],lbols[key]["theta"]])
        elif model in ["Bu2019nsbh"]:
            param_array.append([np.log10(lbols[key]["mej_dyn"]),np.log10(lbols[key]["mej_wind"]),lbols[key]["theta"]])
        elif model in ["Bu2019lw"]:
            param_array.append([np.log10(lbols[key]["mej_wind"]),lbols[key]["phi"],lbols[key]["theta"]])
        elif model == "Bu2019bc":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["phi"],lbols[key]["theta"]])
        elif model == "Bu2019re":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["a"],lbols[key]["theta"]])
        elif model == "Bu2019op":
            param_array.append([np.log10(lbols[key]["kappaLF"]),lbols[key]["gammaLF"],np.log10(lbols[key]["kappaLR"]),lbols[key]["gammaLR"]])
        elif model == "Bu2019ops":
            param_array.append([np.log10(lbols[key]["kappaLF"]),np.log10(lbols[key]["kappaLR"]),lbols[key]["gammaLR"]])
        elif model in ["Bu2019rp","Bu2019rpd"]:
            param_array.append([np.log10(lbols[key]["mej_1"]),np.log10(lbols[key]["mej_2"]),lbols[key]["phi"],lbols[key]["a"],lbols[key]["theta"]])
        elif model == "Bu2019rps":
            param_array.append([np.log10(lbols[key]["mej_1"]),np.log10(lbols[key]["mej_2"]),lbols[key]["a"]])
        elif model == "Wo2020dyn":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["a"],lbols[key]["sd"],lbols[key]["theta"]])
        elif model == "Wo2020dw":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["rwind"],lbols[key]["theta"]])
    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
    for i in range(len(param_mins)):
        param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i]) 

    lbol_array_postprocess = np.array(lbol_array)
    mins,maxs = np.min(lbol_array_postprocess,axis=0),np.max(lbol_array_postprocess,axis=0)
    for i in range(len(mins)):
        lbol_array_postprocess[:,i] = (lbol_array_postprocess[:,i]-mins[i])/(maxs[i]-mins[i])    
    lbol_array_postprocess[np.isnan(lbol_array_postprocess)]=0.0

    UA, sA, VA = np.linalg.svd(lbol_array_postprocess, full_matrices=True)
    VA = VA.T

    n, n = UA.shape
    m, m = VA.shape

    cAmat = np.zeros((n_coeff,n))
    cAvar = np.zeros((n_coeff,n))
    for i in range(n):
        cAmat[:,i] = np.dot(lbol_array_postprocess[i,:],VA[:,:n_coeff])
        ErrorLevel = 2.0
        errors = ErrorLevel*lbol_array_postprocess[i,:]
        cAvar[:,i] = np.diag(np.dot(VA[:,:n_coeff].T,np.dot(np.diag(np.power(errors,2.)),VA[:,:n_coeff])))
    cAstd = np.sqrt(cAvar)

    nsvds, nparams = param_array_postprocess.shape
    if gptype == "sklearn":
        kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
        gps = []
        for i in range(n_coeff):
            gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
            gp.fit(param_array_postprocess, cAmat[i,:])
            gps.append(gp)
    elif gptype == "gpytorch":
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #training_iter = 10
        training_iter = 1

        gps = []
        for i in range(n_coeff):
            if np.mod(i,5) == 0:
                print('Coefficient %d/%d...' % (i, n_coeff))

            train_x = torch.from_numpy(param_array_postprocess).float()
            train_y = torch.from_numpy(cAmat[i,:]).float()

            if torch.cuda.is_available():
                train_x = train_x.cuda()
                train_y = train_y.cuda()
                likelihood = likelihood.cuda()

            model = ExactGPModel(train_x, train_y, likelihood)
            if torch.cuda.is_available():
                model = model.cuda()

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                         lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,
                                                           model)

            for j in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                #print('Coeff %d/%d Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i+1, n_coeff, j + 1, training_iter, loss.item(), model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()))
                optimizer.step()

            gps.append(model.state_dict())

    svd_model = {}
    svd_model["n_coeff"] = n_coeff
    svd_model["param_array"] = param_array
    svd_model["param_array_postprocess"] = param_array_postprocess
    svd_model["cAmat"] = cAmat
    svd_model["cAstd"] = cAstd
    svd_model["VA"] = VA
    svd_model["param_mins"] = param_mins
    svd_model["param_maxs"] = param_maxs
    svd_model["mins"] = mins
    svd_model["maxs"] = maxs
    svd_model["gps"] = gps
    svd_model["tt"] = tt

    print("Finished calculating SVD model of bolometric luminosity...")

    return svd_model

def calc_svd_mag(tini,tmax,dt, n_coeff = 100, model = "BaKa2016",
                 gptype="sklearn"):

    print("Calculating SVD model of lightcurve magnitudes...")

    if model == "BaKa2016":
        fileDir = "../output/barnes_kilonova_spectra"
    elif model == "Ka2017":
        fileDir = "../output/kasen_kilonova_grid"
    elif model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"
    elif model == "Bu2019":
        fileDir = "../output/bulla_1D"
        fileDir = "../output/bulla_1D_phi0"
        fileDir = "../output/bulla_1D_phi90"
    elif model == "Bu2019inc":
        fileDir = "../output/bulla_2D"
    elif model == "Bu2019lf":
        fileDir = "../output/bulla_2Component_lfree"
    elif model == "Bu2019lr":
        fileDir = "../output/bulla_2Component_lrich"
    elif model == "Bu2019lm":
        fileDir = "../output/bulla_2Component_lmid"
    elif model == "Bu2019lw":
        fileDir = "../output/bulla_2Component_lmid_0p005"
    elif model == "Bu2019bc":
        fileDir = "../output/bulla_blue_cone"
    elif model == "Bu2019re":
        fileDir = "../output/bulla_red_ellipse"
    elif model == "Bu2019op":
        fileDir = "../output/bulla_opacity"
    elif model == "Bu2019ops":
        fileDir = "../output/bulla_opacity_slim"
    elif model == "Bu2019rp":
        fileDir = "../output/bulla_reprocess"
    elif model == "Bu2019rps":
        fileDir = "../output/bulla_reprocess_slim"
    elif model == "Bu2019rpd":
        fileDir = "../output/bulla_reprocess_decimated"
    elif model == "Bu2019nsbh":
        fileDir = "../output/bulla_2Component_lnsbh"
    elif model == "Wo2020dyn":
        fileDir = "../output/bulla_rosswog_dynamical"
    elif model == "Wo2020dw":
        fileDir = "../output/bulla_rosswog_wind"

    filenames_all = glob.glob('%s/*.dat'%fileDir)
    idxs = []
    for ii,filename in enumerate(filenames_all):
        if "_Lbol.dat" in filename: continue
        if "_spec.dat" in filename: continue
        idxs.append(ii)
    filenames = [filenames_all[idx] for idx in idxs]

    mags, names = lightcurve_utils.read_files(filenames)
    magkeys = mags.keys()

    tt = np.arange(tini,tmax+dt,dt)
    filters = ["u","g","r","i","z","y","J","H","K"]

    for jj, key in enumerate(magkeys):
        print('Setup %s: %d/%d' % (key, jj, len(magkeys)+1))
        
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            mags[key]["mej"] = mej0
            mags[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                #del mags[key]
                #continue
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e","")) 

            #if (mej0 == 0.05) and (vej0 == 0.2) and (Xlan0 == 1e-3):
            #    del mags[key]
            #    continue
 
            mags[key]["mej"] = mej0
            mags[key]["vej"] = vej0
            mags[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            mags[key]["mej"], mags[key]["vej"], mags[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        elif "gamma" in key:
            kappaLF = float(keySplit[2].replace("kappaLF",""))
            gammaLF = float(keySplit[3].replace("gammaLF",""))
            kappaLR = float(keySplit[4].replace("kappaLR",""))
            gammaLR = float(keySplit[5].replace("gammaLR",""))
            theta = float(keySplit[6])

            mags[key]["kappaLF"] = kappaLF
            mags[key]["gammaLF"] = gammaLF
            mags[key]["kappaLR"] = kappaLR
            mags[key]["gammaLR"] = gammaLR
            mags[key]["theta"] = theta

        elif "nsns" in key:

            mejdyn = float(keySplit[2].replace("mejdyn",""))
            mejwind = float(keySplit[3].replace("mejwind",""))
            phi0 = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])

            mags[key]["mej_dyn"] = mejdyn
            mags[key]["mej_wind"] = mejwind
            mags[key]["phi"] = phi0
            mags[key]["theta"] = theta

        elif keySplit[0] == "nsbh":

            mej_dyn = float(keySplit[2].replace("mejdyn",""))
            mej_wind = float(keySplit[3].replace("mejwind",""))
            phi = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])

            mags[key]["mej_dyn"] = mej_dyn
            mags[key]["mej_wind"] = mej_wind
            #mags[key]["phi"] = phi
            mags[key]["theta"] = theta

        elif "mejdyn" in key:

            mejdyn = float(keySplit[1].replace("mejdyn",""))
            mejwind = float(keySplit[2].replace("mejwind",""))
            phi0 = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])

            mags[key]["mej_dyn"] = mejdyn
            mags[key]["mej_wind"] = mejwind
            mags[key]["phi"] = phi0
            mags[key]["theta"] = theta

        elif "bluecone" in key:

            mej = float(keySplit[2].replace("mej",""))
            phi0 = float(keySplit[3].replace("th",""))
            theta = float(keySplit[4])

            mags[key]["mej"] = mej
            mags[key]["phi"] = 90-phi0
            mags[key]["theta"] = theta

        elif "redellips" in key:

            mej = float(keySplit[2].replace("mej",""))
            a0 = float(keySplit[3].replace("a",""))
            theta = float(keySplit[4])

            mags[key]["mej"] = mej
            mags[key]["a"] = a0
            mags[key]["theta"] = theta

        elif keySplit[0] == "nph1.0e+06":
            #if len(keySplit) == 5:
            #    mej0 = float(keySplit[3].replace("mej",""))
            #    phi0 = float(keySplit[2].replace("opang",""))
            #    T0 = float(keySplit[4].replace("T",""))
            #elif len(keySplit) == 6:
            #    mej0 = float(keySplit[2].replace("mej",""))
            #    phi0 = float(keySplit[3].replace("phi",""))
            #    T0 = float(keySplit[4].replace("T",""))
            #    theta = float(keySplit[5])

            
            mej0 = float(keySplit[1].replace("mej",""))
            phi0 = float(keySplit[2].replace("phi",""))
            theta = float(keySplit[3])

            mags[key]["mej"] = mej0
            mags[key]["phi"] = phi0
            mags[key]["theta"] = theta

        elif keySplit[0] == "kasenReprocess":

            mej1 = float(keySplit[2].replace("mejcone",""))
            mej2 = float(keySplit[3].replace("mejell",""))
            phi = float(keySplit[4].replace("th",""))
            a = float(keySplit[5].replace("a",""))
            theta = float(keySplit[6])

            mags[key]["mej_1"] = mej1
            mags[key]["mej_2"] = mej2
            mags[key]["phi"] = phi
            mags[key]["a"] = a
            mags[key]["theta"] = theta

        elif keySplit[0] == "RGAdyn":

            mej = float(keySplit[2].replace("mej",""))
            a = float(keySplit[3].replace("a",""))
            sd = float(keySplit[4].replace("sd",""))
            theta = float(keySplit[5])

            mags[key]["mej"] = mej
            mags[key]["a"] = a
            mags[key]["sd"] = sd
            mags[key]["theta"] = theta

        elif keySplit[0] == "WoWind":

            mej = float(keySplit[2].replace("mej",""))
            rwind = float(keySplit[3].replace("v",""))
            theta = float(keySplit[4])

            mags[key]["mej"] = mej
            mags[key]["rwind"] = rwind
            mags[key]["theta"] = theta

        mags[key]["data"] = np.zeros((len(tt),len(filters)))

        for jj,filt in enumerate(filters):
            ii = np.where(np.isfinite(mags[key][filt]))[0]
            f = interp.interp1d(mags[key]["t"][ii], mags[key][filt][ii], fill_value='extrapolate')
            maginterp = f(tt)
            mags[key]["data"][:,jj] = maginterp

        mags[key]["data_vector"] = np.reshape(mags[key]["data"],(len(tt)*len(filters),1))

    magkeys = mags.keys()
    param_array = []
    for key in magkeys:
        if model == "BaKa2016":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["vej"]])
        elif model == "Ka2017":
            param_array.append([np.log10(mags[key]["mej"]),np.log10(mags[key]["vej"]),np.log10(mags[key]["Xlan"])])
        elif model == "RoFe2017":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["vej"],mags[key]["Ye"]])    
        elif model == "Bu2019":
            param_array.append([np.log10(mags[key]["mej"]),np.log10(mags[key]["T"])])
        elif model == "Bu2019inc":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["phi"],mags[key]["theta"]])
        elif model in ["Bu2019lf","Bu2019lr","Bu2019lm"]:
            param_array.append([np.log10(mags[key]["mej_dyn"]),np.log10(mags[key]["mej_wind"]),mags[key]["phi"],mags[key]["theta"]])
        elif model in ["Bu2019nsbh"]:
            param_array.append([np.log10(mags[key]["mej_dyn"]),np.log10(mags[key]["mej_wind"]),mags[key]["theta"]])
        elif model in ["Bu2019lw"]:
            param_array.append([np.log10(mags[key]["mej_wind"]),mags[key]["phi"],mags[key]["theta"]])
        elif model == "Bu2019bc":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["phi"],mags[key]["theta"]])
        elif model == "Bu2019re":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["a"],mags[key]["theta"]])
        elif model == "Bu2019op":
            param_array.append([np.log10(mags[key]["kappaLF"]),mags[key]["gammaLF"],np.log10(mags[key]["kappaLR"]),mags[key]["gammaLR"]])
        elif model == "Bu2019ops":
            param_array.append([np.log10(mags[key]["kappaLF"]),np.log10(mags[key]["kappaLR"]),mags[key]["gammaLR"]])
        elif model in ["Bu2019rp","Bu2019rpd"]:
            param_array.append([np.log10(mags[key]["mej_1"]),np.log10(mags[key]["mej_2"]),mags[key]["phi"],mags[key]["a"],mags[key]["theta"]])
        elif model == "Bu2019rps":
            param_array.append([np.log10(mags[key]["mej_1"]),np.log10(mags[key]["mej_2"]),mags[key]["a"]])
        elif model == "Wo2020dyn":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["a"],mags[key]["sd"],mags[key]["theta"]])
        elif model == "Wo2020dw":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["rwind"],mags[key]["theta"]])

    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
    for i in range(len(param_mins)):
        param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    svd_model = {}
    for jj,filt in enumerate(filters):
        print('Computing filter %s...' % filt)
        mag_array = []
        for key in magkeys:
            mag_array.append(mags[key]["data"][:,jj])

        mag_array_postprocess = np.array(mag_array)
        mins,maxs = np.min(mag_array_postprocess,axis=0),np.max(mag_array_postprocess,axis=0)
        for i in range(len(mins)):
            mag_array_postprocess[:,i] = (mag_array_postprocess[:,i]-mins[i])/(maxs[i]-mins[i])
        mag_array_postprocess[np.isnan(mag_array_postprocess)]=0.0
        UA, sA, VA = np.linalg.svd(mag_array_postprocess, full_matrices=True)
        VA = VA.T

        n, n = UA.shape
        m, m = VA.shape

        cAmat = np.zeros((n_coeff,n))
        cAvar = np.zeros((n_coeff,n))
        for i in range(n):
            ErrorLevel = 1.0
            cAmat[:,i] = np.dot(mag_array_postprocess[i,:],VA[:,:n_coeff])
            errors = ErrorLevel*np.ones_like(mag_array_postprocess[i,:])
            cAvar[:,i] = np.diag(np.dot(VA[:,:n_coeff].T,np.dot(np.diag(np.power(errors,2.)),VA[:,:n_coeff])))
        cAstd = np.sqrt(cAvar)

        nsvds, nparams = param_array_postprocess.shape

        if gptype == "sklearn":
            kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
            gps = []
            for i in range(n_coeff):
                if np.mod(i,5) == 0:
                    print('Coefficient %d/%d...' % (i, n_coeff))
                gp = GaussianProcessRegressor(kernel=kernel,
                                              n_restarts_optimizer=0)
                gp.fit(param_array_postprocess, cAmat[i,:])
                gps.append(gp)
        elif gptype == "gpytorch":
            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            #training_iter = 10
            training_iter = 11

            gps = []
            for i in range(n_coeff):
                if np.mod(i,5) == 0:
                    print('Coefficient %d/%d...' % (i, n_coeff))

                train_x = torch.from_numpy(param_array_postprocess).float()
                train_y = torch.from_numpy(cAmat[i,:]).float()

                if torch.cuda.is_available():
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    likelihood = likelihood.cuda()

                model = ExactGPModel(train_x, train_y, likelihood)
                if torch.cuda.is_available():
                    model = model.cuda()

                # Find optimal model hyperparameters
                model.train()
                likelihood.train()

                # Use the adam optimizer
                optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                             lr=0.1)

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,
                                                               model)

                for j in range(training_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(train_x)
                    # Calc loss and backprop gradients
                    loss = -mll(output, train_y)
                    loss.backward()
                    #print('Coeff %d/%d Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i+1, n_coeff, j + 1, training_iter, loss.item(), model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()))
                    optimizer.step()

                gps.append(model.state_dict())

        svd_model[filt] = {}
        svd_model[filt]["n_coeff"] = n_coeff
        svd_model[filt]["param_array"] = param_array
        svd_model[filt]["param_array_postprocess"] = param_array_postprocess
        svd_model[filt]["cAmat"] = cAmat
        svd_model[filt]["cAstd"] = cAstd
        svd_model[filt]["VA"] = VA
        svd_model[filt]["param_mins"] = param_mins
        svd_model[filt]["param_maxs"] = param_maxs
        svd_model[filt]["mins"] = mins
        svd_model[filt]["maxs"] = maxs
        svd_model[filt]["gps"] = gps
        svd_model[filt]["tt"] = tt

    print("Finished calculating SVD model of lightcurve magnitudes...")

    return svd_model

def calc_svd_color_model(tini,tmax,dt, n_coeff = 100, model = "a2.0"):

    print("Calculating SVD model of inclination colors...")

    if model in ["DZ2","gamA2","gamB2"]:
        fileDir = "../output/wollaeger/%s" % model
    else:
        fileDir = "../output/kasen_kilonova_2D/%s" % model

    filenames_all = glob.glob('%s/*.dat'%fileDir)
    idxs = []
    for ii,filename in enumerate(filenames_all):
        if "_Lbol.dat" in filename: continue
        if "_spec.dat" in filename: continue
        idxs.append(ii)
    filenames = [filenames_all[idx] for idx in idxs]

    mags, names = lightcurve_utils.read_files(filenames)
    magkeys = mags.keys()

    tt = np.arange(tini,tmax+dt,dt)
    filters = ["u","g","r","i","z","y","J","H","K"]

    for key in magkeys:
        keySplit = key.split("_")
        if model in ["DZ2","gamA2","gamB2"]:
            incl = float(keySplit[1].split("=")[1])*360.0/(2*np.pi)
            mags[key]["iota"] = incl
        else:
            mags[key]["iota"] = float(keySplit[-1])

        mags[key]["data"] = np.zeros((len(tt),len(filters)))
        for jj,filt in enumerate(filters):
            ii = np.where(np.isfinite(mags[key][filt]))[0]
            if len(ii) > 1:
                f = interp.interp1d(mags[key]["t"][ii], mags[key][filt][ii], fill_value='extrapolate')
                maginterp = f(tt)
                mags[key]["data"][:,jj] = maginterp
            else:
                mags[key]["data"][:,jj] = 0.0

        mags[key]["data_vector"] = np.reshape(mags[key]["data"],(len(tt)*len(filters),1))

    magkeys = mags.keys()
    param_array = []
    for key in magkeys:
        param_array.append(mags[key]["iota"])

    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0), np.max(param_array_postprocess,axis=0)
    param_array_postprocess = (param_array_postprocess-param_mins)/(param_maxs-param_mins)
    #for i in range(len(param_mins)):
    #    param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    svd_model = {}
    for jj,filt in enumerate(filters):
        mag_array = []
        for key in magkeys:
            mag_array.append(mags[key]["data"][:,jj])

        mag_array_postprocess = np.array(mag_array)
        nmag, ntime = mag_array_postprocess.shape
        mag_array_postprocess_mean = np.median(mag_array_postprocess,axis=0)
        for i in range(nmag):
            mag_array_postprocess[i,:] = mag_array_postprocess[i,:] - mag_array_postprocess_mean

        mins,maxs = np.min(mag_array_postprocess,axis=0),np.max(mag_array_postprocess,axis=0)
        for i in range(len(mins)):
            mag_array_postprocess[:,i] = (mag_array_postprocess[:,i]-mins[i])/(maxs[i]-mins[i])
        mag_array_postprocess[np.isnan(mag_array_postprocess)]=0.0
        UA, sA, VA = np.linalg.svd(mag_array_postprocess, full_matrices=True)
        VA = VA.T

        n, n = UA.shape
        m, m = VA.shape

        cAmat = np.zeros((n_coeff,n))
        cAvar = np.zeros((n_coeff,n))
        for i in range(n):
            ErrorLevel = 0.01
            cAmat[:,i] = np.dot(mag_array_postprocess[i,:],VA[:,:n_coeff])
            errors = ErrorLevel*np.ones_like(mag_array_postprocess[i,:])
            cAvar[:,i] = np.diag(np.dot(VA[:,:n_coeff].T,np.dot(np.diag(np.power(errors,2.)),VA[:,:n_coeff])))
        cAstd = np.sqrt(cAvar)

        nsvds, nparams = np.atleast_2d(param_array_postprocess).shape
        kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
        gps = []
        for i in range(n_coeff):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            gp.fit(np.atleast_2d(param_array_postprocess).T, cAmat[i,:])
            gps.append(gp)

        svd_model[filt] = {}
        svd_model[filt]["n_coeff"] = n_coeff
        svd_model[filt]["param_array"] = param_array
        svd_model[filt]["cAmat"] = cAmat
        svd_model[filt]["cAstd"] = cAstd
        svd_model[filt]["VA"] = VA
        svd_model[filt]["param_mins"] = param_mins
        svd_model[filt]["param_maxs"] = param_maxs
        svd_model[filt]["mins"] = mins
        svd_model[filt]["maxs"] = maxs
        svd_model[filt]["gps"] = gps
        svd_model[filt]["tt"] = tt

    print("Finished calculating SVD model of inclination colors...")

    return svd_model


def calc_svd_spectra(tini,tmax,dt,lambdaini,lambdamax,dlambda, n_coeff = 100, model = "BaKa2016"):

    print("Calculating SVD model of lightcurve spectra...")

    if model == "BaKa2016":
        fileDir = "../output/barnes_kilonova_spectra"
    elif model == "Ka2017":
        fileDir = "../output/kasen_kilonova_grid"
    elif model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"

    filenames = glob.glob('%s/*_spec.dat'%fileDir)

    specs, names = lightcurve_utils.read_files_spec(filenames)
    speckeys = specs.keys()

    tt = np.arange(tini,tmax+dt,dt)
    lambdas = np.arange(lambdaini,lambdamax+dlambda,dlambda)

    for key in speckeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            specs[key]["mej"] = mej0
            specs[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                #del specs[key]
                #continue
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))

            #if (mej0 == 0.05) and (vej0 == 0.2) and (Xlan0 == 1e-3):
            #    del specs[key]
            #    continue

            specs[key]["mej"] = mej0
            specs[key]["vej"] = vej0
            specs[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            specs[key]["mej"], specs[key]["vej"], specs[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        data = specs[key]["data"].T
        data[data==0.0] = 1e-20
        f = interp.interp2d(specs[key]["t"], specs[key]["lambda"], np.log10(data), kind='cubic')
        #specs[key]["data"] = (10**(f(tt,lambdas))).T
        specs[key]["data"] = f(tt,lambdas).T

    speckeys = specs.keys()
    param_array = []
    for key in speckeys:
        if model == "BaKa2016":
            param_array.append([np.log10(specs[key]["mej"]),specs[key]["vej"]])
        elif model == "Ka2017":
            param_array.append([np.log10(specs[key]["mej"]),specs[key]["vej"],np.log10(specs[key]["Xlan"])])
        elif model == "RoFe2017":
            param_array.append([np.log10(specs[key]["mej"]),specs[key]["vej"],specs[key]["Ye"]])

    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
    for i in range(len(param_mins)):
        param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    svd_model = {}
    for jj,lambda_d in enumerate(lambdas):
        if np.mod(jj,1) == 0:
            print("%d / %d"%(jj,len(lambdas)))

        spec_array = []
        for key in speckeys:
            spec_array.append(specs[key]["data"][:,jj])

        spec_array_postprocess = np.array(spec_array)
        mins,maxs = np.min(spec_array_postprocess,axis=0),np.max(spec_array_postprocess,axis=0)
        for i in range(len(mins)):
            spec_array_postprocess[:,i] = (spec_array_postprocess[:,i]-mins[i])/(maxs[i]-mins[i])
        spec_array_postprocess[np.isnan(spec_array_postprocess)]=0.0
        UA, sA, VA = np.linalg.svd(spec_array_postprocess, full_matrices=True)
        VA = VA.T

        n, n = UA.shape
        m, m = VA.shape

        cAmat = np.zeros((n_coeff,n))
        cAvar = np.zeros((n_coeff,n))
        ErrorLevel=2
        for i in range(n):
            cAmat[:,i] = np.dot(spec_array_postprocess[i,:],VA[:,:n_coeff])
            errors = ErrorLevel*spec_array_postprocess[i,:]
            cAvar[:,i] = np.diag(np.dot(VA[:,:n_coeff].T,np.dot(np.diag(np.power(errors,2.)),VA[:,:n_coeff])))
        cAstd = np.sqrt(cAvar)

        nsvds, nparams = param_array_postprocess.shape
        kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)

        gps = []
        for i in range(n_coeff):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            gp.fit(param_array_postprocess, cAmat[i,:])
            gps.append(gp)

        svd_model[lambda_d] = {}
        svd_model[lambda_d]["n_coeff"] = n_coeff
        svd_model[lambda_d]["param_array"] = param_array
        svd_model[lambda_d]["cAmat"] = cAmat
        svd_model[lambda_d]["cAstd"] = cAstd
        svd_model[lambda_d]["VA"] = VA
        svd_model[lambda_d]["param_mins"] = param_mins
        svd_model[lambda_d]["param_maxs"] = param_maxs
        svd_model[lambda_d]["mins"] = mins
        svd_model[lambda_d]["maxs"] = maxs
        svd_model[lambda_d]["gps"] = gps
        svd_model[lambda_d]["tt"] = tt

    print("Finished calculating SVD model of lightcurve spectra...")

    return svd_model

def calc_color(tini,tmax,dt,param_list,svd_mag_color_model=None, model = "a2.0"):

    tt = np.arange(tini,tmax+dt,dt)

    if svd_mag_color_model == None:
        svd_mag_color_model = calc_svd_color_mag(tini,tmax,dt,model=model)

    filters = ["u","g","r","i","z","y","J","H","K"]
    mAB = np.zeros((9,len(tt)))
    for jj,filt in enumerate(filters):
        n_coeff = svd_mag_color_model[filt]["n_coeff"]
        param_array = svd_mag_color_model[filt]["param_array"]
        cAmat = svd_mag_color_model[filt]["cAmat"]
        VA = svd_mag_color_model[filt]["VA"]
        param_mins = svd_mag_color_model[filt]["param_mins"]
        param_maxs = svd_mag_color_model[filt]["param_maxs"]
        mins = svd_mag_color_model[filt]["mins"]
        maxs = svd_mag_color_model[filt]["maxs"]
        gps = svd_mag_color_model[filt]["gps"]
        tt_interp = svd_mag_color_model[filt]["tt"]

        param_list_postprocess = np.atleast_2d(np.array(param_list))
        #for i in range(len(param_mins)):
        #    param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

        param_list_postprocess = (param_list_postprocess-param_mins)/(param_maxs-param_mins)
        cAproj = np.zeros((n_coeff,))
        cAstd = np.zeros((n_coeff,))
        for i in range(n_coeff):
            gp = gps[i]
            y_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            cAproj[i] = y_pred
            cAstd[i] = sigma2_pred

        coverrors = np.dot(VA[:,:n_coeff],np.dot(np.power(np.diag(cAstd[:n_coeff]),2),VA[:,:n_coeff].T))
        errors = np.diag(coverrors)

        mag_back = np.dot(VA[:,:n_coeff],cAproj)
        mag_back = mag_back*(maxs-mins)+mins
        mag_back = scipy.signal.medfilt(mag_back,kernel_size=3)

        ii = np.where(~np.isnan(mag_back))[0]
        if len(ii) < 2:
            maginterp = np.nan*np.ones(tt.shape)
        else:
            f = interp.interp1d(tt_interp[ii], mag_back[ii], fill_value='extrapolate')
            maginterp = f(tt)
        mAB[jj,:] = maginterp

    return np.squeeze(tt), mAB


def calc_lc(tini,tmax,dt,param_list,svd_mag_model=None,svd_lbol_model=None,
            model = "BaKa2016", gptype="sklearn", n_coeff_lim=None):

    tt = np.arange(tini,tmax+dt,dt)

    if svd_mag_model == None:
        svd_mag_model = calc_svd_mag(tini,tmax,dt,model=model) 
    if svd_lbol_model == None:
        svd_lbol_model = calc_svd_lbol(tini,tmax,dt,model=model)

    filters = ["u","g","r","i","z","y","J","H","K"]
    mAB = np.zeros((9,len(tt)))
    for jj,filt in enumerate(filters):
        if n_coeff_lim is None:
            n_coeff = svd_mag_model[filt]["n_coeff"]
        else:
            n_coeff = n_coeff_lim
        param_array = svd_mag_model[filt]["param_array"]
        if gptype == "gpytorch":
            param_array_postprocess = svd_mag_model[filt]["param_array_postprocess"]
        cAmat = svd_mag_model[filt]["cAmat"]
        VA = svd_mag_model[filt]["VA"]
        param_mins = svd_mag_model[filt]["param_mins"]
        param_maxs = svd_mag_model[filt]["param_maxs"]
        mins = svd_mag_model[filt]["mins"]
        maxs = svd_mag_model[filt]["maxs"]
        gps = svd_mag_model[filt]["gps"]
        tt_interp = svd_mag_model[filt]["tt"]

        param_list_postprocess = np.array(param_list)
        for i in range(len(param_mins)):
            param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

        cAproj = np.zeros((n_coeff,))
        cAstd = np.zeros((n_coeff,))
        for i in range(n_coeff):
            gp = gps[i]
            if gptype == "sklearn":
                y_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
                cAproj[i] = y_pred
                cAstd[i] = sigma2_pred

            elif gptype == "gpytorch":
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPModel(torch.from_numpy(param_array_postprocess).float(),
                                     torch.from_numpy(cAmat[i,:]).float(), likelihood)
                model.load_state_dict(gp)

                # Get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()

                f_preds = model(torch.from_numpy(np.atleast_2d(param_list_postprocess)).float())
                y_preds = likelihood(model(torch.from_numpy(np.atleast_2d(param_list_postprocess)).float()))
                f_mean = f_preds.mean
                f_var = f_preds.variance
                f_covar = f_preds.covariance_matrix               

                cAproj[i] = f_mean
                cAstd[i] = f_var

        coverrors = np.dot(VA[:,:n_coeff],np.dot(np.power(np.diag(cAstd[:n_coeff]),2),VA[:,:n_coeff].T))
        errors = np.diag(coverrors)

        mag_back = np.dot(VA[:,:n_coeff],cAproj)
        mag_back = mag_back*(maxs-mins)+mins
        #mag_back = scipy.signal.medfilt(mag_back,kernel_size=3)

        ii = np.where(~np.isnan(mag_back))[0]
        if len(ii) < 2:
            maginterp = np.nan*np.ones(tt.shape)
        else:
            f = interp.interp1d(tt_interp[ii], mag_back[ii], fill_value='extrapolate')
            maginterp = f(tt)
        mAB[jj,:] = maginterp

    if n_coeff_lim is None:
        n_coeff = svd_lbol_model["n_coeff"]
    else:
        n_coeff = n_coeff_lim

    param_array = svd_lbol_model["param_array"]
    cAmat = svd_lbol_model["cAmat"]
    VA = svd_lbol_model["VA"]
    param_mins = svd_lbol_model["param_mins"]
    param_maxs = svd_lbol_model["param_maxs"]
    mins = svd_lbol_model["mins"]
    maxs = svd_lbol_model["maxs"]
    gps = svd_lbol_model["gps"]
    tt_interp = svd_lbol_model["tt"]

    param_list_postprocess = np.array(param_list)
    for i in range(len(param_mins)):
        param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    cAproj = np.zeros((n_coeff,))
    for i in range(n_coeff):
        gp = gps[i]
        if gptype == "sklearn":
            y_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            cAproj[i] = y_pred
        elif gptype == "gpytorch":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(torch.from_numpy(param_array_postprocess).float(),
                                 torch.from_numpy(cAmat[i,:]).float(), likelihood)
            model.load_state_dict(gp)

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            f_preds = model(torch.from_numpy(np.atleast_2d(param_list_postprocess)).float())
            y_preds = likelihood(model(torch.from_numpy(np.atleast_2d(param_list_postprocess)).float()))
            f_mean = f_preds.mean
            f_var = f_preds.variance
            f_covar = f_preds.covariance_matrix

            cAproj[i] = f_mean
            cAstd[i] = f_var

    lbol_back = np.dot(VA[:,:n_coeff],cAproj)
    lbol_back = lbol_back*(maxs-mins)+mins
    #lbol_back = scipy.signal.medfilt(lbol_back,kernel_size=3)

    ii = np.where(~np.isnan(lbol_back))[0]
    if len(ii) < 2:
        lbolinterp = np.nan*np.ones(tt.shape)
    else:
        f = interp.interp1d(tt_interp[ii], lbol_back[ii], fill_value='extrapolate')
        lbolinterp = 10**f(tt)
    lbol = lbolinterp

    return np.squeeze(tt), np.squeeze(lbol), mAB

def calc_spectra(tini,tmax,dt,lambdaini,lambdamax,dlambda,param_list,svd_spec_model=None,model = "BaKa2016"):

    tt = np.arange(tini,tmax+dt,dt)
    #lambdas = np.arange(lambdaini,lambdamax+dlambda,dlambda)
    lambdas = np.arange(lambdaini,lambdamax,dlambda)

    if svd_spec_model == None:
        svd_spec_model = calc_svd_spec(tini,tmax,dt,lambdaini,lambdamax,dlambda,model=model)
 
    spec = np.zeros((len(lambdas),len(tt)))
    for jj,lambda_d in enumerate(lambdas):
        n_coeff = svd_spec_model[lambda_d]["n_coeff"]
        param_array = svd_spec_model[lambda_d]["param_array"]
        cAmat = svd_spec_model[lambda_d]["cAmat"]
        cAstd = svd_spec_model[lambda_d]["cAstd"]
        VA = svd_spec_model[lambda_d]["VA"]
        param_mins = svd_spec_model[lambda_d]["param_mins"]
        param_maxs = svd_spec_model[lambda_d]["param_maxs"]
        mins = svd_spec_model[lambda_d]["mins"]
        maxs = svd_spec_model[lambda_d]["maxs"]
        gps = svd_spec_model[lambda_d]["gps"]
        tt_interp = svd_spec_model[lambda_d]["tt"]

        param_list_postprocess = np.array(param_list)
        for i in range(len(param_mins)):
            param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

        cAproj = np.zeros((n_coeff,))
        for i in range(n_coeff):
            gp = gps[i]
            y_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            cAproj[i] = y_pred

        spectra_back = np.dot(VA[:,:n_coeff],cAproj)
        spectra_back = spectra_back*(maxs-mins)+mins
        #spectra_back = scipy.signal.medfilt(spectra_back,kernel_size=3)

        N  = 3    # Filter order
        Wn = 0.1 # Cutoff frequency
        B, A = scipy.signal.butter(N, Wn, output='ba')
        #spectra_back = scipy.signal.filtfilt(B,A,spectra_back)

        ii = np.where(~np.isnan(spectra_back))[0]
        if len(ii) < 2:
            specinterp = np.nan*np.ones(tt.shape)
        else:
            f = interp.interp1d(tt_interp[ii], spectra_back[ii], fill_value='extrapolate')
            specinterp = 10**f(tt)
        spec[jj,:] = specinterp

    for jj, t in enumerate(tt):
        spectra_back = np.log10(spec[:,jj])
        spectra_back[~np.isfinite(spectra_back)] = -99.0
        if t < 7.0:
            spectra_back[1:-1] = scipy.signal.medfilt(spectra_back,kernel_size=5)[1:-1]
        else:
            spectra_back[1:-1] = scipy.signal.medfilt(spectra_back,kernel_size=5)[1:-1]
        ii = np.where((spectra_back!=0) & ~np.isnan(spectra_back))[0] 
        if len(ii) < 2:
            specinterp = np.nan*np.ones(lambdas.shape)
        else:
            f = interp.interp1d(lambdas[ii], spectra_back[ii], fill_value='extrapolate')
            specinterp = 10**f(lambdas)
        spec[:,jj] = specinterp

    return np.squeeze(tt), np.squeeze(lambdas), spec

