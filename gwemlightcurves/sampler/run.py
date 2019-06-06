
import os, sys
import numpy as np
import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves import lightcurve_utils, Global

def multinest(opts,plotDir):
   
    #n_live_points = 1000
    #n_live_points = 100
    n_live_points = opts.n_live_points
    evidence_tolerance = 0.5
    #evidence_tolerance = 10000.0
    max_iter = 0
    best = []

    if opts.model in ["KaKy2016","DiUj2017","Me2017","Me2017_A","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017inc","Ka2017_A","Ka2017x2","Ka2017x2inc","Ka2017x3","Ka2017x3inc","RoFe2017"]:
    
        if opts.doMasses:
            if opts.model == "KaKy2016":
                if opts.doEOSFit:
                    parameters = ["t0","q","chi_eff","mns","c","th","ph","zp"]
                    labels = [r"$T_0$",r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm ns}$",r"$C$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_KaKy2016_EOSFit, myprior_KaKy2016_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","q","chi_eff","mns","mb","c","th","ph","zp"]
                    labels = [r"$T_0$",r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm ns}$",r"$M_{\rm b}$",r"$C$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_KaKy2016, myprior_KaKy2016, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "DiUj2017":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","th","ph","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_DiUj2017_EOSFit, myprior_DiUj2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","th","ph","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_DiUj2017, myprior_DiUj2017, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "BaKa2016":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_BaKa2016_EOSFit, myprior_BaKa2016_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_BaKa2016, myprior_BaKa2016, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","xlan","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$","$X_{\rm lan}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Ka2017_EOSFit, myprior_Ka2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                elif opts.doBNSFit:
                    parameters = ["t0","m1","c1","m2","c2","xlan","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$","Xlan","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Ka2017_EOSFit, myprior_Ka2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","xlan","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$","$X_{\rm lan}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Ka2017, myprior_Ka2017, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "RoFe2017":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","ye","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$","Ye","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_RoFe2017_EOSFit, myprior_RoFe2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","ye","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$","Ye","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_RoFe2017, myprior_RoFe2017, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Me2017":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$",r"$\alpha$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Me2017_EOSFit, myprior_Me2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","beta","kappa_r","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$",r"$\alpha$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Me2017, myprior_Me2017, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "WoKo2017":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$",r"$\theta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_WoKo2017_EOSFit, myprior_WoKo2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","beta","kappa_r","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$",r"$\theta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_WoKo2017, myprior_WoKo2017, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "SmCh2017":
                if opts.doEOSFit:
                    parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$",r"$\beta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_SmCh2017_EOSFit, myprior_SmCh2017_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","m1","mb1","c1","m2","mb2","c2","beta","kappa_r","zp"]
                    labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$",r"$\beta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_SmCh2017, myprior_SmCh2017, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
        elif opts.doEjecta:
            if opts.model == "KaKy2016":
                parameters = ["t0","mej","vej","th","ph","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_KaKy2016_ejecta, myprior_KaKy2016_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "DiUj2017":
                parameters = ["t0","mej","vej","th","ph","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_DiUj2017_ejecta, myprior_DiUj2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "BaKa2016":
                parameters = ["t0","mej","vej","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_BaKa2016_ejecta, myprior_BaKa2016_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017":
                parameters = ["t0","mej","vej","xlan","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"${\rm log}_{10} (X_{\rm lan})$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Ka2017_ejecta, myprior_Ka2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017inc":
                parameters = ["t0","mej","vej","xlan","iota","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"${\rm log}_{10} (X_{\rm lan})$",r"$\iota$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Ka2017inc_ejecta, myprior_Ka2017inc_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017_A":
                parameters = ["t0","mej","vej","xlan","A","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"${\rm log}_{10} (X_{\rm lan})$",r"${\rm log}_{10} (A)$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Ka2017_A_ejecta, myprior_Ka2017_A_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017x2":
                if opts.doFitSigma:
                    parameters = ["t0","mej1","vej1","xlan1","mej2","vej2","xlan2","sigma","zp"]
                    labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (X_{\rm lan 1})$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (X_{\rm lan 2})$",r"$\sigma$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Ka2017x2_ejecta_sigma, myprior_Ka2017x2_ejecta_sigma, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
                else:
                    parameters = ["t0","mej1","vej1","xlan1","mej2","vej2","xlan2","zp"]
                    labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (X_{\rm lan 1})$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (X_{\rm lan 2})$","ZP"]
                    n_params = len(parameters)
                    pymultinest.run(myloglike_Ka2017x2_ejecta, myprior_Ka2017x2_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017x2inc":
                parameters = ["t0","mej1","vej1","xlan1","mej2","vej2","xlan2","iota","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (X_{\rm lan 1})$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (X_{\rm lan 2})$",r"$\iota$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Ka2017x2inc_ejecta, myprior_Ka2017x2inc_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017x3":
                parameters = ["t0","mej1","vej1","xlan1","mej2","vej2","xlan2","mej3","vej3","xlan3","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (X_{\rm lan 1})$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (X_{\rm lan 2})$",r"${\rm log}_{10} (M_{\rm ej 3})$",r"$v_{\rm ej 3}$",r"${\rm log}_{10} (X_{\rm lan 3})$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Ka2017x3_ejecta, myprior_Ka2017x3_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Ka2017x3inc":
                parameters = ["t0","mej1","vej1","xlan1","mej2","vej2","xlan2","mej3","vej3","xlan3","emcee","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (X_{\rm lan 1})$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (X_{\rm lan 2})$",r"${\rm log}_{10} (M_{\rm ej 3})$",r"$v_{\rm ej 3}$",r"${\rm log}_{10} (X_{\rm lan 3})$",r"$\iota$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Ka2017x3inc_ejecta, myprior_Ka2017x3inc_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "RoFe2017":
                parameters = ["t0","mej","vej","xlan","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$","$X_{\rm lan}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_RoFe2017_ejecta, myprior_RoFe2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Me2017":
                parameters = ["t0","mej","vej","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\alpha$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Me2017_ejecta, myprior_Me2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Me2017_A":
                parameters = ["t0","mej","vej","beta","kappa_r","zp","A"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\alpha$",r"${\rm log}_{10} \kappa_{\rm r}$","A","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Me2017_A_ejecta, myprior_Me2017_A_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "Me2017x2":
                parameters = ["t0","mej1","vej1","beta1","kappa_r1","mej2","vej2","beta2","kappa_r2","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"$\alpha_1$",r"${\rm log}_{10} \kappa_{\rm r 1}$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"$\alpha_2$",r"${\rm log}_{10} \kappa_{\rm r 2}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_Me2017x2_ejecta, myprior_Me2017x2_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "WoKo2017":
                parameters = ["t0","mej","vej","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\theta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_WoKo2017_ejecta, myprior_WoKo2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
            elif opts.model == "SmCh2017":
                parameters = ["t0","mej","vej","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\beta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_SmCh2017_ejecta, myprior_SmCh2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
        else:
            print("Enable --doEjecta or --doMasses")
            exit(0)
    elif opts.model in ["SN"]:
    
        parameters = ["t0","z","x0","x1","c","zp"]
        labels = [r"$T_0$", r"$z$", r"$x_0$", r"$x_1$",r"$c$","ZP"]
        n_params = len(parameters)
    
        pymultinest.run(myloglike_sn, myprior_sn, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
    
    elif opts.model in ["BoxFit"]:

        parameters = ["t0","theta_0","E","n","theta_obs","p","epsilon_B","epsilon_E","ksi_N","zp"]
        labels = [r"$T_0$", r"$theta_0$", r"$E$", r"$n$",r"$theta_{\rm obs}$","$p$","$epsilon_B$","$epsilon_E$","$ksi_N$","ZP"]
        n_params = len(parameters)

        pymultinest.run(myloglike_boxfit, myprior_boxfit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    elif opts.model in ["TrPi2018"]:

        parameters = ["t0","theta_v","E0","theta_c","theta_w","n","p","epsilon_E","epsilon_B","zp"]
        labels = [r"$T_0$", r"$\theta_v$", r"$E_0$", r"$\theta_c$", r"$\theta_w$", r"$n$",r"$p$", "$\epsilon_E$","$\epsilon_B$","ZP"]
        n_params = len(parameters)

        pymultinest.run(myloglike_TrPi2018, myprior_TrPi2018, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    elif opts.model in ["Ka2017_TrPi2018"]:

        parameters = ["t0","mej","vej","xlan","theta_v","E0","theta_c","theta_w","n","p","epsilon_E","epsilon_B","zp"]
        labels = [r"$T_0$", r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"${\rm log}_{10} (X_{\rm lan})$", r"$\theta_v$", r"$E_0$", r"$\theta_c$", r"$\theta_w$", r"$n$",r"$p$", "$\epsilon_E$","$\epsilon_B$","ZP"]
        n_params = len(parameters)

        pymultinest.run(myloglike_Ka2017_TrPi2018, myprior_Ka2017_TrPi2018, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    elif opts.model in ["Ka2017_TrPi2018_A"]:

        parameters = ["t0","mej","vej","xlan","theta_v","E0","theta_c","theta_w","n","p","epsilon_E","epsilon_B","zp","A"]
        labels = [r"$T_0$", r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"${\rm log}_{10} (X_{\rm lan})$", r"$\theta_v$", r"$E_0$", r"$\theta_c$", r"$\theta_w$", r"$n$",r"$p$", "$\epsilon_E$","$\epsilon_B$","${\rm log}_{10} (A)","ZP"]
        n_params = len(parameters)

        pymultinest.run(myloglike_Ka2017_TrPi2018_A, myprior_Ka2017_TrPi2018_A, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    #multifile= os.path.join(plotDir,'2-.txt')
    multifile = lightcurve_utils.get_post_file(plotDir)
    data = np.loadtxt(multifile)
    
    if opts.model == "KaKy2016":
        if opts.doMasses:
            if opts.doEOSFit:
                t0 = data[:,0]
                q = data[:,1]
                chi_eff = data[:,2]
                mns = data[:,3]
                c = data[:,4]
                th = data[:,5]
                ph = data[:,6]
                zp = data[:,7]
                loglikelihood = data[:,8]
                idx = np.argmax(loglikelihood)
                mb = lightcurve_utils.EOSfit(mns,c)
    
                t0_best = data[idx,0]
                q_best = data[idx,1]
                chi_best = data[idx,2]
                mns_best = data[idx,3]
                c_best = data[idx,4]
                th_best = data[idx,5]
                ph_best = data[idx,6]
                zp_best = data[idx,7]
                mb_best = mb[idx]
    
                tmag, lbol, mag = KaKy2016_model(q_best,chi_best,mns_best,mb_best,c_best,th_best,ph_best)
            else:
                t0 = data[:,0]
                q = data[:,1]
                chi_eff = data[:,2]
                mns = data[:,3]
                mb = data[:,4]
                c = data[:,5]
                th = data[:,6]
                ph = data[:,7]
                zp = data[:,8]
                loglikelihood = data[:,9]
                idx = np.argmax(loglikelihood)
    
                t0_best = data[idx,0]
                q_best = data[idx,1]
                chi_best = data[idx,2]
                mns_best = data[idx,3]
                mb_best = data[idx,4]
                c_best = data[idx,5]
                th_best = data[idx,6]
                ph_best = data[idx,7]
                zp_best = data[idx,8]
    
                tmag, lbol, mag = KaKy2016_model(q_best,chi_best,mns_best,mb_best,c_best,th_best,ph_best)
        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            th = data[:,3]
            ph = data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]
            idx = np.argmax(loglikelihood)
    
            t0_best = data[idx,0]
            mej_best = 10**data[idx,1]
            vej_best = data[idx,2]
            th_best = data[idx,3]
            ph_best = data[idx,4]
            zp_best = data[idx,5]
    
            tmag, lbol, mag = KaKy2016_model_ejecta(mej_best,vej_best,th_best,ph_best)
    
    elif opts.model == "DiUj2017":
    
        if opts.doMasses:
            if opts.doEOSFit:
    
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                th = data[:,5]
                ph = data[:,6]
                zp = data[:,7]
                loglikelihood = data[:,8]
                idx = np.argmax(loglikelihood)
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                c1_best = data[idx,2]
                m2_best = data[idx,3]
                c2_best = data[idx,4]
                th_best = data[idx,5]
                ph_best = data[idx,6]
                zp_best = data[idx,7]
                mb1_best = mb1[idx]
                mb2_best = mb2[idx]
    
                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","th","ph","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0] = data[:,0]
                data_new[:,1] = 1/q
                data_new[:,2] = mchirp
                data_new[:,3] = data[:,2]
                data_new[:,4] = data[:,4]
                data_new[:,5] = data[:,5]
                data_new[:,6] = data[:,6]
                data_new[:,7] = data[:,7]
                data = data_new
    
            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                th = data[:,7]
                ph = data[:,8]
                zp = data[:,9]
                loglikelihood = data[:,10]
                idx = np.argmax(loglikelihood)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                mb1_best = data[idx,2]
                c1_best = data[idx,3]
                m2_best = data[idx,4]
                mb2_best = data[idx,5]
                c2_best = data[idx,6]
                th_best = data[idx,7]
                ph_best = data[idx,8]
                zp_best = data[idx,9]
    
            tmag, lbol, mag = DiUj2017_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,th_best,ph_best)
        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            th = data[:,3]
            ph = data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]
            idx = np.argmax(loglikelihood)
    
            t0_best = data[idx,0]
            mej_best = 10**data[idx,1]
            vej_best = data[idx,2]
            th_best = data[idx,3]
            ph_best = data[idx,4]
            zp_best = data[idx,5]
            tmag, lbol, mag = DiUj2017_model_ejecta(mej_best,vej_best,th_best,ph_best)
    
    elif opts.model == "BaKa2016":

        if opts.doMasses:
            if opts.doEOSFit:

                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                zp = data[:,5]
                loglikelihood = data[:,6]
                idx = np.argmax(loglikelihood)
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)

                t0_best = data[idx,0]
                m1_best = data[idx,1]
                c1_best = data[idx,2]
                m2_best = data[idx,3]
                c2_best = data[idx,4]
                zp_best = data[idx,5]
                mb1_best = mb1[idx]
                mb2_best = mb2[idx]

                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0] = data[:,0]
                data_new[:,1] = 1/q
                data_new[:,2] = mchirp
                data_new[:,3] = data[:,2]
                data_new[:,4] = data[:,4]
                data_new[:,5] = data[:,5]
                data = data_new

            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                zp = data[:,7]
                loglikelihood = data[:,8]
                idx = np.argmax(loglikelihood)

                t0_best = data[idx,0]
                m1_best = data[idx,1]
                mb1_best = data[idx,2]
                c1_best = data[idx,3]
                m2_best = data[idx,4]
                mb2_best = data[idx,5]
                c2_best = data[idx,6]
                zp_best = data[idx,7]

            tmag, lbol, mag = BaKa2016_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best)

        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            zp = data[:,3]
            loglikelihood = data[:,4]
            idx = np.argmax(loglikelihood)

            t0_best = data[idx,0]
            mej_best = 10**data[idx,1]
            vej_best = data[idx,2]
            zp_best = data[idx,3]

            tmag, lbol, mag = BaKa2016_model_ejecta(mej_best,vej_best)

    elif opts.model == "Ka2017":
        if opts.doMasses:
            if opts.doEOSFit or opts.doBNSFit:

                t0, m1, c1, m2, c2, Xlan, zp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
                idx = np.argmax(loglikelihood)
                mb1, mb2 = lightcurve_utils.EOSfit(m1,c1), lightcurve_utils.EOSfit(m2,c2)

                t0_best, m1_best, c1_best, m2_best, c2_best, Xlan_best, zp_best, mb1_best, mb2_best = data[idx,0], data[idx,1], data[idx,2], data[idx,3], data[idx,4], 10**data[idx,5], data[idx,6], mb1[idx], mb2[idx]

                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","Xlan","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$","Xlan","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0], data_new[:,1], data_new[:,2], data_new[:,3], data_new[:,4], data_new[:,5], data_new[:,6] = data[:,0], 1/q, mchirp, data[:,2], data[:,4], data[:,5], data[:,6]
                data = data_new

            else:
                t0, m1, mb1, c1, m2, mb2, c2, Xlan, zp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9]
                idx = np.argmax(loglikelihood)
                t0_best, m1_best, mb1_best, c1_best, m2_best, mb2_best, c2_best, Xlan_best, zp_best  = data[idx,0], data[idx,1], data[idx,2], data[idx,3], data[idx,4], data[idx,5], data[idx,6], data[idx,7], data[idx,8]

            tmag, lbol, mag = Ka2017_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,Xlan_best)

        elif opts.doEjecta:
            t0, mej, vej, Xlan, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], data[:,4], data[:,5]
            idx = np.argmax(loglikelihood)
            t0_best, mej_best, vej_best, Xlan_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], data[idx,4]
            tmag, lbol, mag = Ka2017_model_ejecta(mej_best,vej_best,Xlan_best)
    elif opts.model == "Ka2017inc":
        if opts.doEjecta:
            t0, mej, vej, Xlan, iota, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], data[:,4], data[:,5], data[:,6]
            idx = np.argmax(loglikelihood)
            t0_best, mej_best, vej_best, Xlan_best, iota_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], data[idx,4], data[idx,5]
            tmag, lbol, mag = Ka2017inc_model_ejecta(mej_best,vej_best,Xlan_best,iota_best)
    elif opts.model == "Ka2017_A":
        if opts.doEjecta:
            t0, mej, vej, Xlan, A, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], 10**data[:,4], data[:,5], data[:,6]
            idx = np.argmax(loglikelihood)
            t0_best, mej_best, vej_best, Xlan_best, A_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], 10**data[idx,4], data[idx,5]
            tmag, lbol, mag = Ka2017_A_model(mej_best,vej_best,Xlan_best,A_best)
    elif opts.model == "Ka2017x2":
        if opts.doEjecta:
            t0, mej_1, vej_1, Xlan_1, mej_2, vej_2, Xlan_2, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], 10**data[:,4], data[:,5], 10**data[:,6], data[:,7], data[:,8]
            idx = np.argmax(loglikelihood)
            t0_best, mej_1_best, vej_1_best, Xlan_1_best, mej_2_best, vej_2_best, Xlan_2_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], 10**data[idx,4], data[idx,5], 10**data[idx,6], data[idx,7]
            tmag, lbol, mag = Ka2017x2_model_ejecta(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best)
    elif opts.model == "Ka2017x2inc":
        if opts.doEjecta:
            t0, mej_1, vej_1, Xlan_1, mej_2, vej_2, Xlan_2, iota, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], 10**data[:,4], data[:,5], 10**data[:,6], data[:,7], data[:,8], data[:,9]
            idx = np.argmax(loglikelihood)
            t0_best, mej_1_best, vej_1_best, Xlan_1_best, mej_2_best, vej_2_best, Xlan_2_best, iota_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], 10**data[idx,4], data[idx,5], 10**data[idx,6], data[idx,7], data[idx,8]
            tmag, lbol, mag = Ka2017x2inc_model_ejecta(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,iota_best)
    elif opts.model == "Ka2017x3":
        if opts.doEjecta:
            t0, mej_1, vej_1, Xlan_1, mej_2, vej_2, Xlan_2, mej_3, vej_3, Xlan_3, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], 10**data[:,4], data[:,5], 10**data[:,6], 10**data[:,7], data[:,8], 10**data[:,9],  data[:,10], data[:,11]
            idx = np.argmax(loglikelihood)
            t0_best, mej_1_best, vej_1_best, Xlan_1_best, mej_2_best, vej_2_best, Xlan_2_best, mej_3_best, vej_3_best, Xlan_3_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], 10**data[idx,4], data[idx,5], 10**data[idx,6], 10**data[idx,7], data[idx,8], 10**data[idx,9], data[idx,10]
            tmag, lbol, mag = Ka2017x3_model_ejecta(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,mej_3_best,vej_3_best,Xlan_3_best)
    elif opts.model == "Ka2017x3inc":
        if opts.doEjecta:
            t0, mej_1, vej_1, Xlan_1, mej_2, vej_2, Xlan_2, mej_3, vej_3, Xlan_3, iota, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], 10**data[:,4], data[:,5], 10**data[:,6], 10**data[:,7], data[:,8], 10**data[:,9],  data[:,10], data[:,11], data[:,12]
            idx = np.argmax(loglikelihood)
            t0_best, mej_1_best, vej_1_best, Xlan_1_best, mej_2_best, vej_2_best, Xlan_2_best, mej_3_best, vej_3_best, Xlan_3_best, iota_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], 10**data[idx,4], data[idx,5], 10**data[idx,6], 10**data[idx,7], data[idx,8], 10**data[idx,9], data[idx,10], data[idx,11]
            tmag, lbol, mag = Ka2017x3inc_model_ejecta(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,mej_3_best,vej_3_best,Xlan_3_best,iota_best)
    elif opts.model == "RoFe2017":
        if opts.doMasses:
            if opts.doEOSFit:

                t0, m1, c1, m2, c2, Ye, zp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
                idx = np.argmax(loglikelihood)
                mb1, mb2 = lightcurve_utils.EOSfit(m1,c1), lightcurve_utils.EOSfit(m2,c2)

                t0_best, m1_best, c1_best, m2_best, c2_best, Ye_best, zp_best, mb1_best, mb2_best = data[idx,0], data[idx,1], data[idx,2], data[idx,3], data[idx,4], data[idx,5], data[idx,6], mb1[idx], mb2[idx]

                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","ye","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$","Ye","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0], data_new[:,1], data_new[:,2], data_new[:,3], data_new[:,4], data_new[:,5], data_new[:,6] = data[:,0], 1/q, mchirp, data[:,2], data[:,4], data[:,5], data[:,6]
                data = data_new

            else:
                t0, m1, mb1, c1, m2, mb2, c2, Ye, zp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9]
                idx = np.argmax(loglikelihood)
                t0_best, m1_best, mb1_best, c1_best, m2_best, mb2_best, c2_best, Ye_best, zp_best  = data[idx,0], data[idx,1], data[idx,2], data[idx,3], data[idx,4], data[idx,5], data[idx,6], data[idx,7], data[idx,8]

            tmag, lbol, mag = RoFe2017_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,Ye_best)

        elif opts.doEjecta:
            t0, mej, vej, Ye, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
            idx = np.argmax(loglikelihood)
            t0_best, mej_best, vej_best, Ye_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], data[idx,3], data[idx,4]
            tmag, lbol, mag = RoFe2017_model_ejecta(mej_best,vej_best,Ye_best)

    elif opts.model == "Me2017":
    
        if opts.doMasses:
            if opts.doEOSFit:
    
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                beta = data[:,5]
                kappa_r = 10**data[:,6]
                zp = data[:,7]
                loglikelihood = data[:,8]
                idx = np.argmax(loglikelihood)
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                c1_best = data[idx,2]
                m2_best = data[idx,3]
                c2_best = data[idx,4]
                beta_best = data[idx,5]
                kappa_r_best = 10**data[idx,6]
                zp_best = data[idx,7]
                mb1_best = mb1[idx]
                mb2_best = mb2[idx]
    
                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$",r"$\beta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0] = data[:,0]
                data_new[:,1] = 1/q
                data_new[:,2] = mchirp
                data_new[:,3] = data[:,2]
                data_new[:,4] = data[:,4]
                data_new[:,5] = data[:,5]
                data_new[:,6] = data[:,6]
                data_new[:,7] = data[:,7]
                data = data_new
    
            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                beta = data[:,7]
                kappa_r = 10**data[:,8]
                zp = data[:,9]
                loglikelihood = data[:,10]
                idx = np.argmax(loglikelihood)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                mb1_best = data[idx,2]
                c1_best = data[idx,3]
                m2_best = data[idx,4]
                mb2_best = data[idx,5]
                c2_best = data[idx,6]
                beta_best = data[idx,7]
                kappa_r_best = 10**data[idx,8]
                zp_best = data[idx,9]
    
            tmag, lbol, mag = Me2017_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,beta_best,kappa_r_best)
    
        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            beta = data[:,3]
            kappa_r = 10**data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]
            idx = np.argmax(loglikelihood)
    
            t0_best = data[idx,0]
            mej_best = 10**data[idx,1]
            vej_best = data[idx,2]
            beta_best = data[idx,3]
            kappa_r_best = 10**data[idx,4]
            zp_best = data[idx,5]
    
            tmag, lbol, mag = Me2017_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best)
  
    elif opts.model == "Me2017_A":
        t0 = data[:,0]
        mej = 10**data[:,1]
        vej = data[:,2]
        beta = data[:,3]
        kappa_r = 10**data[:,4]
        A = 10**data[:,5]
        zp = data[:,6]
        loglikelihood = data[:,6]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        mej_best = 10**data[idx,1]
        vej_best = data[idx,2]
        beta_best = data[idx,3]
        kappa_r_best = 10**data[idx,4]
        A_best = 10**data[idx,5]
        zp_best = data[idx,6]

        tmag, lbol, mag = Me2017_A_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best,A_best)
 
    elif opts.model == "Me2017x2":
        if opts.doEjecta:
            t0 = data[:,0]
            mej_1 = 10**data[:,1]
            vej_1 = data[:,2]
            beta_1 = data[:,3]
            kappa_r_1 = 10**data[:,4]
            mej_2 = 10**data[:,5]
            vej_2 = data[:,6]
            beta_2 = data[:,7]
            kappa_r_2 = 10**data[:,8]
            zp = data[:,9]
            loglikelihood = data[:,10]
            idx = np.argmax(loglikelihood)

            t0_best = data[idx,0]
            mej_1_best = 10**data[idx,1]
            vej_1_best = data[idx,2]
            beta_1_best = data[idx,3]
            kappa_r_1_best = 10**data[idx,4]
            mej_2_best = 10**data[idx,5]
            vej_2_best = data[idx,6]
            beta_2_best = data[idx,7]
            kappa_r_2_best = 10**data[idx,8]
            zp_best = data[idx,9]

            tmag, lbol, mag = Me2017x2_model_ejecta(mej_1_best,vej_1_best,beta_1_best,kappa_r_1_best,mej_2_best,vej_2_best,beta_2_best,kappa_r_2_best) 

    elif opts.model == "WoKo2017":
    
        if opts.doMasses:
            if opts.doEOSFit:
    
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                beta = data[:,5]
                kappa_r = 10**data[:,6]
    
                zp = data[:,7]
                loglikelihood = data[:,8]
                idx = np.argmax(loglikelihood)
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                c1_best = data[idx,2]
                m2_best = data[idx,3]
                c2_best = data[idx,4]
                theta_r_best = data[idx,5]
                kappa_r_best = 10**data[idx,6]
                zp_best = data[idx,7]
                mb1_best = mb1[idx]
                mb2_best = mb2[idx]
    
                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$",r"$\theta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0] = data[:,0]
                data_new[:,1] = 1/q
                data_new[:,2] = mchirp
                data_new[:,3] = data[:,2]
                data_new[:,4] = data[:,4]
                data_new[:,5] = data[:,5]
                data_new[:,6] = data[:,6]
                data_new[:,7] = data[:,7]
                data = data_new
    
            else:
    
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                theta_r = data[:,7]
                kappa_r = 10**data[:,8]
                zp = data[:,9]
                loglikelihood = data[:,10]
                idx = np.argmax(loglikelihood)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                mb1_best = data[idx,2]
                c1_best = data[idx,3]
                m2_best = data[idx,4]
                mb2_best = data[idx,5]
                c2_best = data[idx,6]
                theta_r_best = data[idx,7]
                kappa_r_best = 10**data[idx,8]
                zp_best = data[idx,9]
    
            tmag, lbol, mag = WoKo2017_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,theta_r_best,kappa_r_best)
    
        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            theta_r = data[:,3]
            kappa_r = 10**data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]
            idx = np.argmax(loglikelihood)
    
            t0_best = data[idx,0]
            mej_best = 10**data[idx,1]
            vej_best = data[idx,2]
            theta_r_best = data[idx,3]
            kappa_r_best = 10**data[idx,4]
            zp_best = data[idx,5]
    
            tmag, lbol, mag = WoKo2017_model_ejecta(mej_best,vej_best,theta_r_best,kappa_r_best)
    
    elif opts.model == "SmCh2017":
    
        if opts.doMasses:
            if opts.doEOSFit:
    
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                slope_r = data[:,5]
                kappa_r = 10**data[:,6]
                zp = data[:,7]
                loglikelihood = data[:,8]
                idx = np.argmax(loglikelihood)
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                c1_best = data[idx,2]
                m2_best = data[idx,3]
                c2_best = data[idx,4]
                slope_r_best = data[idx,5]
                kappa_r_best = 10**data[idx,6]
                zp_best = data[idx,7]
                mb1_best = mb1[idx]
                mb2_best = mb2[idx]
    
                data_new = np.zeros(data.shape)
                parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$",r"$\beta$",r"${\rm log}_{10} \kappa_{\rm r}$","ZP"]
                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                data_new[:,0] = data[:,0]
                data_new[:,1] = 1/q
                data_new[:,2] = mchirp
                data_new[:,3] = data[:,2]
                data_new[:,4] = data[:,4]
                data_new[:,5] = data[:,5]
                data_new[:,6] = data[:,6]
                data_new[:,7] = data[:,7]
                data = data_new
    
            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                slope_r = data[:,7]
                kappa_r = 10**data[:,8]
                zp = data[:,9]
                loglikelihood = data[:,10]
                idx = np.argmax(loglikelihood)
    
                t0_best = data[idx,0]
                m1_best = data[idx,1]
                mb1_best = data[idx,2]
                c1_best = data[idx,3]
                m2_best = data[idx,4]
                mb2_best = data[idx,5]
                c2_best = data[idx,6]
                slope_r_best = data[idx,7]
                kappa_r_best = 10**data[idx,8]
                zp_best = data[idx,9]
    
            tmag, lbol, mag = SmCh2017_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,slope_r_best,kappa_r_best)
    
        elif opts.doEjecta:
            #idx = np.where((data[:,1]>=-2.0) & (data[:,3]<=0.5))[0]
            #data = data[idx,:]

            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            slope_r = data[:,3]
            kappa_r = 10**data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]
            idx = np.argmax(loglikelihood)
    
            t0_best = data[idx,0]
            mej_best = 10**data[idx,1]
            vej_best = data[idx,2]
            slope_r_best = data[idx,3]
            kappa_r_best = 10**data[idx,4]
            zp_best = data[idx,5]
    
            tmag, lbol, mag = SmCh2017_model_ejecta(mej_best,vej_best,slope_r_best,kappa_r_best)
   
            data = np.delete(data,-5,1)
            labels.pop(-4)
    
    elif opts.model == "SN":
    
        t0 = data[:,0]
        z = data[:,1]
        x0 = data[:,2]
        x1 = data[:,3]
        c = data[:,4]
        zp = data[:,5]
        loglikelihood = data[:,6]
        idx = np.argmax(loglikelihood)
    
        t0_best = data[idx,0]
        z_best = data[idx,1]
        x0_best = data[idx,2]
        x1_best = data[idx,3]
        c_best = data[idx,4]
        zp_best = data[idx,5]
    
        tmag, lbol, mag = sn_model(z_best,0.0,x0_best,x1_best,c_best)

    elif opts.model == "TrPi2018":

        t0 = data[:,0]
        theta_v = data[:,1]
        E0 = 10**data[:,2]
        theta_c = data[:,3]
        theta_w = data[:,4]
        n = 10**data[:,5]
        p = data[:,6]
        epsilon_E = 10**data[:,7]
        epsilon_B = 10**data[:,8]
        zp = data[:,9]
        loglikelihood = data[:,10]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        theta_v_best = data[idx,1]
        E0_best = 10**data[idx,2]
        theta_c_best = data[idx,3]
        theta_w_best = data[idx,4]
        n_best = 10**data[idx,5]
        p_best = data[idx,6]
        epsilon_E_best = 10**data[idx,7]
        epsilon_B_best = 10**data[idx,8]
        zp_best = data[idx,9]

        tmag, lbol, mag = TrPi2018_model(theta_v_best, E0_best, theta_c_best, theta_w_best, n_best, p_best, epsilon_E_best, epsilon_B_best)

    elif opts.model == "Ka2017_TrPi2018":

        t0, mej, vej, Xlan, theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], data[:,4], 10**data[:,5], data[:,6], data[:,7], 10**data[:,8], data[:,9], 10**data[:,10], 10**data[:,11], data[:,12], data[:,13]
        idx = np.argmax(loglikelihood)

        t0_best, mej_best, vej_best, Xlan_best, theta_v_best, E0_best, theta_c_best, theta_w_best, n_best, p_best, epsilon_E_best, epsilon_B_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], data[idx,4], 10**data[idx,5], data[idx,6], data[idx,7], 10**data[idx,8], data[idx,9], 10**data[idx,10], 10**data[idx,11], data[idx,12]

        tmag, lbol, mag = Ka2017_TrPi2018_model(mej_best, vej_best, Xlan_best, theta_v_best, E0_best, theta_c_best, theta_w_best, n_best, p_best, epsilon_E_best, epsilon_B_best)

    elif opts.model == "Ka2017_TrPi2018_A":

        t0, mej, vej, Xlan, theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B, A, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], data[:,4], 10**data[:,5], data[:,6], data[:,7], 10**data[:,8], data[:,9], 10**data[:,10], 10**data[:,11], 10**data[:,12], data[:,13], data[:,14]
        idx = np.argmax(loglikelihood)

        t0_best, mej_best, vej_best, Xlan_best, theta_v_best, E0_best, theta_c_best, theta_w_best, n_best, p_best, epsilon_E_best, epsilon_B_best, A_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], data[idx,4], 10**data[idx,5], data[idx,6], data[idx,7], 10**data[idx,8], data[idx,9], 10**data[idx,10], 10**data[idx,11], 10**data[idx,12], data[idx,13]

        tmag, lbol, mag = Ka2017_TrPi2018_A_model(mej_best, vej_best, Xlan_best, theta_v_best, E0_best, theta_c_best, theta_w_best, n_best, p_best, epsilon_E_best, epsilon_B_best, A_best)

    if opts.model == "KaKy2016":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l,m,n,o in zip(t0,q,chi_eff,mns,mb,c,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,q_best,chi_best,mns_best,mb_best,c_best,zp_best))
            fid.close()
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l in zip(t0,mej,vej,zp):
                fid.write('%.5f %.5f %.5f %.5f\n'%(i,j,k,l))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,zp_best))
            fid.close()
    
    elif opts.model == "DiUj2017":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()
    
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n in zip(t0,mej,vej,th,ph,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,th_best,ph_best,zp_best))
            fid.close()

    elif opts.model == "BaKa2016":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()   
 
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l in zip(t0,mej,vej,zp):
                fid.write('%.5f %.5f %.5f %.5f\n'%(i,j,k,l))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,zp_best))
            fid.close()

    elif opts.model == "Ka2017":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()

        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m in zip(t0,mej,vej,Xlan,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,Xlan_best,zp_best))
            fid.close()
    elif opts.model == "Ka2017x2":
        if opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,zp_best))
            fid.close()
    elif opts.model == "Ka2017x2inc":
        if opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p, q in zip(t0,mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,iota,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,iota_best,zp_best))
            fid.close()
    elif opts.model == "Ka2017x3":
        if opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p, q, r, s in zip(t0,mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,mej_3,vej_3,Xlan_3,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p,q,r,s))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,mej_3_best,vej_3_best,Xlan_3_best,zp_best))
            fid.close()
    elif opts.model == "Ka2017x3inc":
        if opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p, q, r, s, t in zip(t0,mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,mej_3,vej_3,Xlan_3,iota,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p,q,r,s,t))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best,mej_3_best,vej_3_best,Xlan_3_best,iota_best,zp_best))
            fid.close()
    elif opts.model == "Me2017":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n in zip(t0,mej,vej,beta,kappa_r,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,beta_best,kappa_r_best,zp_best))
            fid.close()
            best = [t0_best,np.log10(mej_best),vej_best,beta_best,kappa_r_best,zp_best]
    elif opts.model == "Me2017x2": 
        if opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p, q, r in zip(t0,mej_1,vej_1,beta_1,kappa_r_1,mej_2,vej_2,beta_2,kappa_r_2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p,q,r))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_1_best,vej_1_best,beta_1_best,kappa_r_1_best,mej_2_best,vej_2_best,beta_2_best,kappa_r_2_best,zp_best))
            fid.close()
    elif opts.model == "RoFe2017":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()

            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m in zip(t0,mej,vej,Ye,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m))
            fid.close()
 
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,Ye_best,zp_best))
            fid.close()

    elif opts.model == "WoKo2017":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n in zip(t0,mej,vej,theta_r,kappa_r,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,theta_r_best,kappa_r_best,zp_best))
            fid.close()
    
    elif opts.model == "SmCh2017":
        if opts.doMasses:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
            fid.close()
        elif opts.doEjecta:
            filename = os.path.join(plotDir,'samples.dat')
            fid = open(filename,'w+')
            for i, j, k, l, m, n in zip(t0,mej,vej,slope_r,kappa_r,zp):
                fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
            fid.close()
    
            filename = os.path.join(plotDir,'best.dat')
            fid = open(filename,'w')
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,slope_r_best,kappa_r_best,zp_best))
            fid.close()
            best = [t0_best,np.log10(mej_best),slope_r_best,kappa_r_best,zp_best] 
    elif opts.model == "SN":
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l,m,n in zip(t0,z,x0,x1,c,zp):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
        fid.close()
    
        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,z_best,x0_best,x1_best,c_best,zp_best))
        fid.close()

    elif opts.model == "TrPi2018":
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l,m,n,o,p,q,r in zip(t0,theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B,zp):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p,q,r))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,theta_v_best, E0_best, theta_c_best, theta_w_best, n_best, p_best, epsilon_E_best, epsilon_B_best,zp_best))
        fid.close()

    return data, tmag, lbol, mag, t0_best, zp_best, n_params, labels, best    
    
