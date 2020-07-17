from gwemlightcurves.KNModels import KNTable
import numpy as np
from scipy.interpolate import interpolate as interp
from gwemlightcurves import lightcurve_utils
import os 


np.random.seed(0)
tini = 0.1
tmax = 50.0
dt = 0.1

vmin = 0.02
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 1

beta = 3.0
kappa_r = 0.1
slope_r = -1.2
theta_r = 0.0
Ye = 0.3

ModelPath = "./output/svdmodels"

tt = np.arange(0.05, 7.0, 0.05)
filters = ["u","g","r","i","z","y","J","H","K"]
filts = ["u","g","r","i","z","y","J","H","K"]
magidxs = [0,1,2,3,4,5,6,7,8]


class EM_Counterpart():
        def __init__(self, input_samples, Xlan_fixed, phi_fixed, eostype = "spec"):
                self.samples = KNTable.initialize_object(input_samples)
                
                m1s, m2s, dists_mbta = [], [], []
                lambda1s, lambda2s, chi_effs = [], [], []
                Xlans = []
                mbnss = []
                if eostype == "gp":
                        # read Phil + Reed's EOS files
                        filenames = glob.glob("/home/philippe.landry/gw170817eos/gp/macro/MACROdraw-*-0.csv")
                        idxs = []
                        for filename in filenames:
                                filenameSplit = filename.replace(".csv","").split("/")[-1].split("-")
                                idxs.append(int(filenameSplit[1]))
                        idxs = np.array(idxs)
                elif eostype == "Sly":
                        eosname = "SLy"
                        eos = EOS4ParameterPiecewisePolytrope(eosname)

                
     
                for ii, row in enumerate(self.samples):
                        print(ii) 
                        m1, m2, dist_mbta, chi_eff = row["m1"], row["m2"], row["dist_mbta"], row["chi_eff"]
                        nsamples = 30
                        if eostype == "spec":
                                indices = np.random.randint(0, 2395, size=nsamples)
                        elif eostype == "gp":
                                indices = np.random.randint(0, len(idxs), size=nsamples)
                        for jj in range(nsamples):
                                if (eostype == "spec") or (eostype == "gp"):
                                        index = indices[jj] 
                                # samples lambda's from Phil + Reed's files
                                if eostype == "spec":
                                        eospath = "/home/philippe.landry/gw170817eos/spec/macro/macro-spec_%dcr.csv" % index
                                        data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                                        marray, larray = data_out["M"], data_out["Lambda"]
                                        f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                                        lambda1, lambda2 = f(m1), f(m2) 
                                        mbns = np.max(marray)   
                                elif eostype == "gp":
                                        lambda1, lambda2 = 0.0, 0.0
                                        phasetr = 0
                                        while (lambda1==0.0) or (lambda2 == 0.0):
                                                eospath = "/home/philippe.landry/gw170817eos/gp/macro/MACROdraw-%06d-%d.csv" % (idxs[index], phasetr)
                                                if not os.path.isfile(eospath):
                                                        break
                                                data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                                                marray, larray = data_out["M"], data_out["Lambda"]
                                                f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                                                lambda1_tmp, lambda2_tmp = f(m1), f(m2)
                                                if (lambda1_tmp>0) and (lambda1==0.0):
                                                        lambda1 = lambda1_tmp
                                                if (lambda2_tmp>0) and (lambda2 == 0.0):
                                                        lambda2 = lambda2_tmp
                                                phasetr = phasetr + 1
                                                mbns = np.max(marray)
                                elif eostype == "Sly":
                                        lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)

                                m1s.append(m1)
                                m2s.append(m2)
                                dists_mbta.append(dist_mbta)
                                lambda1s.append(lambda1)
                                lambda2s.append(lambda2)
                                chi_effs.append(chi_eff)
                                mbnss.append(mbns)
                                np.random.uniform(0)

                       

  
                Xlans = [10**Xlan_fixed] * len(self.samples) * nsamples
                phis = [phi_fixed] * len(self.samples) * nsamples 
                thetas = 180. * np.arccos(np.random.uniform(-1., 1., len(self.samples) * nsamples)) / np.pi
                idx_thetas = np.where(thetas > 90.)[0]
                thetas[idx_thetas] = 180. - thetas[idx_thetas]
                thetas = list(thetas)

       
                # make final arrays of masses, distances, lambdas, spins, and lanthanide fractions 
                data = np.vstack((m1s,m2s,dists_mbta,lambda1s,lambda2s,chi_effs,thetas, phis, mbnss,Xlans)).T
                self.samples = KNTable(data, names=('m1', 'm2', 'dist_mbta', 'lambda1', 'lambda2','chi_eff','theta', 'phi', 'mbns', "Xlan"))  
                
                self.samples = self.samples.calc_tidal_lambda(remove_negative_lambda=True)
                self.samples = self.samples.calc_compactness(fit=True)
                self.samples = self.samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
                print(self.samples)

        def calc_ejecta(self, model_KN):
                idx1 = np.where((self.samples['m1'] <= self.samples['mbns']) & (self.samples['m2'] <= self.samples['mbns']))[0]
                idx2 = np.where((self.samples['m1'] > self.samples['mbns']) & (self.samples['m2'] <= self.samples['mbns']))[0]
                idx3 = np.where((self.samples['m1'] > self.samples['mbns']) & (self.samples['m2'] > self.samples['mbns']))[0]

                

                mej, vej = np.zeros(self.samples['m1'].shape), np.zeros(self.samples['m1'].shape)
                from gwemlightcurves.EjectaFits.CoDi2019 import calc_meje, calc_vej
                # calc the mass of ejecta
                mej1 = calc_meje(self.samples['m1'], self.samples['c1'], self.samples['m2'], self.samples['c2'])
                # calc the velocity of ejecta
                vej1 = calc_vej(self.samples['m1'], self.samples['c1'], self.samples['m2'], self.samples['c2'])

                self.samples['mchirp'], self.samples['eta'], self.samples['q'] = lightcurve_utils.ms2mc(self.samples['m1'], self.samples['m2'])
                self.samples['q'] = 1.0 / self.samples['q']

                from gwemlightcurves.EjectaFits.KrFo2019 import calc_meje, calc_vave
                # calc the mass of ejecta
       
        
                mej2 = calc_meje(self.samples['q'], self.samples['chi_eff'], self.samples['c2'], self.samples['m2'])
                # calc the velocity of ejecta
                vej2 = calc_vave(self.samples['q'])
       

                # calc the mass of ejecta
                mej3 = np.zeros(self.samples['m1'].shape)
                # calc the velocity of ejecta
                vej3 = np.zeros(self.samples['m1'].shape) + 0.2
        
                mej[idx1], vej[idx1] = mej1[idx1], vej1[idx1]
                mej[idx2], vej[idx2] = mej2[idx2], vej2[idx2]
                mej[idx3], vej[idx3] = mej3[idx3], vej3[idx3]

                print("(mej[1284], self.samples[1284]['m1'], self.samples[1284]['m2'], self.samples[1284]['c1'], self.samples[1284]['c2'], self.samples[1284]['q'], self.samples[1284]['chi_eff'])")
                print((mej[1284], self.samples[1284]['m1'], self.samples[1284]['m2'], self.samples[1284]['c1'], self.samples[1284]['c2'], self.samples[1284]['q'], self.samples[1284]['chi_eff']))

                self.samples['mej'] = mej
                self.samples['vej'] = vej
     

                # Add draw from a gaussian in the log of ejecta mass with 1-sigma size of 70%
                erroropt = 'none'
                if erroropt == 'none':
                        print("Not applying an error to mass ejecta")
                elif erroropt == 'log':
                        self.samples['mej'] = np.power(10.,np.random.normal(np.log10(self.samples['mej']), 0.236))
                elif erroropt == 'lin':
                        self.samples['mej'] = np.random.normal(self.samples['mej'], 0.72*self.samples['mej'])
                elif erroropt == 'loggauss':
                        self.samples['mej'] = np.power(10.,np.random.normal(np.log10(self.samples['mej']), 0.312))
  

                idx = np.where(self.samples['mej'] <= 0)[0]
                self.samples['mej'][idx] = 1e-11
        
       
                if (model_KN == "Bu2019inc"):  
                        idx = np.where(self.samples['mej'] <= 1e-6)[0]
                        self.samples['mej'][idx] = 1e-11
                elif (model_KN == "Ka2017"):
                        idx = np.where(self.samples['mej'] <= 1e-3)[0]
                        self.samples['mej'][idx] = 1e-11
           
        
                print("Probability of having ejecta")
                print(100 * (len(self.samples) - len(idx)) /len(self.samples))
                return self.samples['mej']	                 

        def calc_lightcurve(self, model_KN):             
                self.samples = self.samples.downsample(Nsamples=1000) 
                #add default values from above to table
                self.samples['tini'] = tini
                self.samples['tmax'] = tmax
                self.samples['dt'] = dt
                self.samples['vmin'] = vmin
                self.samples['th'] = th
                self.samples['ph'] = ph
                self.samples['kappa'] = kappa
                self.samples['eps'] = eps
                self.samples['alp'] = alp
                self.samples['eth'] = eth
                self.samples['flgbct'] = flgbct
                self.samples['beta'] = beta
                self.samples['kappa_r'] = kappa_r
                self.samples['slope_r'] = slope_r
                self.samples['theta_r'] = theta_r
                self.samples['Ye'] = Ye

                kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
                kwargs["doAB"] = True
                kwargs["doSpec"] = False        

                model_tables = {}
                models = model_KN.split(",")
                for model in models:
                        model_tables[model] = KNTable.model(model, self.samples, **kwargs)
                        if (model_KN == "Bu2019inc"):
                                idx = np.where(model_tables[model]['mej'] <= 1e-6)[0]
                                model_tables[model]['mag'][idx] = 10.
                                model_tables[model]['lbol'][idx] = 1e30
                        elif (model_KN == "Ka2017"):
                                idx = np.where(model_tables[model]['mej'] <= 1e-3)[0]
                                model_tables[model]['mag'][idx] = 10.
                                model_tables[model]['lbol'][idx] = 1e30
          
                for model in models:
                        model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model]) 
             

                mag_all = {}
                app_mag_all_mbta = {}
                lbol_all = {}
                
                for model in models:
                        mag_all[model] = {}
                        app_mag_all_mbta[model] = {}
                        lbol_all[model] = {}
                        lbol_all[model] = np.empty((0,len(tt)), float)
                        for filt, magidx in zip(filts,magidxs):
                                mag_all[model][filt] = np.empty((0,len(tt)))
                                app_mag_all_mbta[model][filt] = np.empty((0,len(tt)))

                peak_mags_all = {}
                for model in models:
                        model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model])
                        for row in model_tables[model]:
                                t, lbol, mag = row["t"], row["lbol"], row["mag"]
                                dist_mbta = row['dist_mbta']

                                if np.sum(lbol) == 0.0:
                                        #print "No luminosity..."
                                        continue

                                allfilts = True
                                for filt, magidx in zip(filts,magidxs):
                                        idx = np.where(~np.isnan(mag[magidx]))[0]
                                        if len(idx) == 0:
                                                allfilts = False
                                                break
                                if not allfilts: continue
                                for filt, magidx in zip(filts,magidxs):
                                        idx = np.where(~np.isnan(mag[magidx]))[0] 
                                        f = interp.interp1d(t[idx], mag[magidx][idx], fill_value='extrapolate')
                                        maginterp = f(tt)
                                        app_maginterp_mbta = maginterp + 5*(np.log10((dist_mbta)*1e6) - 1)
                                        mag_all[model][filt] = np.append(mag_all[model][filt],[maginterp],axis=0)
                                        app_mag_all_mbta[model][filt] = np.append(app_mag_all_mbta[model][filt],[app_maginterp_mbta],axis=0)
                                idx = np.where((~np.isnan(np.log10(lbol))) & ~(lbol==0))[0]
                                f = interp.interp1d(t[idx], np.log10(lbol[idx]), fill_value='extrapolate')
                                lbolinterp = 10**f(tt)
                                lbol_all[model] = np.append(lbol_all[model],[lbolinterp],axis=0)

                return (mag_all, app_mag_all_mbta, lbol_all)                                  


        def write_output(self, output_path, model_KN, mej, mag_all, app_mag_mbta_all, lbol_all):         
                output_path = os.path.join(output_path, model_KN)        
                if (not os.path.isdir(output_path)):
                        os.mkdir(output_path)
                with open(os.path.join(output_path, "ejecta.txt"), 'w') as f:          
                        f.write("#HasEjecta(%) mej_percentile(10%) mej_percentile(50%) mej_percentile(90%)" + "\n")
                        if (model_KN == "Bu2019inc"):
                                idx = np.where(mej <= 1e-6)[0]
                        elif (model_KN == "Ka2017"):
                                idx = np.where(mej <= 1e-3)[0]
                        
                        f.write(str(100 * (len(mej) - len(idx)) /len(mej)) + " " + str(np.percentile(mej, 10)) + " " + str(np.percentile(mej, 50)) + " " + str(np.percentile(mej, 90)))
                        f.close  
                with open(os.path.join(output_path, "lbol.txt"), 'w') as f: 
                        f.write("#lbol_percentile(10%) lbol_percentile(50%) lbol_percentile(90%)" + "\n")
                        f.write(str(np.percentile(lbol_all[model_KN], 10)) + " " + str(np.percentile(lbol_all[model_KN], 50)) + " " + str(np.percentile(lbol_all[model_KN], 90)))
                        f.close
                for filt in filts:
                        magmed = np.percentile(mag_all[model_KN][filt], 50, axis=0)
                        magmax = np.percentile(mag_all[model_KN][filt], 90, axis=0) 
                        magmin = np.percentile(mag_all[model_KN][filt], 10, axis=0) 
                                
                        app_magmed_mbta = np.percentile(app_mag_mbta_all[model_KN][filt], 50, axis=0)
                        app_magmax_mbta = np.percentile(app_mag_mbta_all[model_KN][filt], 90, axis=0) 
                        app_magmin_mbta = np.percentile(app_mag_mbta_all[model_KN][filt], 10, axis=0)

                        with open(os.path.join(output_path, filt + "_filter.txt"), 'w') as f:
                                f.write("#time mag_percentile(10%) mag_percentile(50%) mag_percentile(90%) appmag_percentile(10%) appmag_percentile(50%) appmag_percentile(90%)" + "\n")
                                for i in range(len(magmed)):
                                        f.write(str(tt[i]) + " " + str(magmin[i]) + " " + str(magmed[i]) + " " + str(magmax[i]) + " " + str(app_magmin_mbta[i]) + " " + str(app_magmed_mbta[i]) + " " + str(app_magmax_mbta[i]) + "\n")
                                f.close
                          

