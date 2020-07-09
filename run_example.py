import numpy as np
from gwemlightcurves.em_counterpart import EM_Counterpart


output_path = "/home/cosmin.stachie/public_html/popular" #the place where you want to save the data
model_KN = "Bu2019inc" #model to use to generate the KN lightcurve; for the moment the model should be in {Ka2017, Bu2019inc}
Xlan_fixed = -4 # the logarithm of the lanthanide fraction; values in [10^-9, 10^-1]; it is used only for the Kasen model
phi_fixed = 45 # the half opening angle of the lanthanide rich ejecta component: it is used only for the Bulla model

weight_array = [21.7,  9.3,  2.6,  3.4,  5.3,  4.8,  6.3, 11. , 12. ,  9. ,  4. , 1.5,  1.5,  1.2,  1.7,  1.9,  0.8,  0.4,  0.6,  0.4,  0.1,  0.1, 0.1,  0.2,  0.1]
m1_array = [2.269, 1.838, 1.885, 1.82 , 1.791, 1.826, 2.269, 1.843, 1.945, 1.797, 2.095, 1.901, 2.225, 1.947, 1.938, 1.9, 3.077, 2.59 , 1.79 , 2.263, 1.973, 2.238, 2.516, 2.107, 1.731]
m2_array = [1.305, 1.587, 1.552, 1.604, 1.627, 1.599, 1.305, 1.587, 1.501, 1.626, 1.405, 1.534, 1.327, 1.501, 1.507, 1.537, 1.006, 1.161, 1.634, 1.312, 1.484, 1.323, 1.189, 1.398, 1.686]
s1_array = [0.08, -0.04,  0.03,  0.02, -0.01,  0.01,  0.08,  0.04, -0.04, 0.01,  0.06, -0.05,  0.02, -0.04, -0.02,  0.01,  0.18,  0.09, 0.04,  0.17,  0.02,  0.12,  0.04,  0.1 ,  0.05]
s2_array = [-0.01,  0.02,  0.04,  0.01, -0.04, -0.02, -0.01,  0.05, -0.02, 0.04,  0.04, -0.04, -0.02, -0.02, -0.03, -0.01, -0.02,  0.02, 0.04, -0.03,  0.05, -0.03,  0.  ,  0.03,  0.04]
dist_array = [213., 231., 232., 232., 233., 235., 225., 235., 237., 242., 226., 256., 229., 257., 242., 245., 245., 235., 258., 245., 257., 236., 254., 249., 267.]

input_samples = np.array([weight_array, m1_array, m2_array, s1_array, s2_array, dist_array]).T

em_object = EM_Counterpart(input_samples, Xlan_fixed, phi_fixed)
mej = em_object.calc_ejecta(model_KN)
(mag_all, app_mag_mbta_all, lbol_all) = em_object.calc_lightcurve(model_KN)
em_object.write_output(output_path, model_KN,  mej, mag_all, app_mag_mbta_all, lbol_all)
