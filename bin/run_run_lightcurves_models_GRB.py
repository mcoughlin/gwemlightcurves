
import os, sys, glob
import optparse

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-g","--grb",default="GRB130603B")

    parser.add_option("-e","--errorbudget",default=0.25,type=float)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

errorbudget = opts.errorbudget

filename = "../lightcurves/GRB.dat"
lines = [line.rstrip('\n') for line in open(filename)]

for line in lines:
    lineSplit = line.split(" ")
    if lineSplit[0] == opts.grb:
        grb = lineSplit[0]
        filts = lineSplit[1]
        mjd = float(lineSplit[2])
        dist = float(lineSplit[3])
        break

if grb in ["GRB061201","GRB050509B","GRB080905A","GRB050709","GRB051210","GRB060502B","GRB130603B"]:
    doExtrapolate = "--doWaveformExtrapolate"
else:
    doExtrapolate = ""

model = "Bu2019inc"
system_command = "python run_lightcurves_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --distance %.5f --T0 %.5f --filters %s --errorbudget %.2f --doFixZPT0 --doEjecta %s"%(model,grb,dist,mjd,filts,errorbudget,doExtrapolate)
print(system_command)
#os.system(system_command)
print(stop)

model = "TrPi2018"
system_command = "python run_lightcurves_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --distance %.5f --T0 %.5f --filters %s --errorbudget %.2f --doFixZPT0 --doEjecta %s"%(model,grb,dist,mjd,filts,errorbudget,doExtrapolate)
print(system_command)
#os.system(system_command)

model = "Ka2017_TrPi2018"
system_command = "python run_lightcurves_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --distance %.5f --T0 %.5f --filters %s --errorbudget %.2f --doFixZPT0 --doEjecta %s"%(model,grb,dist,mjd,filts,errorbudget,doExtrapolate)
print(system_command)
print(stop)
os.system(system_command)
print(stop)

lambdamin, lambdamax = 0, 500
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 0, 1000
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 0, 2000
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 200, 500
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 200, 1000
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 200, 2000
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 200, 2000
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

lambdamin, lambdamax = 0, 5000
system_command = "python run_fitting_models.py --doEvent --model %s --name %s --tmin 0.0 --tmax 10.0 --filters %s --errorbudget %.2f --doFixZPT0 --lambdamin %.0f --lambdamax %.0f --doJoint --doLightcurves"%(model,grb,filts,errorbudget,lambdamin, lambdamax)
#os.system(system_command)

print system_command


