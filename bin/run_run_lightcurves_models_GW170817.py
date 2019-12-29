
import os, sys, glob
import optparse

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-m","--model",default="Ka2017")
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doBNSFit",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("--doJoint",  action="store_true", default=False)
    parser.add_option("--doJointLambda",  action="store_true", default=False)
    parser.add_option("--doJointDisk",  action="store_true", default=False)
    parser.add_option("--doJointSpin",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doLightcurves",  action="store_true", default=False)
    parser.add_option("--doLuminosity",  action="store_true", default=False)
    parser.add_option("-f","--filters",default="u,g,r,i,z,y,J,H,K")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.0,type=float)
    parser.add_option("--colormodel",default="a2.0")
    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017inc","Ka2017x2","Ka2017x2inc","Ka2017x3","RoFe2017","BoxFit","TrPi2018","Ka2017_TrPi2018_A","Bu2019","Bu2019inc"]:
    print("Model must be either: DiUj2017,KaKy2016,Me2017,Me2017x2,SmCh2017,WoKo2017,BaKa2016,Ka2017,Ka2017inc,Ka2017x2,Ka2017x2inc,Ka2017x3,RoFe2017,BoxFit,TrPi2018,Ka2017_TrPi2018_A,Bu2019,Bu2019inc")
    exit(0)

if opts.doEOSFit:
    eosfitFlag = "--doEOSFit"
elif opts.doBNSFit:
    eosfitFlag = "--doBNSFit"
else:
    eosfitFlag = ""

if opts.doFixZPT0:
    fixzpt0Flag = "--doFixZPT0"
else:
    fixzpt0Flag = ""

if opts.doMasses:
    typeFlag = "--doMasses"
elif opts.doEjecta:
    typeFlag = "--doEjecta"
elif opts.doJoint:
    typeFlag = "--doJoint"
elif opts.doJointLambda:
    typeFlag = "--doJointLambda"
elif opts.doJointDisk:
    typeFlag = "--doJointDisk"
elif opts.doJointSpin:
    typeFlag = "--doJointSpin"
else:
    print("Must specify --doMasses, --doEjecta, --doJoint, --doJointDisk, --doJointLambda, or --doJointSpin")
    exit(0)

if not (opts.doLuminosity or opts.doLightcurves):
    print("Must specify --doLuminosity or --doLightcurves")
    exit(0)

if opts.doLightcurves:
    system_command = "python run_lightcurves_models.py --doEvent --model %s --name GW170817 --tmin %.2f --tmax %.2f --filters %s --errorbudget %.2f --colormodel %s %s %s %s"%(opts.model,opts.tmin,opts.tmax,opts.filters,opts.errorbudget,opts.colormodel,eosfitFlag,fixzpt0Flag,typeFlag)
    #os.system(system_command)
    print(system_command)
    print(stop)

    lambdamin, lambdamax = 0, 1140
    system_command = "python run_fitting_models.py --doLightcurves --doEvent --model %s --name GW170817 --tmin %.2f --tmax %.2f --filters %s --errorbudget %.2f %s %s %s --lambdamin %.0f --lambdamax %.0f"%(opts.model,opts.tmin,opts.tmax,opts.filters,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag,lambdamin, lambdamax)
    print(system_command)
    print(stop)
    os.system(system_command)

if opts.doLuminosity:
    system_command = "python run_luminosity_models.py --doEvent --model %s --name GW170817_Lbol --tmin %.2f --tmax %.2f --errorbudget %.2f %s %s %s "%(opts.model,opts.tmin,opts.tmax,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag)
    os.system(system_command)

