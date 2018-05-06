
import os, sys, glob
import optparse

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-m","--model",default="TrPi2018")
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doBNSFit",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("--doJoint",  action="store_true", default=False)
    parser.add_option("--doJointLambda",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doLightcurves",  action="store_true", default=False)
    parser.add_option("--doLuminosity",  action="store_true", default=False)
    parser.add_option("-f","--filters",default="g,V,F606W,r,i,z,J,K,F160W")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017x2","Ka2017x3","RoFe2017","BoxFit","TrPi2018"]:
    print "Model must be either: DiUj2017,KaKy2016,Me2017,Me2017x2,SmCh2017,WoKo2017,BaKa2016,Ka2017,Ka2017x2,Ka2017x3,RoFe2017,BoxFit,TrPi2018"
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
else:
    print "Must specify --doMasses, --doEjecta, --doJoint or --doJointLambda"
    exit(0)

if not (opts.doLuminosity or opts.doLightcurves):
    print "Must specify --doLuminosity or --doLightcurves"
    exit(0)

if opts.doLightcurves:
    system_command = "python run_lightcurves_models.py --doEvent --model %s --name GRB130603B --tmin 0.0 --tmax 10.0 --distance 1915.1 --T0 56446.00840278 --filters %s --errorbudget %.2f %s %s %s"%(opts.model,opts.filters,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag)
    #os.system(system_command)
    print(system_command)
    print(stop)

    lambdamin, lambdamax = 0, 640
    system_command = "python run_fitting_models.py --doLightcurves --doEvent --model %s --name GW170817 --tmin 0.0 --tmax 14.0 --filters %s --errorbudget %.2f %s %s %s --lambdamin %.0f --lambdamax %.0f"%(opts.model,opts.filters,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag,lambdamin, lambdamax)
    #print system_command 
    #os.system(system_command)

    lambdamin, lambdamax = 0, 1500
    system_command = "python run_fitting_models.py --doLightcurves --doEvent --model %s --name GW170817 --tmin 0.0 --tmax 14.0 --filters %s --errorbudget %.2f %s %s %s --lambdamin %.0f --lambdamax %.0f"%(opts.model,opts.filters,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag,lambdamin, lambdamax)
    #os.system(system_command)

if opts.doLuminosity:
    system_command = "python run_luminosity_models.py --doEvent --model %s --name G298048_XSH_PESSTO_redo --tmin 0.0 --tmax 6.0 --errorbudget %.2f %s %s %s "%(opts.model,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag)
    #os.system(system_command)

    system_command = "python run_luminosity_models.py --doEvent --model %s --name G298048_XSH_PESSTO_redo --tmin 0.0 --tmax 14.0 --errorbudget %.2f %s %s %s "%(opts.model,opts.errorbudget,eosfitFlag,fixzpt0Flag,typeFlag)
    #os.system(system_command)

    print system_command

