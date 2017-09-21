
import os, sys, glob
import optparse

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017"]:
    print "Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017"
    exit(0)

if opts.doEOSFit:
    eosfitFlag = "--doEOSFit"
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
else:
    print "Must specify --doMasses or --doEjecta"
    exit(0)

system_command = "python run_lightcurves_models.py --doEvent --model %s --name G298048_PS1_GROND_SOFI --tmin 0.0 --tmax 14.0 --filters u,g,r,i,z,y,J,H,K %s %s %s "%(opts.model,eosfitFlag,fixzpt0Flag,typeFlag)
os.system(system_command)

system_command = "python run_lightcurves_models.py --doEvent --model %s --name G298048_PS1_GROND_SOFI --tmin 2.0 --tmax 14.0 --filters u,g,r,i,z,y,J,H,K %s %s %s"%(opts.model,eosfitFlag,fixzpt0Flag,typeFlag)
os.system(system_command)

system_command = "python run_lightcurves_models.py --doEvent --model %s --name G298048_PS1_GROND_SOFI --tmin 7.0 --tmax 14.0 --filters u,g,r,i,z,y,J,H,K %s %s %s"%(opts.model,eosfitFlag,fixzpt0Flag,typeFlag)
os.system(system_command)


