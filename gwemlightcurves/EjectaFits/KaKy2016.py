
# Modified from: http://www2.yukawa.kyoto-u.ac.jp/~kyohei.kawaguchi/kn_calc/main.html
# Reference: Kawaguchi et al. https://arxiv.org/abs/1601.07711

import numpy as np
import scipy

def calc_meje(q,chi_eff,c,mb,mns):

    a1=-2.269e-3
    a2=4.464e-2 
    a3=2.431    
    a4=-0.4159   
    n1=1.352
    n2=0.2497

    tmp1=r_isco(chi_eff)*(q**n1)*a1;
    tmp2=(q**n2)*(1-2*c)*a2/c
    tmp3=(1-mns/mb)*a3+a4

    meje_fit=mb*np.maximum(tmp1+tmp2+tmp3,0);

    return meje_fit

def calc_vave(q):
    return 1.5333330951369120e-2*q+0.19066667068621043

def r_isco(chi):
  z1=1+((1-chi*chi)**(1/3.0))*(((1+chi)**(1/3.0))+(1-chi)**(1/3.0))
  z2=(3*chi*chi+z1*z1)**(1/2.0)
  return 3+z2-np.sign(chi)*((3-z1)*(3+z1+2*z2))**(1/2.0)
