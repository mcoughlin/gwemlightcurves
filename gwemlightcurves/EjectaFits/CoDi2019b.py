
# Modified from: http://www2.yukawa.kyoto-u.ac.jp/~kyohei.kawaguchi/kn_calc/main.html
# Reference: Kawaguchi et al. https://arxiv.org/abs/1601.07711

import numpy as np
import scipy

def calc_meje(q,chi_eff,c,mb,mns):

    a1= 0.007116
    a2=0.001436
    a3=-0.0276
    n1=0.8636
    n2=1.6840

    tmp1=(1-2*c)*(q**n1)*a1;
    tmp2=-a2*(q**n2)*(r_isco(chi_eff))
    tmp3=a3

    meje_fit=mb*np.maximum(tmp1+tmp2+tmp3,0)

    return meje_fit

def calc_vave(q):
    return 1.5333330951369120e-2*q+0.19066667068621043

def r_isco(chi):
  z1=1+((1-chi*chi)**(1/3.0))*(((1+chi)**(1/3.0))+(1-chi)**(1/3.0))
  z2=(3*chi*chi+z1*z1)**(1/2.0)
  return 3+z2-np.sign(chi)*((3-z1)*(3+z1+2*z2))**(1/2.0)
