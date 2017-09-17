
# Modified from: http://www2.yukawa.kyoto-u.ac.jp/~kyohei.kawaguchi/kn_calc_bns1/main.html
# Reference: Dietrich et al. http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1612.03665 

import numpy as np
import scipy

def calc_meje(m1,mb1,c1,m2,mb2,c2):

    a= -1.35695
    b=  6.11252 
    c=-49.43355      
    d=  16.1144    
    n=  -2.5484

    tmp1=((mb1*((m2/m1)**(1.0/3.0))*(1.0-2.0*c1)/c1)+(mb2*((m1/m2)**(1.0/3.0))*(1.0-2.0*c2)/c2)) * a
    tmp2=(mb1*((m2/m1)**n)+mb2*((m1/m2)**n)) * b
    tmp3=(mb1*(1.0-m1/mb1)+mb2*(1.0-m2/mb2)) * c

    meje_fit=np.maximum(tmp1+tmp2+tmp3+d,0)/1000.0
  
    return meje_fit

def calc_vrho(m1,c1,m2,c2):
    a=-0.219479
    b=0.444836 
    c=-2.67385      
  
    return ((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))*a+b

def calc_vz(m1,c1,m2,c2):
    a=-0.315585
    b=0.63808 
    c=-1.00757      

    return ((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))*a +b

def calc_vej(m1,c1,m2,c2):
    return np.sqrt(calc_vrho(m1,c1,m2,c2)**2.0+calc_vz(m1,c1,m2,c2)**2.0)

def calc_qej(m1,c1,m2,c2):
    vrho=calc_vrho(m1,c1,m2,c2)
    vz=calc_vz(m1,c1,m2,c2)
    vrho2=vrho*vrho
    vz2=vz*vz

    tmp1=3.*vz+np.sqrt(9*vz2+4*vrho2)
    qej=((2.0**(4.0/3.0))*vrho2+(2.*vrho2*tmp1)**(2.0/3.0))/((vrho**5.0)*tmp1)**(1.0/3.0)

    return qej

def calc_phej(m1,c1,m2,c2):
  return 4.0*calc_qej(m1,c1,m2,c2)*np.pi/2.0

