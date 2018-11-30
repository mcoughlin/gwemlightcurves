#Algorithm from https://uglyduckling.nl/library_files/PRE-PRINT-UD-2014-03.pdf
import numpy as np

def interpolate(x_values,y_values):
    """
    https://uglyduckling.nl/library_files/PRE-PRINT-UD-2014-03.pdf
    """
    #make sure the x and y arrays are the same size
    if len(x_values)!=len(y_values):
        print("X and Y arrays must be the same length!")
        return

    n=len(x_values)
    h=np.zeros(n)
    m=np.zeros(n)
    beta=np.zeros(n)
    #l=np.zeros(n)
    #mu=np.zeros(n)
    #z=np.zeros(n)
    consts=np.zeros((n,3))

    for i in np.arange(0,n-1,1):
        h[i]=x_values[i+1]-x_values[i]
        if h[i]==0:
            h[i]=10**-4
        m[i]=(y_values[i+1]-y_values[i])/h[i]

    consts[0,0]=0.
    consts[n-1,0]=0.

    for i in np.arange(1,n-1,1):
       if (m[i-1]*m[i]<=0.):
           consts[i,0]=0.
       elif (m[i-1]*m[i]>0.):
          beta[i]=(3.*m[i-1]*m[i])/(max(m[i-1],m[i])+2.*min(m[i-1],m[i]))
          if np.sign(m[i-1])==1 and np.sign(m[i])==1:
             consts[i,0]= min(max(0.,beta[i]),3.*min(m[i-1],m[i]))
          elif np.sign(m[i-1])==-1 and np.sign(m[i])==-1:
             consts[i,0]=max(min(0.,beta[i]),3.*max(m[i-1],m[i]))
          else:
             print("We have a problem")

    for i in np.arange(0,n-1,1):
       consts[i,1]=(3*m[i]-consts[i+1,0]-2.*consts[i,0])/h[i]
       consts[i,2]=(consts[i+1,0]+consts[i,0]-2.*m[i])/h[i]**2

    return consts


def lin_extrapolate(x_values,y_values):
    """
    https://uglyduckling.nl/library_files/PRE-PRINT-UD-2014-03.pdf
    """
    n=len(x_values)  

    m1=(y_values[1]-y_values[0])/(x_values[1]-x_values[0])
    b1=y_values[0]-m1*x_values[0]

    m2=(y_values[n-1]-y_values[n-2])/(x_values[n-1]-x_values[n-2])
    b2=y_values[n-1]-m2*x_values[n-1]

    line_consts=np.array([[m1,b1],[m2,b2]])

    return line_consts
