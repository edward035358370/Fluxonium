from impedance_match import relax_rate
import numpy as np
import pylab as plt

def r(f_sweep,f_cent,gamma,p,rabi):
    temp = 0.5*gamma**2 +rabi**2 + 2*(f_sweep-f_cent)**2
    temp = ((0.5*gamma**2) - 1j*(f_sweep-f_cent)*gamma)/temp
    temp = 2*p*temp
    return 1 - temp

x_axis = [0.4877,0.4877,1]

E = [5.6,0.8,0.7]

gamma_array,x ,f_in_GHz = relax_rate(E,x_axis,3.5,[[0,1],[0,2],[1,2]])

pos = 2

points = 1001
flux = 0
width = 0.02 #MHz
title = "f12"

non_rate = 0.0039 * 1e4
gamma_array = non_rate + gamma_array
f_sweep = np.linspace(f_in_GHz[pos][flux]-width,f_in_GHz[pos][flux]+width,points)*1e9

reflective = np.zeros((3,points),dtype ='complex')
reflective_circle = []


for ii in range(1,4):
    
    for i in range(points):
        reflective[ii-1][i] = r(f_sweep[i],f_in_GHz[pos][flux]*1e9,gamma_array[pos][flux],1,ii*1e6)
    
    reflective_circle.append([np.real(reflective[ii-1]),np.imag(reflective[ii-1])])
    plt.figure(0)
    plt.title(title)
    plt.plot(f_sweep,abs(reflective[ii-1]),label = "rabi="+str(ii)+"MHz")
    plt.legend()
    
    plt.figure(1,figsize=(4,4))
    plt.title(title)
    plt.plot(reflective_circle[-1][0],reflective_circle[-1][1],label = "rabi="+str(ii)+"MHz")
    plt.legend()

    



