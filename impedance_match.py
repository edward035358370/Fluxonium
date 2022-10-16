from fluxonium_qutip import Fluxonium
import numpy as np
import pylab as plt
import math
"""
the program is only for [0,2] and [1,2] transition
"""
def coth(x):
    points = len(x)
    coth_array = np.linspace(0,0,points)
    
    for i in range(points):
        coth_array[i] = math.cosh(x[i]) / math.sinh(x[i])
    return coth_array

def Bode(f,fc):
    return (f/fc)/(1+(f/fc)**2)**0.5

def plotting(x,gamma_array,transition,traces = 2,plus_1 = True):
    move = 1
    if plus_1 == False:
        move = 0
    for i in range(traces):
        plt.plot(x,gamma_array[i+move],label = transition[i])
    plt.legend()

def relax_rate(E,x_axis,fc,transition =  [[0,1],[0,2],[1,2]]):
    #fluxonium parameters
    Ej = E[0]
    Ec = E[1]
    El = E[2]
    start = x_axis[0]
    stop = x_axis[1]
    points = x_axis[2]
    levels = len(transition)
    
    #relaxation rate parameters
    fc = 2*np.pi*fc*10**9 #Hz 

    porpotional = 0.0012067/9.08 #count by paper gamma03
    fluxonium = Fluxonium(Ej,Ec,El,101,start=start,stop=stop,points=points)
    
    x = np.linspace(start,stop,points)
    
    
    f_in_GHz = fluxonium.eigen_spectrum(levels,[True,False]) #GHz
    f = 2*np.pi*f_in_GHz*10**9 #Hz
    
    traces = len(transition)
    matrix_ele = fluxonium.matrix_ele('n',transition)
    
    gamma_array = np.zeros((traces,points))
    
    for i in range(traces):
        T = abs(Bode(f[i],fc))**2
        gamma_array[i] = f[i]*T*(abs(matrix_ele[i])**2)*porpotional
    
    return gamma_array,x,f_in_GHz

def deph_rate(E,x_axis,transition=[[0,1],[0,2],[1,2]]):
        #fluxonium parameters
        Ej = E[0]
        Ec = E[1]
        El = E[2]
        start = x_axis[0]
        stop = x_axis[1]
        points = x_axis[2]
        levels = 3
        
        x = np.linspace(start,stop,points)
        
        
        
        fluxonium = Fluxonium(Ej,Ec,El,101,start=start,stop=stop,points=points)

        f_in_GHz = fluxonium.eigen_spectrum(levels,[True,False]) #GHz
        
        flux_noise_amp = (4.3e-6)**2 #set by the transmon experiment
        dephase_rate = np.zeros((len(transition),points-1)) # points -1 is for differential
        
        for i in range(len(transition)):
            Ene = f_in_GHz[i]
            phi_ext = x
            E_diff = np.diff(Ene)/np.diff(phi_ext)
            E_diff_const = 2*np.pi*1e9*(flux_noise_amp*np.log(2))**0.5
            dephase_rate[i] = np.abs(E_diff)*E_diff_const
            
        x = np.linspace(start,stop,points-1) #remove one point for plot
        return dephase_rate,x
    
def pick_imp_match_para(path,E,x_axis,fc):
    
    gamma_array,x ,f_in_GHz = relax_rate(E,x_axis,fc)
    gamma_diff = abs(gamma_array[1] - gamma_array[2])
    gamma_deph,_ = deph_rate(E,x_axis)
    
    freq_GHz_diff = abs(f_in_GHz[1] - f_in_GHz[2])
    f = open(path +"/[Ej,Ec,El]= %s.txt"%(E),"a")
    for i in range(len(gamma_diff)):
        if gamma_diff[i] <= 1e5 and freq_GHz_diff[i] <= 1:
            
            f.write("==============\n")
            f.write("flux: %s\n"%(x[i]))
            f.write("f01=%s,f02=%s (GHz),f12=%s (GHz)\n"%(f_in_GHz[0][i],f_in_GHz[1][i],f_in_GHz[2][i]))
            f.write("gamma02= %s (MHz),gamma12= %s (MHz)\n"%(gamma_array[1][i]/1e6,gamma_array[2][i]/1e6))
            f.write("gamma01= %s (MHz),dephasing 01= %s (MHz)\n"%(gamma_array[0][i]/1e6,gamma_deph[0][i-1]/1e6))
            f.write("==============\n")
            f.write("  \n")
    f.close()

if __name__ == "__main__":
    x_axis = [0.5,0.55,101]
    path = "./find_impedance_match/dephasing/cutoff 6GHz/change Ej@Ec=1.2,El=0.62"
    E = [6.25,0.75,0.91]
    
    """
    E is base on paper electron shelving
    
    for i in range(72,73):
        E = [i/10.,0.62,1.2]
        pick_imp_match_para(path,E,x_axis,6)
        
    """
    transition = []
    count = 0
    for i in range(3):
        for ii in range(i+1):
            if i!= ii:
                count += 1
                transition.append([ii,i])
    gamma_array,x ,f_in_GHz = relax_rate(E,x_axis,3.5,transition)
    plotting(x,gamma_array,transition,traces = count,plus_1 = False)
    
    """
    gamma_deph,x = deph_rate(E,x_axis)
    plotting(x,gamma_deph,[[0,1],[0,2],[1,2]],traces = 3,plus_1 = False)
    """