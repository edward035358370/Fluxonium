# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:55:59 2021

@author: edwar
"""

import numpy as np
from scipy.linalg import expm
from inspect import getfullargspec as showarg
import qutip as qt

class Fluxonium():
    def __init__(self,Ej,Ec,El,cutoff,start = -1,
                                      stop = 1,
                                      points = 201):
        """

        Parameters
        ----------
        Ej : josephson junction energy (GHz)
        Ec : capacitance energy (GHz)
        El : inductance energy (GHz)
        cutoff :decide how huge the hamiltonian be made

        Returns
        -------
        None.

        """
        self.Ej = Ej
        self.Ec = Ec
        self.El = El
        self.cutoff = cutoff
        self.start = start
        self.stop = stop
        self.points = points
        self.x = np.linspace(self.start,self.stop,self.points)
        
    def get_x(self):
        """
        Formatted external flux phi
    
        Parameters
        ----------
        start : float, optional
            start points of flux. The default is -1.
        stop : float, optional
            stop points of flux. The default is 1.
        points : reslution for external flux, default is 201
    
        Returns
        -------
        np.array
    
        """
        self.x = np.linspace(self.start,self.stop,self.points)
        return self.x
    
    def hamiltonian_exp(self,phi_ext):
        """
        Return the Hamiltonian matrix with energy in GHz calculated with
        the exponential method.
        
        H = 4Ec|n_op><n_op| + 0.5El|phi_op><phi_op| - Ej*cos(phi_op+phi)
        
        Parameters
        ----------
        phi : float
            Reduced Φ_ext/Φ_0.
        Returns
        ----------
        Hamiltonian : np.matrix
            Hamiltonian matrix.
        """
        ope = 1.0j * (self.op_phi() + phi_ext*2*np.pi)
        
        return 4.0 * self.Ec * self.op_n() ** 2.0 + 0.5 * self.El * self.op_phi() ** 2.0\
            - 0.5 * self.Ej * (ope.expm() + (-ope).expm())
        
    def op_phi(self):
        """
        create the operator for the change of phase (normalized flux)

        Returns
        -------
        np.array 2D array

        """
        n = self.cutoff+1
        a = qt.tensor(qt.destroy(n)) #80代表計算精確度
        phi = (a + a.dag()) * (8.0 * self.Ec / self.El) ** (0.25) / np.sqrt(2.0)
        return phi
    
    def op_n(self):
        """
        create the operator for the change of charge (normalized charge)

        Returns
        -------
        np.array 2D array

        """
        n = self.cutoff + 1
        a = qt.tensor(qt.destroy(n))
        na = 1.0j * (a.dag() - a) * (El / (8 * Ec)) ** (0.25) / np.sqrt(2.0)
        return na
    
    def op_sin_phi(self,phi_ext):
        ope = 1.0j * (self.op_phi() + np.eye(self.cutoff+1)*phi_ext*2*np.pi)
        sine_ope = ((ope/2.0).expm() - (-ope/2.0).expm())/(2.0j)
        
        return sine_ope
    def eigen(self,H):
        """
        extracte the eigen value and eigen state

        Parameters
        ----------
        H : hamiltonian, is 2D array

        Returns
        -------
        w : eigen value np.array
        v : eigen vector  2D array

        """
        eigen_energies, eigen_states = H.eigenstates()
        return eigen_energies, eigen_states
    
    def harmonic_oscilator_energy(self,n):
        """
               ________
        energy/8*Ec*El  * (n+0.5)

        Parameters
        ----------
        n : position of the hamiltonian

        Returns
        -------
        float

        """
        return np.sqrt(8.*Ec*El)*(n + 0.5)
    
    def eigen_spectrum(self,levels,transition = [True,True]):
        """
        get the different energy state with different external flux,
        and count the transition energy

        Parameters
        ----------
        levels : the result we want to show (must lower than self.cutoff)
        transition : [boolen,boolen], optional
            to decide to show energy state or transition energy, and decide
            to show every transition or only 0--->i transition. 
            The default is [True,True].

        Returns
        -------
        2D array

        """
        energy = []
        for i in range(levels):
            energy.append([])
        energy = np.array(energy)
        
        for i in self.x:
            H = self.hamiltonian_exp(i)
            temp,_ = self.eigen(H)
            energy = np.concatenate((energy,np.reshape(temp[0:levels],(levels,1))),axis=1)
        if transition[0]:
            spectrum = np.linspace(0,0,len(self.x))
            for high in range(len(energy)):
                for low in range(len(energy)):
                    if low == 0 and transition[1]:
                        spectrum = np.vstack((spectrum,energy[high] - energy[low]))
                    elif high > low and transition[1] == False:
                        spectrum = np.vstack((spectrum,energy[high] - energy[low]))
            return spectrum[1:,:]
        return energy
    
    def matrix_ele(self,operator: str,bra_ket: list):
        """
        for matrix element
        <eigen state|n_op|eigen state> or <eigen state|phi_op|eigen state>
        Parameters
        ----------
        operator : str
            can decide to use phi operator or charge operator.
            input phi or n for working.
        bra_ket : list
            the transition energy we want to see [i,j]
            i ---> j

        Returns
        -------
        np.array 2D array

        """
        k = 0
        if operator == "phi":
            operate = self.op_phi()
        elif operator == "n":
            operate = self.op_n()
        elif operator == "sin(phi/2)":
            k = 1
            
        
        matrix_element = np.zeros((len(bra_ket),self.points))
        
        for colum, i in  enumerate(self.x):
            
            H = self.hamiltonian_exp(i)
            _,temp = self.eigen(H)
            
            if k == 1:
                operate = self.op_sin_phi(i)
            for roll, transition in enumerate(bra_ket):
                matrix_element[roll][colum] = abs(operate.matrix_element(temp[transition[0]],temp[transition[1]]))
                
        return np.array(matrix_element)
    
    def T1_die_loss(self,times):
        """
        use fluctuation noise source for simulating
        the T1 which caused by dieletric loss

        Parameters
        ----------
        El_array : list
            the parameter you want to see

        Returns
        -------
        res : np.array
            DESCRIPTION.

        """
        
        kB = 1.38064852e-23
        T = 0.01 #10 mk
        h_bar = 6.62e-34/(2*np.pi)    #Placnk's reduce constant
        
        res = np.zeros((len(times),self.points))
        for index,i in enumerate(times):

            w = abs(self.eigen_spectrum(3,[True,True])[1]*1e9*2*np.pi)
            y_phi_ele = self.matrix_ele('phi', [[0,1]])[0]
            
            Ec_hbar = h_bar*self.Ec*2*np.pi*1e9
            Q_cap = 1e6*(2*np.pi*6e9/w)**(-0.1)
            Q_cap = Q_cap*i
            gamma_cap = h_bar/(4*Ec_hbar*Q_cap)
            gamma_cap = gamma_cap*(w**2)*((y_phi_ele)**2)
            for idx in range(len(gamma_cap)):
                gamma_cap[idx] = gamma_cap[idx]*abs(coth(h_bar*w[idx]/(2*kB*T))+1)
            res[index] = np.log10(1e6/(2*np.pi*gamma_cap))
        return res
    
    def T_qp_sing_junc(self,xqp_list,transition = [0,1]):
        
        relax_time_qp = np.zeros((len(xqp_list),self.points))
        h = 6.626e-34
        e = 1.6e-19
        gap = 93e-6*e
        for idx,xqp in enumerate(xqp_list):
            temp = (self.matrix_ele('sin(phi/2)',[transition])[0])**2
            E01 = self.eigen_spectrum(9,transition = [True,True])[transition[1]]*1e9
            constant = 8*self.Ej*h*1e9*xqp/(h*0.5)
            constant = constant*(2*gap/(h*E01))**0.5
            relaxa_qp = temp*constant
            relax_time_qp[idx] = 1e6/relaxa_qp
        return np.log10(relax_time_qp)
    def T_qp_array(self,xqp_list,transition = [0,1]):
        
        relax_time_qp = np.zeros((len(xqp_list),self.points))
        h = 6.626e-34
        e = 1.6e-19
        gap = 93e-6*e
        for idx,xqp in enumerate(xqp_list):
            temp = (self.matrix_ele('phi',[transition])[0]/2)**2
            E01 = self.eigen_spectrum(9,transition = [True,True])[transition[1]]*1e9
            constant = 8*self.Ej*h*1e9*xqp/(h*0.5)
            constant = constant*(2*gap/(h*E01))**0.5
            relaxa_qp = temp*constant
            relax_time_qp[idx] = 1e6/relaxa_qp
        return np.log10(relax_time_qp)
    def T_dephasing(self,transition = [1]):
        """
        

        Parameters
        ----------
        transition : list
            decide the transition you want.
            ex.
                0--->1: 1
                0--->2: 2.....
            transition = [1,2]
        Returns
        -------
        None.

        """
        dephase_time = np.zeros((len(transition),self.points-1)) # points -1 is for differential
        flux_noise_amp = (4.3e-6)**2 #set by the transmon experiment
        energy = self.eigen_spectrum(9,[True,True])
        
        for i in transition:
            E = energy[i]
            phi_ext = self.x
            E_diff = np.diff(E)/np.diff(phi_ext)
            E_diff_const = 2*np.pi*1e9*(flux_noise_amp*np.log(2))**0.5
            dephase_rate = np.abs(E_diff)*E_diff_const
            
            dephase_time[i-1] = 1/dephase_rate
        self.x = np.linspace(self.start,self.stop,self.points-1) #remove one point for plot
        return dephase_time
    
    def function_mappings(self):
        """
        str to func mappings, the IO for Plot1DSpectrum
        """
        return {
            'eigen_spectrum': self.eigen_spectrum,
            'matrix_ele': self.matrix_ele,
            'T1_die_loss': self.T1_die_loss,
            'T_dephasing': self.T_dephasing,
            'T_qp_sing_junc': self.T_qp_sing_junc,
            'T_qp_array': self.T_qp_array
            }
            
    def setFunc(self,
            func: str, funcArg: list
            ):
        """
        Shape function wrapper.
    
        Parameters
        ----------
        func : str/function handle
            Shape function to be used, can be specified by string or function
            handle.
        funcArg : list/dict
            List/dict of arguments of the function. In dict mode, the user can
            declare key-value pair with each key corresponding to the argument of
            the function. The key must match the argument name.
        start : start of the external flux
        stop : stop of the external flux
        points : points of the external flux
    
        """

        if isinstance(func, str):
            func = self.function_mappings()[func]
        argNames = showarg(func).args[1:] #get function args besides x
        if isinstance(funcArg, dict):
            funcArg = [funcArg[arg] for arg in argNames]
    
        generator = {
            'function': func.__name__, #__name__is function's name
            'X': {'start': self.start, 'stop': self.stop, 'points': self.points},
            'Y': dict(zip(argNames, funcArg))
            }
        return generator

    def parse(self,generator: dict):
        """
        Parse and compile generator data into wave x-y data
    
        Parameters
        ----------
        generator : dict
            Discription dict for wave generation.
    
        Returns
        -------
        x : np.array
            x (phi) data.
        y : np.array
            y (amplitude) data.
    
        """

        # y
        func = generator['function']
        if isinstance(func, str):
            func = self.function_mappings()[func]
        argNames = showarg(func).args[1:]
        funcArg = [generator['Y'][arg] for arg in argNames]
        y = func(*funcArg)
        
        
        return self.x, y
    
    def plot1Dspectrum(self,func_name, argdict):
        """
        plot the data can be count, and the way of parameters are counted
        is given by parse and setFunc

        Parameters
        ----------
        func_name : str
            get the func, same as function_mappings
        argdict : dict
            parameters in the function, parameters name must same as func above
        start : float, optional
            start points of flux. The default is -1.
        stop : float, optional
            stop points of flux. The default is 1.
        points : reslution for external flux, default is 201

        Returns
        -------
        x : np.array
            phi
        y : 2D array
            for different parameters in the argdict

        """
        from pylab import figure,plot, show
        
        generator = self.setFunc(func_name, argdict)
        x,y = self.parse(generator)
        
        figure()
        for yi in y:
            plot(x,yi)
        show()
        return x,y
    
if __name__ == "__main__":
    """
    1, use Qutip as the function for solving the eigen
    
    2, dispersive shift......
    """
    from mpmath import coth
    
    start = 0
    stop = 1
    points = 201
    levels = 3
    
    loop_size = 3
    res = np.zeros((loop_size,points))

    Ej = 3
    Ec = 1
    El = 1
    
    fluxonium = Fluxonium(Ej,Ec,El,101,start=start,stop=stop,points=points)
    
    #a = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'phi','bra_ket':[[0,1]]})
    #c = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'n','bra_ket':[[1,2],[0,2]]})
    #b = fluxoniumq.plot1Dspectrum('eigen_spectrum', {'levels':levels,'transition':[True,True]})
    #d = fluxoniumq.plot1Dspectrum('T1_die_loss', {'times':[1,2]})
    #e = fluxonium.plot1Dspectrum('T_dephasing', {'transition':[1,2]})
    #f = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'sin(phi/2)','bra_ket':[[0,1]]})
    #g = fluxonium.plot1Dspectrum('T_qp_sing_junc', {'xqp_list':[4e-7],'transition':[0,1]})
    #h = fluxonium.plot1Dspectrum('T_qp_array', {'xqp_list':[4e-7],'transition':[0,2]})
