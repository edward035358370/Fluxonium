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
    """
    1, For characterizing the single junction fluxonium
    in different Ej, Ec, El.
    
    2, For quickly, can use plot 1D spectrum to see the
    every function result, or can directly use the 
    function you want to count.
    
    3, The way to use plot1Dspectrum is under the main
    can be used.
    
    4, 
    """
    def __init__(self,Ej1,Ej2,Ec,El,cutoff,start = -1,
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
        self.Ej1 = Ej1
        self.Ej2 = Ej2
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
    
    def hamiltonian_exp(self,phi_total):
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
        

        phi_ext = phi_total*(10/11)*2*np.pi
        phi_squid = phi_total*(1/11)*2*np.pi
        phi1 = 1.0j * (-phi_ext + self.op_phi())
        phi2 = 1.0j * (self.op_phi() - phi_squid - phi_ext)
        
        return 4.0 * self.Ec * self.op_n() ** 2.0 + 0.5 * self.El * self.op_phi() ** 2.0\
            - 0.5 * self.Ej1* (phi1.expm() + (-phi1).expm())- 0.5 * self.Ej2 * (phi2.expm() + (-phi2).expm())
        
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
        na = 1.0j * (a.dag() - a) * (self.El / (8 * self.Ec)) ** (0.25) / np.sqrt(2.0)
        return na
    
    def op_sin_phi(self,phi_ext):
        """
        

        Parameters
        ----------
        phi_ext : float 
            (external flux without 2pi).

        Returns
        -------
        sine_ope : 
            np.array 2D array

        """
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
    
    
    def function_mappings(self):
        """
        str to func mappings, the IO for Plot1DSpectrum
        """
        return {
            'eigen_spectrum': self.eigen_spectrum,
            'matrix_ele': self.matrix_ele
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
        
        """

        if isinstance(func, str):
            name = func
            func = self.function_mappings()[func]
        argNames = showarg(func).args[1:] #get function args besides x
        if isinstance(funcArg, dict):
            funcArg = [funcArg[arg] for arg in argNames]
    
        generator = {
            'function': func.__name__, #__name__is function's name
            'X': {'start': self.start, 'stop': self.stop, 'points': self.points},
            'Y': dict(zip(argNames, funcArg)),
            'name': name
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
        title : str
                for title of the plot
        """

        # y
        func = generator['function']
        if isinstance(func, str):
            func = self.function_mappings()[func]
        argNames = showarg(func).args[1:]
        funcArg = [generator['Y'][arg] for arg in argNames]
        y = func(*funcArg)
        title = generator['name']
        
        
        return self.x, y, title
    
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
        from pylab import figure,plot, show,title
        
        generator = self.setFunc(func_name, argdict)
        x,y,name = self.parse(generator)
        
        figure(0)
        for yi in y:
            plot(x,yi)
            title(name)
        show()
        return x,y
    
if __name__ == "__main__":     
    """
    1, didn't fit Long's thesis
    """
    
    start = 0
    stop = 12
    points = 1201
    levels = 3
    
    loop_size = 3
    res = np.zeros((loop_size,points))

    Ej1 = 5.5
    Ej2 = 4.5
    Ec = 1
    El = 1
    
    fluxonium = Fluxonium(Ej1,Ej2,Ec,El,101,start=start,stop=stop,points=points)
    
    #a = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'phi','bra_ket':[[0,1]]})
    c = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'n','bra_ket':[[0,2]]})
    #b = fluxonium.plot1Dspectrum('eigen_spectrum', {'levels':levels,'transition':[True,True]})
   
