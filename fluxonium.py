# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:55:59 2021

@author: edwar
"""

import numpy as np
from scipy.linalg import expm
from inspect import getfullargspec as showarg

class Fluxonium():
    def __init__(self,Ej,Ec,El,cutoff):
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
    
    def get_x(self,start = -1,
                   stop = 1,
                   points = 201):
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
        self.x = np.linspace(start,stop,points)
        return self.x
    
    def hamiltonian_exp(self,phi):
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
            n_square = np.dot(self.op_n(),self.op_n())
            phi_square = np.dot(self.op_phi(),self.op_phi())
            cos_ex =  1./2.*np.exp(-1j*phi*np.pi*2.)*expm(1j*self.op_phi())\
                     +1./2.*np.exp(1j*phi*np.pi*2.)*expm(-1j*self.op_phi())
            

            return 4*self.Ec*n_square + 0.5*self.El*phi_square -self.Ej*cos_ex
        
    def op_phi(self):
        """
        create the operator for the change of phase (normalized flux)

        Returns
        -------
        np.array 2D array

        """
        n = self.cutoff + 1
        c_p = np.zeros([n, n])
        c_m = np.zeros([n, n])
        for row in range(n):
            for column in range(n):
                if row == column - 1:
                    c_p[row, column] = np.sqrt(row+1.)
                elif row == column + 1:
                    c_m[row, column] = np.sqrt(row)
        
        phi_0 = (8.*self.Ec/self.El)**(1./4.)
        return phi_0/np.sqrt(2.)*(c_p + c_m)
    
    def op_n(self):
        """
        create the operator for the change of charge (normalized charge)

        Returns
        -------
        np.array 2D array

        """
        n = self.cutoff + 1
        c_p = np.zeros([n, n])
        c_m = np.zeros([n, n])
        for row in range(n):
            for column in range(n):
                if row == column - 1:
                    c_p[row, column] = np.sqrt(row+1.)
                elif row == column + 1:
                    c_m[row, column] = np.sqrt(row)
        
        phi_0 = (self.El/(8*self.Ec))**(1./4.)
        return 1j*phi_0/np.sqrt(2.)*(c_p - c_m)
    
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
        w, v = np.linalg.eigh(H)
        
        idx = w.argsort()
        w = np.sort(w)
        
        v = v[:,idx]
        v = np.transpose(v.real)
        return w, v
    
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
        if operator == "phi":
            operate = self.op_phi()
        elif operator == "n":
            operate = self.op_n()
            
        
        matrix_element = np.zeros((len(bra_ket),len(self.x)))
        
        for colum, i in  enumerate(self.x):
            H = self.hamiltonian_exp(i)
            _,temp = self.eigen(H)
            for roll, transition in enumerate(bra_ket):
                matrix_element[roll][colum] = abs(np.dot(temp[:][transition[0]].T,np.dot(operate,temp[:][transition[1]])))
        return np.array(matrix_element)
    
    def T1_die_loss(self,El_array):
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
        El_array = np.append(El_array,self.El)
        kB = 1.38064852e-23
        T = 0.01 #10 mk
        h_bar = 6.62e-34/(2*np.pi)    #Placnk's reduce constant
        
        res = np.zeros((len(El_array),len(self.x)))
        for index,i in enumerate(El_array):
            self.El = i
            w = self.eigen_spectrum(3,[True,True])[1]*1e9*2*np.pi
            y_phi_ele = self.matrix_ele('phi', [[0,1]])[0]
            
            
            Ec_hbar = h_bar*self.Ec*2*np.pi*1e9
            Q_cap = 1e6*(2*np.pi*6e9/w)**(-0.1)
            
            gamma_cap = h_bar/(4*Ec_hbar*Q_cap)
            gamma_cap = gamma_cap*(w**2)*((y_phi_ele)**2)
            for idx in range(len(gamma_cap)):
                gamma_cap[idx] = gamma_cap[idx]*abs(coth(h_bar*w[idx]/(2*kB*T))+1)
            res[index] = np.log10(1e6/(2*np.pi*gamma_cap))
        return res
    
    def function_mappings(self):
        """
        str to func mappings, the IO for Plot1DSpectrum
        """
        return {
            'eigen_spectrum': self.eigen_spectrum,
            'matrix_ele': self.matrix_ele,
            'T1_die_loss': self.T1_die_loss
            }

    def setFunc(self,
            func: str, funcArg: list,
            start: float = -1, stop: float = 1,points: int = 201
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
            'X': {'start': start, 'stop': stop, 'points': points},
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
        
        # x
        start = generator['X']['start']
        stop = generator['X']['stop']
        points = generator['X']['points']
        self.get_x(start,stop,points)
        
        # y
        func = generator['function']
        if isinstance(func, str):
            func = self.function_mappings()[func]
        argNames = showarg(func).args[1:]
        funcArg = [generator['Y'][arg] for arg in argNames]
        y = func(*funcArg)
        
        
        return self.x, y
    
    def plot1Dspectrum(self,func_name, argdict,start = -1,stop = 1,points = 101):
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
        
        generator = self.setFunc(func_name, argdict,start,stop,points)
        x,y = self.parse(generator)
        
        figure()
        for yi in y:
            plot(x,yi)
        show()
        return x,y
    
if __name__ == "__main__":
    """
    1, dispersive shift......
    """
    from mpmath import coth
    
    start = 0
    stop = 0.5
    points = 101
    levels = 9
    
    loop_size = 3
    res = np.zeros((loop_size,points))

    Ej = 5
    Ec = 1
    El = 1
    
    fluxonium = Fluxonium(Ej,Ec,El,101)
    
    #a = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'phi','bra_ket':[[1,2],[0,2],[0,1]]})
    #c = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'n','bra_ket':[[1,2],[0,2],[0,1]]})
    #b = fluxonium.plot1Dspectrum('eigen_spectrum', {'levels':levels,'transition':[False,False]},-1,1,101)
    #d = fluxonium.plot1Dspectrum('T1_die_loss', {'El_array':[0.5,0.75]},0,0.5,201)
        