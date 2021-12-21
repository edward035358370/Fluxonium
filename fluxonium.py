# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:55:59 2021

@author: edwar
"""

import numpy as np
from scipy.linalg import expm
from inspect import getfullargspec as showarg

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
    
    def op_sin_phi(self,phi_ext):
        ope = 1.0j * (self.op_phi() + np.eye(self.cutoff+1)*phi_ext*2*np.pi)
        sine_ope = (expm(ope/2.0) - expm(-ope/2.0))/2j
        
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
        w, v = np.linalg.eig(H)
        
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
        
        res = np.zeros((len(El_array),self.points))
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
    1, there are some problem in solve the sin(phi/2) matrix element.
    the reason I guess is about the eigen state be solved.
    (but the result of other operator is true)
    
    2, dispersive shift......
    """
    from mpmath import coth
    
    start = 0
    stop = 1
    points = 201
    levels = 9
    
    loop_size = 3
    res = np.zeros((loop_size,points))

    Ej = 3
    Ec = 1
    El = 1
    
    fluxonium = Fluxonium(Ej,Ec,El,101,start=start,stop=stop,points=points)
    
    #a = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'phi','bra_ket':[[0,1],[0,2]]})
    #c = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'n','bra_ket':[[1,2],[0,2]]})
    #b = fluxonium.plot1Dspectrum('eigen_spectrum', {'levels':levels,'transition':[True,True]})
    #d = fluxonium.plot1Dspectrum('T1_die_loss', {'El_array':[0.5,0.75]})
    #e = fluxonium.plot1Dspectrum('T_dephasing', {'transition':[1,2]})
    f = fluxonium.plot1Dspectrum('matrix_ele', {'operator':'sin(phi/2)','bra_ket':[[0,1]]})
    #g = fluxonium.plot1Dspectrum('T_qp_sing_junc', {'xqp_list':[4e-7],'transition':[0,1]})
    #h = fluxonium.plot1Dspectrum('T_qp_array', {'xqp_list':[4e-7],'transition':[0,1]})
    '''
    temp1 = fluxonium.T_qp_array([4e-7],[0,1])[0]
    temp2 = fluxonium.T_qp_sing_junc([4e-7],[0,1])[0]
    res = []
    for i in range(201):
        if temp1[i] >= temp2[i]:
            res.append(temp2[i])
        else:
            res.append(temp1[i])
    '''