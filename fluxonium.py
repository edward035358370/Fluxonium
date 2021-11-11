# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:55:59 2021

@author: edwar
"""

import numpy as np
from scipy.linalg import expm

class Fluxonium():
    def __init__(self,Ej,Ec,El,cutoff,
                                start = 0,
                                stop = 0.5,
                                points = 101):
        self.Ej = Ej
        self.Ec = Ec
        self.El = El
        self.phi = np.linspace(start,stop,points)
        self.points = points
        self.cutoff = cutoff
        
    def hamiltonian_exp(self,phi):
            """
            Return the Hamiltonian matrix with energy in GHz calculated with
            the exponential method.
            Parameters
            ----------
            n : int
                Dimension of the matrix.
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
        create the operator for external flux

        Returns
        -------
        TYPE
            DESCRIPTION.

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
        create the operator for external charge

        Returns
        -------
        TYPE
            DESCRIPTION.

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
        w, v = np.linalg.eigh(H)
        
        idx = w.argsort()
        w = np.sort(w)
        
        v = v[:,idx]
        v = np.transpose(v.real)
        return w, v
    
    def harmonic_oscilator_energy(self,n):
        return np.sqrt(8.*Ec*El)*(n + 0.5)
    
    def eigen_spectrum(self,levels):
        
        energy = []
        for i in range(levels):
            energy.append([])
        energy = np.array(energy)
        
        for i in self.phi:
            H = self.hamiltonian_exp(i)
            temp,_ = self.eigen(H)
            energy = np.concatenate((energy,np.reshape(temp[0:levels],(levels,1))),axis=1)
            
        spectrum = np.linspace(0,0,self.points)
        for high in range(len(energy)):
            for low in range(len(energy)):
                if high > low:
                    spectrum = np.vstack((spectrum,energy[high] - energy[low]))
                
        return spectrum[1:,:]
    
    def matrix_ele(self,operator,bra,ket):
        if operator == "phi":
            operate = self.op_phi()
        elif operator == "n":
            operate = self.op_n()
            
    
        matrix_element = []
        
        for i in self.phi:
            H = self.hamiltonian_exp(i)
            _,temp = self.eigen(H)
            matrix_element.append(abs(np.dot(temp[:][bra].T,np.dot(operate,temp[:][ket]))))
        return np.array(matrix_element)
    
    
    def Plot1DSpectrum(self):
        from pylab import plot,show
        
        func = self.matrix_ele("phi",0,2)
        plot(self.phi,func)
        show()

        
                    
        
if __name__ == "__main__":
    
    Ej = 5
    Ec = 1
    El = 1
    start = 0
    stop = 0.5
    points = 11
    levels = 3
    limit = 5.0

    fluxonium = Fluxonium(Ej,Ec,El,101)
    fluxonium.Plot1DSpectrum()
    
    