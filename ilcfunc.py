# This file contains functions needed to do ILC.

import numpy as np
import sys

# ## Conversion from antenna to thermodynamic temperature
h = 6.62606957E-34
k = 1.3806488E-23
Tcmb = 2.725
c = 3e8
I0 = 2 * (k * Tcmb) ** 3 / (h*c) ** 2 * 1e20

def fv_d(Freq, Td, betad):  # dust spectrum (greybody) for temperature map
    x = h*Freq/(k*Tcmb)
    cv = ((x ** 2 * np.exp(x)) / (np.exp(x) - 1) ** 2) ** (-1)
    # cv = array([1.288, 1.657, 3.003, 13.012])

    # ## Component
    x100 = h*100e9/(k*Tcmb)
    # CMB`
    # Sigma = np.array([1, 1, 1])
    dbeta = betad
    xd = h*Freq/(k*Td)
    xd100 = h*100e9/(k*Td)
    cv100 = ((x100 ** 2 * np.exp(x100)) / (np.exp(x100) - 1) ** 2) ** (-1)
    B = (Freq/100e9) ** 3 / (np.exp(xd) - 1) * (np.exp(xd100) - 1)
    # Thermal dust`
    Spec_d_ant = (Freq/100E9)**(dbeta-2) * B
    Spec_d = cv*Spec_d_ant/cv100
    return Spec_d


def fv_d_pl(Freq, betad):  # dust spectrum (powerlaw) for temperature map.
    x = h*Freq/(k*Tcmb)
    cv = ((x ** 2 * np.exp(x)) / (np.exp(x) - 1) ** 2) ** (-1)
    # cv = array([1.288, 1.657, 3.003, 13.012])
    # ## Component
    x100 = h*100e9/(k*Tcmb)
    # CMB`
    # Sigma = np.array([1, 1, 1])
    dbeta = betad
    cv100 = ((x100 ** 2 * np.exp(x100)) / (np.exp(x100) - 1) ** 2) ** (-1)
    # Thermal dust`
    Spec_d_ant = (Freq/100E9)**(dbeta)
    Spec_d = cv*Spec_d_ant/cv100
    return Spec_d


def fv_y(Freq):   # y spectrum for temperature map
    x = h*Freq/(k*Tcmb)
    xd100 = h*100e9/(k*Tcmb)
    Spec_y = (x/np.tanh(x/2.) - 4) /(xd100/np.tanh(xd100/2.) - 4)
    return Spec_y


def fv_d_i(Freq,Td,betad):
    # ## Component
    # CMB`
    # Sigma = np.array([1, 1, 1])
    dbeta = betad
    xd = h*Freq/(k*Td)
    xd100 = h*143e9/(k*Td)
    Spec_d = (Freq/143E9)**(dbeta+3) / (np.exp(xd) - 1) * (np.exp(xd100) - 1) 
    return Spec_d

def fv_d_pl_i(Freq,betad):
    Spec_d_ant = (Freq/143E9)**(betad+2)
    Spec_d = Spec_d_ant
    return Spec_d


def fv_y_i(Freq):
    x = h*Freq/(k*Tcmb)
    xd100 = h*143e9/(k*Tcmb)
    Spec_y = ((x/np.tanh(x/2.) - 4) * x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2) / ((xd100/np.tanh(xd100/2.) - 4) * xd100 ** 4 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2)
    return Spec_y


def fv_cmb_i(Freq):
    x = h*Freq/(k*Tcmb)
    xd100 = h*143e9/(k*Tcmb)
    Spec_cmb = (Freq ** 2 * x ** 2 * np.exp(x) / (np.exp(x) - 1) ** 2) / (143e9 ** 2 * xd100 ** 2 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2)
    return Spec_cmb


def Wmatrix(freqs, Td='NULL', dbeta='NULL', beta1='NULL', beta2='NULL'):  # calculate weight matrix for all components
    fvy = fv_y_i(freqs*1e9)
    fvc = fv_cmb_i(freqs*1e9)

    W = np.vstack([fvy, fvc])
    if Td != 'NULL':
        fvd = fv_d_i(freqs*1e9, Td, dbeta)
        W = np.vstack([W, fvd])
    fvcib = np.zeros(freqs.size)
    if beta1 != 'NULL':
        for i in range(freqs.size):
            fvcib[i] = fv_cib_i(freqs[i]*1e9, beta1, beta2)
        W = np.vstack([W, fvcib])
    return W


def ilc_cov_y(maps, freqs, cov, W):  # calculate y signal. CIB, CMB and dust subtracted
    Ninv = np.linalg.inv(cov)
    covinv = np.dot(W, np.dot(Ninv, W.T))
    cov = np.linalg.inv(covinv)
    matrix = np.dot(cov, np.dot(W, Ninv))
    xd100 = h*143e9/(k*Tcmb)
    matrix[0] /= (xd100/np.tanh(xd100/2.) - 4) * xd100 ** 4 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2
    coef = matrix[0] / I0
    ymap = np.dot(coef, maps)
    return ymap, coef


def ilc_cov_coef(fvc, fvd, fvy, Ninv):
    W = np.vstack([fvc, fvd, fvy])
    covinv = np.dot(W, np.dot(Ninv, W.T))
    cov = np.linalg.inv(covinv)
    coef = np.dot(cov, np.dot(W, Ninv))
    xd100 = h*100e9/(k*Tcmb)
    coef /= (xd100/np.tanh(xd100/2.) - 4)
    return coef


##### some old out-dated codes ######

def ilc_dustmodel_i(Freq,Td,dbeta):
    Spec_d = fv_d_i(Freq,Td,dbeta)
    Spec_y = fv_y_i(Freq)
    Nf = len(Freq)
    Sigma = fv_cmb_i(Freq)
    Ns = 3
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Sigma
    Matrix[1] = Spec_d
    Matrix[2] = Spec_y
    RHS = np.zeros(Nf)
    RHS[2] = 1.

    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.linalg.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    xd100 = h*143e9/(k*Tcmb)
    Soln /= (xd100/np.tanh(xd100/2.) - 4) * xd100 ** 4 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2
    return Soln

def ilc_dustmodel_pl_i(Freq,dbeta):
    Spec_d = fv_d_pl_i(Freq,dbeta)
    Spec_y = fv_y_i(Freq)
    Nf = len(Freq)
    Sigma = fv_cmb_i(Freq)
    Ns = 3
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Sigma
    Matrix[1] = Spec_d
    Matrix[2] = Spec_y
    RHS = np.zeros(Nf)
    RHS[2] = 1.

    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.linalg.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    xd100 = h*143e9/(k*Tcmb)
    Soln /= (xd100/np.tanh(xd100/2.) - 4) * xd100 ** 4 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2
    return Soln

def fv_cib_i(Freq,beta1,beta2):
    if Freq <= 545e9:
        return Freq ** beta1 / (143.*1e9) ** beta1
    else:
        return (Freq / (545. * 1e9)) ** beta2 * (545. / 143. ) ** beta1

def ilc_dustmodel_cmb(Freq,Td,dbeta):
    Spec_d = fv_d_i(Freq,Td,dbeta)
    Spec_y = fv_y_i(Freq)
    Sigma = fv_cmb_i(Freq)
    Spec_cib = np.zeros(Freq.size)
    for i in range(Freq.size):
        Spec_cib[i] = fv_cib_i(Freq[i])
    Nf = len(Freq)
    #    Sigma = np.ones(Freq.size)
    Ns = 4
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()
        
    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Sigma
    Matrix[1] = Spec_d
    Matrix[2] = Spec_y
    Matrix[3] = Spec_cib
    RHS = np.zeros(Nf)
    RHS[0] = 1.
    
    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.linalg.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    xd100 = h*143e9/(k*Tcmb)
    Soln /= (2 * k / c / c * (143e9 ** 2 * xd100 ** 2 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2))
    return Soln

def ilc_y_cib(Freq,Td,dbeta,beta1,beta2):
    Spec_d = fv_d_i(Freq,Td,dbeta)
    Spec_y = fv_y_i(Freq)
    Sigma = fv_cmb_i(Freq)
    Spec_cib = np.zeros(Freq.size)
    for i in range(Freq.size):
        Spec_cib[i] = fv_cib_i(Freq[i],beta1,beta2)
    Nf = len(Freq)
    Ns = 4
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Spec_y
    Matrix[1] = Spec_d
    Matrix[2] = Sigma
    Matrix[3] = Spec_cib
    RHS = np.zeros(Nf)
    RHS[0] = 1.

    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.linalg.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    xd100 = h*143e9/(k*Tcmb)
    Soln /= (xd100/np.tanh(xd100/2.) - 4) * xd100 ** 4 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2
    return Soln


def ilc_cov(N,d):
    Ninv = np.linalg.inv(N)
    fvc = fv_d_i(Freq,Td,dbeta)
    fvd = fv_y_i(Freq)
    fvy = fv_cmb_i(Freq)
    W = np.vstack([fvc,fvd,fvy,0,0,0])
    covinv = np.dot(W,np.dot(Ninv,W.T))
    cov = np.linalg.inv(covinv)
    delta = np.dot(cov,np.dot(W,np.dot(Ninv,d)))
    xd100 = h*143e9/(k*Tcmb)
    
    delta /= (2 * k / c / c * (143e9 ** 2 * xd100 ** 2 * np.exp(xd100) / (np.exp(xd100) - 1) ** 2))
    return delta




def ilc_gary(Freq,dbeta):  # ilc recipies used by Gary
    Spec_d = fv_d_pl(Freq,dbeta)
    Spec_y = fv_y(Freq)
    # Spec_y = array([-1.506, -1.037, -0.001, 2.253])
    # Spec_y = np.array([-4.031, -2.785, 0.187, 6.205]) / Tcmb

    Nf = len(Freq)
    Sigma = np.ones(Nf)
    Ns = 3
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Sigma
    Matrix[1] = Spec_d
    Matrix[2] = Spec_y
    RHS = np.zeros(Nf)
    RHS[2] = 1.

    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    xd100 = h*100e9/(k*Tcmb)
    Soln /= (xd100/np.tanh(xd100/2.) - 4)
    return Soln


def ilc_dustmodel(Freq,Td,dbeta):
    Spec_d = fv_d(Freq,Td,dbeta)
    Spec_y = fv_y(Freq)

    Nf = len(Freq)
    Sigma = np.ones(Nf)
    Ns = 3
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Sigma
    Matrix[1] = Spec_d
    Matrix[2] = Spec_y
    RHS = np.zeros(Nf)
    RHS[2] = 1.

    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    xd100 = h*100e9/(k*Tcmb)
    Soln /= (xd100/np.tanh(xd100/2.) - 4)
    return Soln


def ilc_lgs(): # use 143,217,353GHz
    Freq = np.array([143,217,353]) * 1e9
    R = np.diag([0.000917,0.001347,0.004558])
    x = h*Freq/(k*Tcmb)
    cv = ((x ** 2 * np.exp(x)) / (np.exp(x) - 1) ** 2) ** (-1)
    # cv = array([1.288, 1.657, 3.003, 13.012])
    # ## Component

    # CMB`
    # Sigma = np.array([1, 1, 1])

    # Thermal dust`
    dbeta = 1.8
    Spec_d_ant = (Freq/100E9)**(dbeta)
    Spec_d = cv*Spec_d_ant

    Spec_y = x/np.tanh(x/2.) - 4
    # Spec_y = array([-1.506, -1.037, -0.001, 2.253])
    # Spec_y = np.array([-4.031, -2.785, 0.187, 6.205]) / Tcmb

    Nf = len(Freq)
    Ns = 3
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = np.array([1,1,1])
    Matrix[1] = Spec_d
    Matrix[2] = Spec_y
    A = Matrix.transpose()
    print A
    print np.dot(np.dot(Matrix,np.linalg.inv(R)),A)
    W1 = np.linalg.inv(np.dot(np.dot(Matrix,np.linalg.inv(R)),A))
    W = np.dot(np.dot(W1,Matrix),np.linalg.inv(R))
    return W

def ilc_cib(Freq,Td,dbeta,beta1,beta2): #ILC to calculate CIB, normalized to 143GHz signal
    Spec_d = fv_d_i(Freq,Td,dbeta)
    Spec_y = fv_y_i(Freq)
    Sigma = fv_cmb_i(Freq)
    Spec_cib = np.zeros(Freq.size)
    for i in range(Freq.size):
        Spec_cib[i] = fv_cib_i(Freq[i],beta1,beta2)
    Nf = len(Freq)
    Ns = 4
    if(Nf < Ns):
        print "not enough freedom"
        sys.exit()

    # Populate the linear system to solve for the coefficients
    Matrix = np.zeros((Nf, Nf))
    Matrix[0] = Spec_y
    Matrix[1] = Spec_d
    Matrix[2] = Sigma
    Matrix[3] = Spec_cib
    RHS = np.zeros(Nf)
    RHS[0] = 1.

    # Soln = dot(pinv(Matrix), RHS)
    U, W, V = np.linalg.svd(Matrix)
    V = V.transpose()
    WW = np.zeros((Nf, Nf))
    for M in range(Nf):
        if(W[M] != 0.):
            WW[M, M] = 1. / W[M]
    Soln = np.dot(np.dot(np.dot(V, WW), U.transpose()), RHS)
    # Probe the degeneracy of the system
    Nd = Nf - Ns
    if(Nd > 0):
        Degenerate = np.where(W < 1e-12*max(abs(W)))[0]
        if(len(Degenerate) != Nd):
            print "Degenerate direction(s) not properly identified"
            sys.exit()

        Deg_Soln = V.T[Degenerate]

        # Minimize the degenerate solutions relative to the overall sensitivity
        if(Nd == 1):
            Alpha = -sum(Soln*Deg_Soln*(Sigma**2))/sum((Deg_Soln**2)*(Sigma**2))
            Soln1 = Soln + Alpha*Deg_Soln
        else:
            M_Alpha = np.zeros((Nd, Nd))
            RHS_Alpha = np.zeros(Nd)

            for i in range(Nd):
                RHS_Alpha[i] = -sum(Soln*Deg_Soln[i]*Sigma**2)
                for j in range(Nd):
                    M_Alpha[i, j] = sum(Deg_Soln[i] * Deg_Soln[j] * Sigma ** 2)

            Alpha = np.dot(np.linalg.inv(M_Alpha), RHS_Alpha)
            Soln_0 = Soln
            Soln = Soln_0 + np.dot(Deg_Soln.T, Alpha)
    
    return Soln
