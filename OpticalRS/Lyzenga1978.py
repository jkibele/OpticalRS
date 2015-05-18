# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:18 2015

@author: jkibele
"""

from scipy.stats import linregress
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def regressions(Z,X):
    """
    Z is the depths and X is the log radiance values. X = a - b z is the
    formula so depth is the x in the regression and radiance is the y.
    
    Parameters
    ----------
      Z : np.ma.MaskedArray
        Array of depth values repeated for each band so that Z.shape==X.shape.
        The mask needs to be the same too so that Z.mask==X.mask for all the
        bands.
      X : np.ma.MaskedArray
        The array of log transformed radiance values from equation B1 of
        Lyzenga 1978.
        
    Returns
    -------
      regression array : np.array
        A 3 x N-bands array containing the slopes, intercepts, and r_values
        of the regressions. In terms of equation B1 from Lyzenga 1978, that
        would be (b_i, a_i, r_value_i) for each band.
    """
    nbands = X.shape[-1]
    slopes = []
    intercepts = []
    rvals = []
    for band in range(nbands):
        slope,intercept,r_value,p_value,std_err = linregress(Z[...,band].compressed(),\
                                                                 X[...,band].compressed())
        # eq B1 is X=a-bz
        slopes.append( slope )
        intercepts.append( intercept )
        rvals.append( r_value )
    return np.array((slopes,intercepts,rvals))
    
def slopes(Z,X):
    """
    Get the slopes from the regression of Z against X. See `regressions` doc
    string for more information.
    
    Parameters
    ----------
      Z : np.ma.MaskedArray
        Array of depth values repeated for each band so that Z.shape==X.shape.
        The mask needs to be the same too so that Z.mask==X.mask for all the
        bands.
      X : np.ma.MaskedArray
        The array of log transformed radiance values from equation B1 of
        Lyzenga 1978.
        
    Returns
    -------
      slopes array : np.array
        An N-bands array containing the slopes of the regressions. In terms of 
        equation B1 from Lyzenga 1978, that's the b_i values for each band.
    """
    return regressions(Z,X)[0,:]
    
def B2(b,j):
    return b[j] * (b**2).sum()**(-0.5)
    
def B5(b,i,j):
    return b[i+1] * b[j] * ( (b[:i+1]**2).sum()**(-0.5) ) * ( (b[:i+2]**2).sum()**(-0.5) )  
    
def B6(b,i,j):
    return -1.0 * ( (b[:i+1]**2).sum()**(0.5) ) * ( (b[:i+2]**2).sum()**(-0.5) )
    
def Aij(b):
    """
    Return coordinate system rotation parameters for use in caclulating
    depth invariant index (equation 8, Lyzenga 1978).
    
    Parameters
    ----------
      b : np.array
        Slopes from regressing depths against logged radiance values.
        For more info see the doc string for `Lyzenga1978.regressions`
        or Appendix B of Lyzenga 1978.
        
    Returns
    -------
      A : np.array
        Coordinate system rotation parameters for use in caclulating
        depth invariant index (equation 8, Lyzenga 1978). See equation
        8 and Appendix B of Lyzenga 1978.
    """
    N = len(b)
    A = np.empty( (N,N), dtype='float' )
    for i in range( N ):
        for j in range( N ):
            if i==N-1: # python is zero indexed
                A[i,j] = B2(b,j)
            elif j<=i:
                A[i,j] = B5(b,i,j)
            elif j==i+1:
                A[i,j] = B6(b,i,j)
            else:
                A[i,j] = 0.0
    return A
    
def Y_i(i,A,X):
    return (A[i,:] * X[...,:]).sum(2)
    
def depth_invariant(A,X):
    Yi = []
    N = X.shape[-1]
    for i in range( N ):
        Yi.append( Y_i(i,A,X) )
    return np.ma.dstack( Yi )
    
    
def regression_plot(Z,X,band_names=None):
    if band_names is None:
        band_names = ['band'+str(i) for i in range(X.shape[-1])]
    fig, axs = plt.subplots( 4, 2, figsize=(10,20) )
    regs = regressions(Z,X)
    for i, ax in enumerate(axs.flatten()):
        slp, incpt, rval = regs[:,i]
        x = Z[...,i].compressed()
        y = X[...,i].compressed()
        smth = lowess(y,x,frac=0.2)
        ax.scatter( x, y, alpha=0.05, edgecolor='none' )
        ax.plot(smth.T[0],smth.T[1],c='white',alpha=0.75)
        reglabel = "b=%.2f, r=%.2f" % (slp,rval)
        f = lambda x: incpt + slp * x
        ax.plot( x, f(x), c='r', label=reglabel )
        ax.set_title( band_names[i] )
        ax.set_xlabel( r'$z$' )
        ax.set_ylabel( r'$X_i$' )
        ax.legend()