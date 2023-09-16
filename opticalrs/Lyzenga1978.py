# -*- coding: utf-8 -*-
"""
Lyzenga1978
===========

This module implements some of the methods described in Lyzenga 1978. These
methods are used in depth estimation from multispectral imagery and in water
column correction for bottom type classification.

The methods in this module are mostly derived from Appendix B of Lyzenga 1978.
It's pretty unlikely that anything in this module will make any sense to you if
you don't read Appendix B.

This implementation is the work of the author of this code (Jared Kibele), not
the author of the original paper. I tried to get it  right but I'm not making
any promises. Please check your results and let me  know if you find any
problems.

If you want to calculate depth-invariant indices, it would go something like
this::

    >>> b = slopes(Z,X)
    >>> A = Aij(b)
    >>> di = depth_invariant(A,X)

References
----------
Lyzenga, D.R., 1978. Passive remote sensing techniques for mapping water depth
and bottom features. Appl. Opt. 17, 379–383. doi:10.1364/AO.17.000379

Developed in ClassificationDev/Lyzenga/Lyzenga1978/AppendixB.ipynb
"""

from scipy.stats import linregress
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from ArrayUtils import equalize_array_masks

def regressions(Z,X):
    """
    Carry out the regression discussed in Appendix B of Lyzenga 1978. `Z` is
    the depths and `X` is the log radiance values. X = a - b z is the
    formula so depth is the x in the regression and radiance is the y.

    Parameters
    ----------
    Z : np.ma.MaskedArray
        Array of depth values.
    X : np.ma.MaskedArray
        The array of log transformed radiance values from equation B1 of
        Lyzenga 1978.

    Returns
    -------
    np.array
        A 3 x N-bands array containing the slopes, intercepts, and r_values
        of the regressions. In terms of equation B1 from Lyzenga 1978, that
        would be (b_i, a_i, r_value_i) for each band.
    """
    nbands = X.shape[-1]
    if np.atleast_3d(Z).shape[-1] == 1:
        Z = np.repeat(np.atleast_3d(Z), nbands, 2)
    slopes = []
    intercepts = []
    rvals = []
    for band in range(nbands):
        z, x = equalize_array_masks(Z[...,band], X[...,band])
        # print z.count(), x.count()
        try:
            slope,intercept,r_value,p_value,std_err = linregress(z.compressed(),\
                                                                 x.compressed())
        except IndexError:
            slope,intercept,r_value,p_value,std_err = np.repeat(np.nan, 5, 0)
        # eq B1 is X=a-bz
        slopes.append( slope )
        intercepts.append( intercept )
        rvals.append( r_value )
    return np.array((slopes,intercepts,rvals))

def slopes(Z,X):
    """
    Get the slopes from the regression of Z against X. See `regressions` doc
    string for more information. In terms of Lyzenga 1978, the slopes are the
    b_i values for each band.

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
    np.array
        An N-bands array containing the slopes of the regressions. In terms of
        equation B1 from Lyzenga 1978, that's the b_i values for each band.
    """
    return regressions(Z,X)[0,:]

def B2(b,j):
    """
    Equation B2 from Lyzenga 1978.

    Parameters
    ----------
    b : np.array
        Slopes from regressing depths against logged radiance values.
        For more info see the doc string for `Lyzenga1978.regressions`
        or Appendix B of Lyzenga 1978.
    j : int
        Band number.

    Returns
    -------
    np.array
        Elements of the last row of the tranformation matrix A. If that doesn't
        mean anything to you, read Appendix B of Lyzenga 1978.
    """
    return b[j] * (b**2).sum()**(-0.5)

def B5(b,i,j):
    """
    Equation B5 from Lyzenga 1978.

    Parameters
    ----------
    b : np.array
        Slopes from regressing depths against logged radiance values.
        For more info see the doc string for `Lyzenga1978.regressions`
        or Appendix B of Lyzenga 1978.
    i : int
        Band number.
    j : int
        Band number.

    Returns
    -------
    np.array
        Elements of the tranformation matrix A. If that doesn't mean anything
        to you, read Appendix B of Lyzenga 1978.
    """
    return b[i+1] * b[j] * ( (b[:i+1]**2).sum()**(-0.5) ) * ( (b[:i+2]**2).sum()**(-0.5) )

def B6(b,i,j):
    """
    Equation B6 from Lyzenga 1978.

    Parameters
    ----------
    b : np.array
        Slopes from regressing depths against logged radiance values.
        For more info see the doc string for `Lyzenga1978.regressions`
        or Appendix B of Lyzenga 1978.
    i : int
        Band number.
    j : int
        Band number.

    Returns
    -------
    np.array
        Elements of the tranformation matrix A. If that doesn't mean anything
        to you, read Appendix B of Lyzenga 1978.
    """
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
    np.array
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
    """
    Calculate band `i` of the coordinate rotation described in equation 8 of
    Lyzenga 1978.

    Parameters
    ----------
    i : int
        The band number of `Y` to calculate.
    A : np.array
        The coordinate system rotation parameters calculated by the method
        `Aij` and described in Appendix B of Lyzenga 1978.
    X : np.ma.MaskedArray
        The array of log transformed radiance values from equation B1 of
        Lyzenga 1978.

    Returns
    -------
    np.array
        A single band of the bands described in equation 8 of Lyzenga 1978. The
        band will be depth-invariant if `i` < `N` (the number of bands).
    """
    return (A[i,:] * X[...,:]).sum(2)

def depth_invariant(A,X):
    """
    Calculate N-1 depth-invariant bands and single depth dependent band by
    coordinate rotation described in equation 8 of Lyzenga 1978.

    Parameters
    ----------
    A : np.array
        The coordinate system rotation parameters calculated by the method
        `Aij` and described in Appendix B of Lyzenga 1978.
    X : np.ma.MaskedArray
        The array of log transformed radiance values from equation B1 of
        Lyzenga 1978.

    Returns
    -------
    np.array
        N-1 depth-invariant bands and single depth dependent band by coordinate
        rotation described in equation 8 of Lyzenga 1978.
    """
    Yi = []
    N = X.shape[-1]
    for i in range( N ):
        Yi.append( Y_i(i,A,X) )
    return np.ma.dstack( Yi )


def regression_plot(Z,X,band_names=None,visible_only=True,figsize=(12,7)):
    """
    Produce a figure with a plot for each image band that displays the
    relationship between depth and radiance and gives a visual representation
    of the regression carried out in the `slopes` and `regressions` methods.

    Notes
    -----
    This method doesn't come directly from Lyzenga 1978 but the author of this
    code found it helpful.

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
    figure
        A matplotlib figure.
    """
    if band_names is None:
        band_names = ['Band'+str(i+1) for i in range(X.shape[-1])]
    nbands = X.shape[-1]
    if np.atleast_3d(Z).shape[-1] == 1:
        Z = np.repeat(np.atleast_3d(Z), nbands, 2)
    if visible_only:
        fig, axs = plt.subplots( 2, 3, figsize=figsize)
    else:
        fig, axs = plt.subplots( 2, 4, figsize=figsize )
    regs = regressions(Z,X)
    for i, ax in enumerate(axs.flatten()):
        if i > nbands-1:
            continue
        slp, incpt, rval = regs[:,i]
        # print X.shape, Z.shape
        x, y = equalize_array_masks(Z[...,i], X[...,i])
        if x.count() < 2:
            continue
        x, y = x.compressed(), y.compressed()
        # print "i = {}, x.shape = {}, y.shape = {}".format(i, x.shape, y.shape)
        ax.scatter( x, y, alpha=0.1, edgecolor='none', c='gold' )
        smth = lowess(y,x,frac=0.2)
        # ax.plot(smth.T[0],smth.T[1],c='black',alpha=0.5)
        ax.plot(smth.T[0],smth.T[1],c='black',alpha=0.5,linestyle='--')
        reglabel = "m=%.2f, r=%.2f" % (slp,rval)
        f = lambda x: incpt + slp * x
        ax.plot( x, f(x), c='brown', label=reglabel, alpha=1.0 )
        ax.set_title( band_names[i] )
        ax.set_xlabel( r'Depth (m)' )
        ax.set_ylabel( r'$X_i$' )
        ax.legend(fancybox=True, framealpha=0.5)
    plt.tight_layout()
    return fig
