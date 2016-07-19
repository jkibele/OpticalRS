# -*- coding: utf-8 -*-
"""
AlbedoIndex
===========

Code for generating a water column corrected image from multispectral imagery.
This is a method of water column correction for habitat mapping. It is based on
Maritorena et al. 1994 and is described in detail in Chapters 4 and 5 of my PhD
thesis (Kibele, In Review).

References
----------

Kibele, J. (In Review). Submerged habitats from space: Increasing map production
capacity with new methods and software. University of Auckland. PhD Thesis

Maritorena, S., Morel, A., Gentili, B., 1994. Diffuse Reflectance of Oceanic
Shallow Waters: Influence of Water Depth and Bottom Albedo. Limnology and
Oceanography 39, 1689–1703.

Philpot, W.D., 1987. Radiative transfer in stratified waters: a single-
scattering approximation for irradiance. Applied Optics 26, 4123.
doi:10.1364/AO.26.004123

Philpot, W.D., 1989. Bathymetric mapping with passive multispectral imagery.
Appl. Opt. 28, 1569–1578. doi:10.1364/AO.28.001569

Purkis, S.J., Pasterkamp, R., 2003. Integrating in situ reef-top reflectance
spectra with Landsat TM imagery to aid shallow-tropical benthic habitat mapping.
Coral Reefs 23, 5–20. doi:10.1007/s00338-003-0351-0

Sagawa, T., Boisnier, E., Komatsu, T., Mustapha, K.B., Hattour, A., Kosaka, N.,
Miyazaki, S., 2010. Using bottom surface reflectance to map coastal marine
areas: a new application method for Lyzenga’s model. International Journal of
Remote Sensing 31, 3051–3064. doi:10.1080/01431160903154341
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pylab import subplots
from matplotlib.pyplot import tight_layout
from Const import wv2_center_wavelength, jerlov_Kd

def myR0(z,Rinf,Ad,Kg):
    """
    This is the singly scattering irradiance (SSI) model (Philpot 1987) for
    irradiance reflectance immediately below the water surface for optically
    shallow, homogeneous water (eq. 2 from Philpot 1989). This model is
    essentially the same as the one discussed in appendix A of Lyzenga 1978.
    I've rearranged it a bit (from eq.2, Philpot 1989) but it's equivalent. This
    model is probably best described in Maritorena et al. 1994.

    Parameters
    ----------
    z : array-like
        Depth of water column.
    Rinf : float
        Irradiance reflectance of an optically deep water column.
    Ad : float or array-like of same size as `z`.
        Irradiance reflectance (albedo) of the bottom.
    Kg : float
        A 2 way effective attenuation coefficient of the water. Really
        equivalent to attenuation coefficient (K) times geometric factor (g).

    Returns
    -------
    R(0-) : array of floats
        Irradiance reflectance immediately below the water surface.
    """
    return Rinf + (Ad - Rinf) * np.exp(-1*Kg*z)

def param_df(zsand, Rsand, p0=None, geometric_factor=2.0):
    """
    Estimate the curve parameters using `est_curve_params` and return the
    results in a pandas dataframe.

    Parameters
    ----------
    zsand : array-like
        Depth of water column.
    Rsand : array-like
        Irradiance reflectance immediately below the water surface or, if you
        want to ignore units, atmospheric correction, and whatnot, just
        radiance values. This is a single band.
    p0 : None, scalar, or N-length sequence, optional
        Initial guess for the curve fitting parameters. If None, then the
        initial values will all be 1
    geometric_factor : float
        The geometric factor 'g' used to calculate the attenuation coefficient
        (K) from the estimated value for (Kg). For more information see the
        docstring for `OpticalRS.ParameterEstimator.geometric_factor`. To
        calculate the geometric factor for WorldView-2 imagery, you can use
        `OpticalRS.ParameterEstimator.geometric_factor_from_imd`.

    Returns
    -------
    pandas.dataframe
        A data frame with columns for 'Rinf', 'Ad', 'Kg', and 'K'. Each row
        represents one band of the imagery. Index is by wavelength for
        WorldView-2 imagery. Contact the author if you'd like to use this with
        some other type of imagery. It wouldn't be hard to change it to be more
        general but I don't have time right now and there's a good chance I'll
        forget all about it.
    """
    if Rsand.ndim > 2:
        nbands = Rsand.shape[-1]
    else:
        nbands = 1
    ind = wv2_center_wavelength[:nbands]
    params = est_curve_params(zsand, Rsand, p0=p0)
    cols = ['Rinf', 'Ad', 'Kg']
    paramdf = pd.DataFrame(params, columns=cols, index=ind)
    paramdf['K'] = paramdf.Kg / geometric_factor
    return paramdf

def est_curve_params(zsand, Rsand, p0=None):
    """
    Estimate `Rinf`, `Ad`, and `Kg` given sand depths `zsand` and corresponging
    radiances `Rsand`. Estimate is made by curve fitting using
    `scipy.optimize.curve_fit`.

    Parameters
    ----------
    zsand : array-like
        Depth of water column.
    Rsand : array-like
        Irradiance reflectance immediately below the water surface or, if you
        want to ignore units, atmospheric correction, and whatnot, just
        radiance values. This is a single band.
    p0 : None, scalar, or N-length sequence, optional
        Initial guess for the curve fitting parameters. If None, then the
        initial values will all be 1

    Returns
    -------
    np.array
        A 3 column row of parameters for each band of `Rsand`. Column 1 is the
        Rinf values, col 2 is the estAd values, and col 3 is the est_Kg values.
    """
    nbands = Rsand.shape[-1]
    outlist = []
    for i in range(nbands):
        params = est_curve_params_one_band(zsand, Rsand[...,i], p0=p0)
        outlist.append(params)
    return np.array(outlist)

def est_curve_params_one_band(zsand,Rsand,p0=None):
    """
    Estimate `Rinf`, `Ad`, and `Kg` given sand depths `zsand` and corresponging
    radiances `Rsand`. Estimate is made by curve fitting using
    `scipy.optimize.curve_fit`.

    Parameters
    ----------
    zsand : array-like
        Depth of water column.
    Rsand : array-like
        Irradiance reflectance immediately below the water surface or, if you
        want to ignore units, atmospheric correction, and whatnot, just
        radiance values. This is a single band.
    p0 : None, scalar, or N-length sequence, optional
        Initial guess for the curve fitting parameters. If None, then the
        initial values will all be 1

    Returns
    -------
    estRinf : float
        Estimated irradiance reflectance of an optically deep water column.
    estAd : float
        Estimated bottom albedo for `Rsand`.
    est_Kg : float
        Estimated 2 way effective attenuation coefficient of the water. Really
        equivalent to attenuation coefficient (K) times geometric factor (g).

    Notes
    -----
    `curve_fit` was failing to find a solution when the image array (`Rsand`)
    had a dtype of 'float64'. I don't really understand why that was a problem
    but explicitly casting the arrays to 'float32' seems to work. `curve_fit`
    uses `leastsq` which is a wrapper aound `MINPACK` which was writtin in
    Fortran a long time ago so, for now, it'll have to remain a mystery.
    """
    if np.ma.is_masked(zsand):
        zsand = zsand.compressed()
    if np.ma.is_masked(Rsand):
        Rsand = Rsand.compressed()
    p, pcov = curve_fit(myR0,zsand.astype('float32'),Rsand.astype('float32'),p0)
    estRinf, estAd, est_Kg = p
    return estRinf, estAd, est_Kg

def estAd_single_band(z,L,Rinf,Kg):
    """
    Estimate the albedo `Ad` for radiance `L` at depth `z` assuming `Rinf` and
    `Kg`. This method assumes that L is a single band and will return estimated
    Ad (albedo index) values for that single band.

    Parameters
    ----------
    z : array-like
        Depth of water column.
    L : array-like
        Irradiance reflectance immediately below the water surface or, if you
        want to ignore units, atmospheric correction, and whatnot, just
        radiance values. This is a single band.
    Rinf : float
        Irradiance reflectance of an optically deep water column.
    Kg : float
        A 2 way effective attenuation coefficient of the water. Really
        equivalent to attenuation coefficient (K) times geometric factor (g).

    Returns
    -------
    Ad : float or array-like of same size as `z`.
        Irradiance reflectance (albedo) of the bottom.
    """
    Ad = (L - Rinf + Rinf * np.exp(-1*Kg*z)) / np.exp(-1*Kg*z)
    return Ad

def estAd(z,L,Rinf,Kg):
    """
    Estimate the albedo `Ad` for radiance `L` at depth `z` assuming `Rinf` and
    `Kg`.

    Parameters
    ----------
    z : array-like
        Depth of water column.
    L : array-like
        Irradiance reflectance immediately below the water surface or, if you
        want to ignore units, atmospheric correction, and whatnot, just
        radiance values. Shape: (rows, columns, bands)
    Rinf : float
        Irradiance reflectance of an optically deep water column.
    Kg : float
        A 2 way effective attenuation coefficient of the water. Really
        equivalent to attenuation coefficient (K) times geometric factor (g).

    Returns
    -------
    Ad : float or array-like of same shape as `L`.
        Irradiance reflectance (albedo) of the bottom in each band.
    """
    nbands = L.shape[-1]
    Rinf = Rinf[:nbands]
    Kg = Kg[:nbands]
    z = np.repeat(np.atleast_3d(z), nbands, axis=2)
    Ad = (L - Rinf + Rinf * np.exp(-1*Kg*z)) / np.exp(-1*Kg*z)
    return Ad

def surface_reflectance_correction(imarr, nir_bands=[6,7]):
    nbands = imarr.shape[-1]
    nbandsvisible = nbands - len(nir_bands)
    nir_mean = imarr[...,nir_bands].mean(2)
    sbtrct = np.repeat(np.atleast_3d(nir_mean), nbandsvisible, axis=2)
    corrected = imarr[...,:nbandsvisible] - sbtrct
    return corrected

def surface_refraction_correction(imarr):
    return imarr * 0.54

## Visualization #############################################################

def albedo_parameter_plots(imarr, darr, params=None, plot_params=True,
                           ylabel='Reflectance', visible_only=True,
                           figsize=(12,7)):
    # from matplotlib import style
    # style.use('ggplot')
    if params is None:
        params = est_curve_params(darr, imarr)
    if visible_only:
        fig, axs = subplots(2, 3, figsize=figsize, sharey=False, sharex=True)
    else:
        fig, axs = subplots(2, 4, figsize=figsize, sharey=False, sharex=True)

    for i, ax in enumerate(axs.ravel()):
        if i >= imarr.shape[-1]:
            # This means I've got more axes than image bands so I'll skip plotting
            continue
        ax.scatter(darr.compressed(),imarr[...,i].compressed(), c='gold', alpha=0.2, edgecolor='none')
        cp = params[i]
        plotz = np.arange(darr.min(), darr.max(), 0.2)
        if plot_params:
            ax.plot(plotz, myR0(plotz, *cp), c='brown')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel(ylabel)
        btxt = "Band{b} $R_\infty = {R:.2f}$\n$A^{{toa}} = {A:.2f}$, $K_g = {Kg:.2f}$ "\
                .format(b=i+1, R=cp[0], A=cp[1], Kg=cp[2])
        ax.set_title(btxt)
    tight_layout()
    return fig

## Testing Methods ###########################################################
# This stuff is just for brewing up test data

def checkerboard(sAd=0.35,kAd=0.2):
    """
    Generate a checkerboard of `sAd` and `kAd` values. The board will be 150 x
    150. with 15 x 15 squares.
    """
    u = 15
    b = np.ones(u**2)
    b.shape = (u,u)
    s = b * sAd
    k = b * kAd
    width = 5     # squares across of a single type
    row1 = np.hstack([s,k]*width)
    row2 = np.hstack([k,s]*width)
    board = np.vstack([row1,row2]*width)
    return board

def zGen(errFactor,n=200,zmin=0.5,zmax=20.0):
    """
    Generate an array of depths with measurement error.
    """
    z = np.linspace(zmin,zmax,n)
    # noise increases with depth
    noise = (z * errFactor/100.0) * np.random.normal(size=n)
    return z + noise

def depthboard(zmin=0.5,zmax=20.0,errFactor=0.0):
    r = zGen(errFactor,150,zmin,zmax)
    for i in range(149): # already made one row
        r = np.vstack((r,zGen(errFactor,150,zmin,zmax)))
    return r

def radiance_checkerboard(sAd=0.35,kAd=0.2,Rinf=0.25,Kg=0.16,satErr=0.005):
    Ad = checkerboard(sAd=sAd,kAd=kAd)
    z = depthboard()
    return myR0(z,Rinf,Ad,Kg) + satErr * np.random.normal(size=150**2).reshape(150,150)
