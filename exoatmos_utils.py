# -*- coding: iso-8859-1 -*-
"""
    Created on Sep 2 2020
    
    Description: utilities for the detection of exoplanet atmospheres
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,fsolve
from scipy import interpolate
from scipy import constants

from sklearn.decomposition import PCA

import exoatmoslib.models_lib as models_lib
from scipy import integrate

from astropy.modeling import models
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

from copy import deepcopy
import exoplanetlib

#################################################################################################
def calculate_template(flux_arr, wl=[], mask=[], fit=False, fit_type="constant", median=True, subtract=False, sub_flux_base=1.0, verbose=False, plot=False, pfilename=""):
    """
        Compute the mean/median template spectrum along the time axis and divide/subtract
        each exposure by the mean/median
        
        Inputs:
        - flux_arr: 2D flux matrix (N_exposures, N_wavelengths)
        - wl: 1D wavelength vector (N_wavelengths)
        - mask: define spectra that will be used to build template, use all if =None
        - fit: boolean to fit median spectrum to each observation before normalizing it
        - fit_type: string to define type of fitting. Available: constant, scale, quadratic
        - median: boolean to calculate median instead of mean
        - subtract: boolean to subtract instead of dividing out spectra by the mean/median

        Outputs:
        - loc: python dict containing all products
    """
    
    loc = {}

    loc["mask"] = mask
    loc["fit"] = fit
    loc["fit_type"] = fit_type
    loc["median"] = median
    loc["subtract"] = subtract
    loc["pfilename"] = pfilename

    if mask == []:
        # Compute template by combining all spectra along the time axis
        if median :
            # median combine
            flux_template = np.nanmedian(flux_arr,axis=0)
        else :
            # mean combine
            flux_template = np.nanmean(flux_arr,axis=0)
            #flux_template = np.average(flux_arr, axis=0, weights=weights)
    else:
        if verbose :
            print("Calculating template from {0} out of {1} input spectra".format(len(flux_arr[mask]), len(flux_arr)))
        if median :
            # median combine
            flux_template = np.nanmedian(flux_arr[mask],axis=0)
        else :
            # mean combine
            flux_template = np.nanmean(flux_arr[mask],axis=0)
            #flux_template = np.average(flux_arr[mask],axis=0, weights=weights)

    if fit :
        shift_arr = []
        flux_calib = []
        flux_fit = []
        
        if fit_type == "constant" :
            def flux_model (wls, shift):
                outmodel = flux_template + shift
                return outmodel

        elif fit_type == "scale" :
            scale_arr = []
            def flux_model (wls, shift, scale):
                outmodel = scale * flux_template + shift
                return outmodel
        
        elif fit_type == "quadratic" :
            quadratic_arr, scale_arr = [],[]
            def flux_model (wls, shift, scale, quadratic):
                outmodel = quadratic * (wls**2) + scale * (flux_template - 1.0) + 1.0 + shift
                return outmodel
        else :
            print("ERROR: fit type {0} not supported, exiting ...".format(fit_type))
            exit()
    
        for i in range(len(flux_arr)):
            
            if fit_type == "constant" :
                guess = [0.0001]
            elif fit_type == "scale" :
                guess = [0.0001, 1.001]
            elif fit_type == "quadratic" :
                guess = [0.0001, 1.001, 0.000001]

            nanmask = ~np.isnan(flux_arr[i])
            
            if len(flux_arr[i][nanmask]) > 0 :
                pfit, pcov = curve_fit(flux_model, wl[nanmask], flux_arr[i][nanmask], p0=guess)
            else :
                if fit_type == "constant" :
                    pfit = [0.]
                elif fit_type == "scale" :
                    pfit = [0.,1.]
                elif fit_type == "quadratic" :
                    pfit = [0.,1.,0.]

            flux_template_fit = flux_model(wl, *pfit)

            shift_arr.append(pfit[0])
            if fit_type == "scale" :
                scale_arr.append(pfit[1])
            elif fit_type == "quadratic" :
                scale_arr.append(pfit[1])
                quadratic_arr.append(pfit[2])

            flux_fit.append(flux_template_fit)
            if fit_type == "constant" :
                flux_calib_loc = flux_arr[i] - pfit[0]
            elif fit_type == "scale" :
                flux_calib_loc = (flux_arr[i] - pfit[0]) / pfit[1]
            elif fit_type == "quadratic" :
                flux_calib_loc = (flux_arr[i] - pfit[2] * (wl[nanmask]**2) - pfit[0] - 1.0) / pfit[1] + 1.0

            flux_calib.append(flux_calib_loc)

        loc["shift"] = np.array(shift_arr, dtype=float)
        if fit_type == "scale" :
            loc["scale"] = np.array(scale_arr, dtype=float)
        elif fit_type == "quadratic" :
            loc["scale"] = np.array(scale_arr, dtype=float)
            loc["quadratic"] = np.array(quadratic_arr, dtype=float)

        flux_calib = np.array(flux_calib, dtype=float)
        flux_fit = np.array(flux_fit, dtype=float)

        if mask == []:
            # Compute median on all spectra along the time axis
            if median :
                flux_template_new = np.nanmedian(flux_calib,axis=0)
            else :
                flux_template_new = np.nanmean(flux_calib,axis=0)
                #flux_template_new = np.average(flux_calib,axis=0, weights=weights)
        else:
            # Compute median on in-mask spectra only
            if median :
                flux_template_new = np.nanmedian(flux_calib[mask],axis=0)
            else :
                flux_template_new = np.nanmean(flux_calib[mask],axis=0)
                #flux_template_new = np.average(flux_calib[mask],axis=0, weights=weights)


        flux_template = flux_template_new
        if subtract :
            flux_arr_sub = flux_calib - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_calib / flux_template

        residuals = flux_calib - flux_template
        flux_template_medsig = np.median(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_calib
    else :
        # Divide or subtract each ccf by ccf_med
        if subtract :
            flux_arr_sub = flux_arr - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_arr / flux_template

        residuals = flux_arr - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_arr

    loc["flux"] = flux_template
    loc["fluxerr"] = flux_template_medsig
    loc["wl"] = wl
    loc["flux_arr_sub"] = flux_arr_sub
    loc["flux_residuals"] = residuals
    loc["snr"] = flux_arr / flux_template_medsig

    loc["fluxerr_model"] = models_lib.fit_continuum(wl, flux_template_medsig, function='polynomial', order=6, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux error", plot_fit=False,silent=True)

    if plot :
        plot_template_products(loc, pfilename=pfilename)
    
    return loc


def sigma_clip_remove_bad_columns(template, nsig=2.0, plot=False) :
    
    template_sigma = np.nanstd(np.abs(template["flux_residuals"]),axis=0)

    good_wl_channels = template_sigma < nsig * template["fluxerr_model"]
    
    clean_template = {}
    clean_template["mask"] = template["mask"]
    clean_template["fit"] = template["fit"]
    clean_template["fit_type"] = template["fit_type"]
    clean_template["median"] = template["median"]
    clean_template["subtract"] = template["subtract"]
    clean_template["pfilename"] = template["pfilename"]
    
    flux_arr, flux_arr_sub, flux_residuals, snr = [],[],[],[]
    for i in range(len(template["flux_residuals"])) :
        flux_arr.append(template["flux_arr"][i][good_wl_channels])
        flux_arr_sub.append(template["flux_arr_sub"][i][good_wl_channels])
        flux_residuals.append(template["flux_residuals"][i][good_wl_channels])
        snr.append(template["snr"][i][good_wl_channels])
        if plot :
            plt.plot(template["wl"][good_wl_channels], template["flux_residuals"][i][good_wl_channels], '.', color='g')
            plt.plot(template["wl"][~good_wl_channels], template["flux_residuals"][i][~good_wl_channels], '.', color='r')

    clean_template["flux_arr"] = np.array(flux_arr, dtype=float)
    clean_template["flux_arr_sub"] = np.array(flux_arr_sub, dtype=float)
    clean_template["flux_residuals"] = np.array(flux_residuals, dtype=float)
    clean_template["snr"] = np.array(snr, dtype=float)

    clean_template["fluxerr_model"] =  template["fluxerr_model"][good_wl_channels]
    clean_template["flux"] = template["flux"][good_wl_channels]
    clean_template["fluxerr"] = template["fluxerr"][good_wl_channels]
    clean_template["wl"] = template["wl"][good_wl_channels]

    if plot :
        plt.plot(template["wl"], nsig * template["fluxerr_model"], '-')
        plt.show()

    return clean_template


def detrend_airmass(template, airmass, snr, cont_value=1.0, fluxkey = "flux_arr_sub",  plot=False) :
    
    flux_arr = template[fluxkey]
    wl = template["wl"]
    
    detrended_flux_arr = linear_detrend(flux_arr, wl, airmass, snr, cont_value = cont_value, deg=2, log=False, plot=plot)

    template[fluxkey] = detrended_flux_arr
    
    return template


def linear_detrend(flux, wl, airmass, snr, cont_value = 1.0, deg=2, log=False, plot=False):
    
    """
        Detrend normalized spectra with airmass to remove residuals of tellurics not subracted
        when dividing by template spectrum. Use least square estimator (LSE) to estimate the
        components of a linear (or log) model of airmass
        
        Inputs:
        - flux:  2D matrix of normalised spectra (N_exposures, len(W))
        - wl:    1D wavelength vector
        - airmass: 1D airmass vector
        - deg:    degree of the linear model (e.g. I(t) = a0 + a1*airmass + ... + an*airmass^(n))
        - log:    if 1 fit log(I_norm) instead of I(t)
        - plot:   if true, plot the components removed with this model
        
        Outputs:
        - I_m_tot: Sequence of spectra without the airmass detrended component
    """
    if log:
        flux_tmp = np.log(flux)
    else:
        flux_tmp = flux - cont_value

    ### Covariance matrix of the noise from DRS SNRs
    COV_inv = np.diag(snr**(2))
    
    ### Apply least-square estimator
    X = []
    X.append(np.ones(len(flux_tmp)))
    for k in range(deg):
        X.append(airmass ** (k+1))
    X = np.array(X,dtype=float).T
    A = np.dot(X.T,np.dot(COV_inv,X))
    b = np.dot(X.T,np.dot(COV_inv,flux_tmp))

    flux_best  = np.dot(np.linalg.inv(A),b)

    ### Plot each component estimated with LSE
    if plot:
        fig = plt.figure()
        ax  = plt.subplot(111)
        c  = 0
        col = ["red","green","magenta","cyan","blue","black","yellow"]
        for ii in flux_best:
            lab = "Order " + str(c)
            alp = 1 - c/len(flux_best)
            plt.plot(wl, ii, label=lab, color=col[c], zorder=c+1, alpha=alp)
            c += 1
            if c == 7: break
        plt.legend()
        plt.title("Components removed - airmass detrending")
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel(r"Residuals removed")
        plt.show()

    if log:
        flux_m_tot  = flux_tmp - np.dot(X, flux_best)
        return np.exp(flux_m_tot)
    else:
        flux_m_tot  = flux_tmp + cont_value - np.dot(X, flux_best)
        return flux_m_tot
    #######################################################################


def sigma_clip(template, nsig=3.0, interpolate=False, replace_by_model=True, sub_flux_base=1.0, plot=False) :
    
    out_flux_arr = np.full_like(template["flux_arr"], np.nan)
    out_flux_arr_sub = np.full_like(template["flux_arr_sub"], np.nan)

    for i in range(len(template["flux_arr"])) :
        sigclipmask = np.abs(template["flux_residuals"][i]) > (nsig * template["fluxerr_model"])
        if plot :
            plt.plot(template["wl"], template["flux_residuals"][i], alpha=0.3)
            if len(template["flux_residuals"][i][sigclipmask]) :
                plt.plot(template["wl"][sigclipmask], template["flux_residuals"][i][sigclipmask], "bo")
    
        # set good values first
        out_flux_arr[i][~sigclipmask] = template["flux_arr"][i][~sigclipmask]
        out_flux_arr_sub[i][~sigclipmask] = template["flux_arr_sub"][i][~sigclipmask]
    
        # now decide what to do with outliers
        if interpolate :
            if i > 0 and i < len(template["flux_arr"]) - 1 :
                out_flux_arr[i][sigclipmask] = (template["flux_arr"][i-1][sigclipmask] + template["flux_arr"][i+1][sigclipmask]) / 2.
                out_flux_arr_sub[i][sigclipmask] = (template["flux_arr_sub"][i-1][sigclipmask] + template["flux_arr_sub"][i+1][sigclipmask]) / 2.
            elif i == 0 :
                out_flux_arr[i][sigclipmask] = template["flux_arr"][i+1][sigclipmask]
                out_flux_arr_sub[i][sigclipmask] = template["flux_arr_sub"][i+1][sigclipmask]
            elif i == len(template["flux_arr"]) - 1 :
                out_flux_arr[i][sigclipmask] = template["flux_arr"][i-1][sigclipmask]
                out_flux_arr_sub[i][sigclipmask] = template["flux_arr_sub"][i-1][sigclipmask]
        
        if replace_by_model :
            out_flux_arr[i][sigclipmask] = template["flux"][sigclipmask]
            out_flux_arr_sub[i][sigclipmask] = sub_flux_base

        #if plot :
        #    plt.plot(template["wl"][sigclipmask],out_flux_arr[i][sigclipmask],'b.')

    if plot :
        plt.plot(template["wl"], nsig * template["fluxerr_model"], 'r--', lw=2)
        plt.plot(template["wl"], -nsig * template["fluxerr_model"], 'r--', lw=2)
        plt.show()
    
    template["flux_arr"] = out_flux_arr
    template["flux_arr_sub"] = out_flux_arr_sub

    return template


def apply_pca(template, N_comp=3) :

    mean = np.nanmean(template["flux_arr_sub"])
        
    # subtract mean spectrum has zero mean, which is necessary for good PCA
    template["flux_arr_sub"] -= mean

    flux_pca, flux_del, e_val = pca_filtering(template["flux_arr_sub"], N_comp=3)
    
    template["flux_arr_sub"] = flux_pca

    return template


def pca_filtering(flux_arr, N_comp=3):
    
    """
    Apply principal component analysis (pca) to remove the first N_comp components from flux_arr
    
    Inputs:
    - flux_arr: 2D array of spectra on which we apply PCA (np.array (N_exp, N_wavelengths))
    - N_comp: Number of components to remove (int)
    
    Outputs:
    - flux_pca: Cleaned matrix (after the first N_comp components are removed - np.array, same shape as flux_arr)
    - flux_del: list of components removed with pca (each component -> np.array, shape of flux_arr)
    - e_val: vector of all eigenvalues computed with PCA

    """

    ### Number of components
    NC = N_comp
    M  = flux_arr
    N  = len(M[0,:])
    K  = len(M[:,0])
    
    ### Apply PCA assuming centered matrix
    pca    = PCA(n_components=K)         ### Number of phases in our case
    M_proj = pca.fit_transform(M)        ### Project in the basis of eigenvectors
    comp   = pca.components_             ### All components of the PCA
    e_val  = pca.singular_values_        ### Eigenvalues of the PCA
    
    comp_r = np.array(comp,dtype=float)  ### Init components to reconstruct matrix
    ### Remove the first N_comp components
    for k in range(NC):
        comp_r[k,:] = np.zeros(N)
    
    ### Store the removed components
    flux_del = []
    for k in range(NC):
        comp_del = np.zeros(np.shape(comp))
        comp_del[k,:] = comp[k,:]
        flux_del.append(np.dot(M_proj,comp_del))
    
    ### Project matrix back into init basis using without the first N_comp components
    M_fin = np.dot(M_proj,comp_r)
    flux_pca = M_fin
    flux_del = np.array(flux_del,dtype=float)
    e_val = np.array(e_val,dtype=float)
    
    return flux_pca, flux_del, e_val


def plot_2d(x, y, z, model=[], LIM=None, LAB=None, z_lim=None, transit=None, title="", pfilename="", cmap="gist_heat"):
    """
        Use pcolor to display 2D map of sequence of spectra
    
    Inputs:
    - x:        x array of the 2D map (if x is 1D vector, then meshgrid; else: creation of Y)
    - y:        y 1D vector of the map
    - z:        2D array (sequence of spectra; shape: (len(x),len(y)))
    - LIM:      list containing: [[lim_inf(x),lim_sup(x)],[lim_inf(y),lim_sup(y)],[lim_inf(z),lim_sup(z)]]
    - LAB:      list containing: [label(x),label(y),label(z)] - label(z) -> colorbar
    - title:    title of the map
    - **kwargs: **kwargs of the matplolib function pcolor
    
    Outputs:
    - Display 2D map of the sequence of spectra
    
    """
    
    if len(np.shape(x))==1:
        X,Y  = np.meshgrid(x,y)
    else:
        X = x
        Y = []
        for n in range(len(x)):
            Y.append(y[n] * np.ones(len(x[n])))
        Y = np.array(Y,dtype=float)
    Z = z

    if LIM == None :
        x_lim = [np.min(X),np.max(X)] #Limits of x axis
        y_lim = [np.min(Y),np.max(Y)] #Limits of y axis
        if z_lim == None :
            z_lim = [np.min(Z),np.max(Z)]
        LIM   = [x_lim,y_lim,z_lim]

    if LAB == None :
        ### Labels of the map
        x_lab = r"$Wavelength$ [nm]"   #Wavelength axis
        y_lab = r"Time [BJD]"         #Time axis
        z_lab = r"Flux"     #Intensity (exposures)
        LAB   = [x_lab,y_lab,z_lab]

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,7)
    ax = plt.subplot(111)

    if transit :
        ax.hlines(y=transit["tini"],xmin=LIM[0][0],xmax=LIM[0][1],ls=':',color='k', lw=2)
        ax.hlines(y=transit["tcen"],xmin=LIM[0][0],xmax=LIM[0][1],ls='--',color='k', lw=2)
        ax.hlines(y=transit["tend"],xmin=LIM[0][0],xmax=LIM[0][1],ls=':',color='k', lw=2)

    if len(model) :
        #print("Input model:", model)
        ax.plot(model, Y, ls='--',color='k', lw=2)

    cc = ax.pcolor(X, Y, Z, vmin=LIM[2][0], vmax=LIM[2][1], cmap=cmap)
    cb = plt.colorbar(cc,ax=ax)
    
    ax.set_xlim(LIM[0][0],LIM[0][1])
    ax.set_ylim(LIM[1][0],LIM[1][1])
    
    ax.set_xlabel(LAB[0])
    ax.set_ylabel(LAB[1],labelpad=15)
    cb.set_label(LAB[2],rotation=270,labelpad=30)

    ax.set_title(title,pad=35)

    if pfilename=="" :
        plt.show()
    else :
        plt.savefig(pfilename, format='png')
    plt.clf()
    plt.close()


def plot_template_products(template, pfilename) :

    mask = template['mask']
    wl = template["wl"]
    
    for i in range(len(template["flux_arr"][mask])) :
        
        flux = template["flux_arr"][mask][i]
        resids = template["flux_residuals"][mask][i]
        
        if i == len(template["flux_arr"][mask]) - 1 :
            plt.plot(wl, flux,"-", color='#1f77b4', lw=0.6, alpha=0.6, label="In-transit data")
            plt.plot(wl, resids,"-", color='#1f77b4', lw=0.6, alpha=0.6, label="In-transit residuals")
        else :
            plt.plot(wl, flux, "-", color='#1f77b4', lw=0.6, alpha=0.6)
            plt.plot(wl, resids, "-", color='#1f77b4', lw=0.6, alpha=0.6)
        
    for i in range(len(template["flux_arr"][~mask])) :
        
        flux = template["flux_arr"][~mask][i]
        resids = template["flux_residuals"][~mask][i]

        if i == len(template["flux_arr"][~mask]) - 1 :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="Out-transit data")
            plt.plot(wl, resids,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="Out-transit residuals")
        else :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6)
            plt.plot(wl, resids,"-", color='#ff7f0e', lw=0.6, alpha=0.6)

    plt.plot(template["wl"], template["flux"],"-", color="red", lw=2, label="Template spectrum")
        
    sig_clip = 3.0
    plt.plot(template["wl"], sig_clip * template["fluxerr"],"--", color="olive", lw=0.8)
    plt.plot(template["wl"], sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8)

    plt.plot(template["wl"],-sig_clip * template["fluxerr"],"--", color="olive", lw=0.8, label="{0:.0f} x Sigma (MAD)".format(sig_clip))
    plt.plot(template["wl"],-sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8)

    plt.legend()
    plt.xlabel(r"$\lambda$ [nm]")
    plt.ylabel(r"Flux")
    if pfilename != "" :
        plt.savefig(pfilename, format='png')
    else :
        plt.show()
    plt.clf()
    plt.close()


def reduce_spectra(spectra, transit, mask_type='out_transit', fit_type="constant", nsig_clip=0.0, combine_by_median=False, airmass_detrend=False, pca_cleaning=False, subtract=True, verbose=False, plot=False, output="") :
    
    signals, ref_snrs, noises,  orders = [], [], [], []
    rel_noises = []
    snrs, snrs_err = [], []
    template = []

    if subtract :
        sub_flux_base = 0.0
    else :
        sub_flux_base = 1.0
    
    for order in range(49) :
        
        # 1st pass to build template for each order and subtract out all spectra by template
        order_template = calculate_template(spectra["flux"][order], wl=spectra["wl"][order], mask=transit[mask_type], fit=True, fit_type=fit_type, median=combine_by_median, subtract=True, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # Recover fluxes already shifted and re-scaled to match the template
        fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base

        # airmass detrending
        if airmass_detrend :
            # 2nd pass to build template for each order and subtract out all spectra by template
            order_template = calculate_template(fluxes, wl=spectra["wl"][order], mask=transit[mask_type], fit=True, fit_type=fit_type, median=combine_by_median, subtract=True, sub_flux_base=0., verbose=False, plot=False)
            
            order_template = detrend_airmass(order_template, spectra["airmass"], spectra["snr"][order], cont_value=sub_flux_base, fluxkey="flux_arr_sub", plot=False)
        
            # Recover fluxes already shifted and re-scaled to match the template
            fluxes = order_template["flux_arr_sub"] + order_template["flux"]
        

        # apply PCA cleaning
        if pca_cleaning :
        # 3rd pass to build template after airmass detrending
            order_template = calculate_template(fluxes, wl=spectra["wl"][order], mask=transit[mask_type], fit=True, fit_type=fit_type, median=combine_by_median, subtract=True, sub_flux_base=0., verbose=False, plot=False)
            
            order_template = apply_pca(order_template, N_comp=3)

            # Recover fluxes already shifted and re-scaled to match the template
            fluxes = order_template["flux_arr_sub"] + order_template["flux"]


        # 3rd pass to build template after airmass detrending
        order_template = calculate_template(fluxes, wl=spectra["wl"][order], mask=transit[mask_type], fit=True, fit_type=fit_type, median=combine_by_median, subtract=subtract, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # apply sigma-clip using template and median dispersion in time as clipping criteria
        # bad values can either be replaced by the template values, by interpolated values or by NaNs
        if nsig_clip > 0 :
            order_template = sigma_clip(order_template, nsig=nsig_clip, interpolate=False, replace_by_model=False, sub_flux_base=sub_flux_base, plot=False)
            
            order_template = sigma_clip_remove_bad_columns(order_template, nsig=2., plot=False)

        # Recover fluxes already shifted and re-scaled to match the template
        if subtract :
            fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base
        else:
            fluxes = order_template["flux_arr_sub"] * order_template["flux"]


        # Calculate a final template combined by the mean and divided out instead of subtracted
        order_template = calculate_template(fluxes, wl=spectra["wl"][order], mask=transit[mask_type], fit=True, fit_type=fit_type, median=False, subtract=subtract, sub_flux_base=1.0, verbose=False, plot=False)

        # save number of spectra in the time series
        order_template['NSPECTRA'] = spectra['NSPECTRA']

        mean_signal, noise = [], []
        rel_noise = []
        for i in range(spectra['NSPECTRA']) :
            mean_signal.append(np.mean(spectra["flux"][order][i]))
            noise.append(np.std(order_template["flux_residuals"][i]))
            rel_noise.append(np.std(order_template["flux_arr_sub"][i]))

        mean_signal, noise = np.array(mean_signal), np.array(noise)
        rel_noise = np.array(rel_noise)
        
        m_signal = np.nanmean(mean_signal)
        m_ref_snr = np.nanmean(spectra["snr"][order])
        m_noise = np.nanmean(noise)
        m_snr = np.nanmean(mean_signal/noise)
        sig_snr = np.std(mean_signal/noise)
        m_rel_noise = np.nanmedian(rel_noise)

        if verbose :
            print("Order {0}: ref_snr={1:.0f} snr={2:.0f}+-{3:.0f} sigma={4:.2f}%".format(order, m_ref_snr, m_snr, sig_snr,m_rel_noise*100.))

        signals.append(m_signal)
        noises.append(m_noise)
        ref_snrs.append(m_ref_snr)
        snrs.append(m_snr)
        snrs_err.append(sig_snr)
        orders.append(order)

        order_template["mean_signal"] = mean_signal
        order_template["ref_snr"] = spectra["snr"][order]
        order_template["noise"] = noise
        order_template["mean_snr"] = mean_signal/noise
        order_template["rel_noise"] = m_rel_noise

        template.append(order_template)

    signals = np.array(signals)
    ref_snrs = np.array(ref_snrs)
    noises = np.array(noises)
    snrs, snrs_err = np.array(snrs), np.array(snrs_err)
    orders = np.array(orders)

    if plot :
        plt.errorbar(orders, snrs, yerr=snrs_err, fmt='o', color='k', label="Measured noise")
        plt.plot(orders, snrs, 'k-', lw=0.5)
        plt.plot(orders, ref_snrs, '--', label="Photon noise")
        plt.xlabel(r"Spectral order")
        plt.ylabel(r"Signal-to-noise ratio (SNR)")
        plt.legend()
        plt.show()

    if output != "":
        np.save(output, template)

    return template


def plot_2Dtime_series(template, bjd, wl0=1083.0, wlf=1083.8, transit=None) :

    wl = np.array([])
    sigma = np.array([])
    wlmask = []
    for order in range(len(template)) :
        order_template = template[order]
        
        temp_wl = order_template['wl']
        temp_wlmask = temp_wl > wl0
        temp_wlmask &= temp_wl < wlf
        
        wlmask.append(temp_wlmask)
        wl = np.append(temp_wl[temp_wlmask], wl)
        if len(temp_wl[temp_wlmask]) > 0 :
            sigma = np.append(order_template["rel_noise"],sigma)
    
    mediansig = np.median(sigma)

    wlsorted = np.argsort(wl)

    fluxes = []
    for i in range(len(bjd)) :
        flux = np.array([])
        for order in range(len(template)) :
            order_template = template[order]
            temp_flux = order_template['flux_arr_sub'][i]
            flux = np.append(temp_flux[wlmask[order]], flux)
        fluxes.append(flux[wlsorted])
    fluxes = np.array(fluxes, dtype=float)

    plot_2d(wl[wlsorted], bjd, fluxes, z_lim = [1.-2*mediansig,1.+2*mediansig], transit=transit, cmap="coolwarm")

    #plot_2d(wl, bjd, fluxes, z_lim = [0.98,1.02], cmap="gist_heat")



def retrieve_exoatmos_model(model_source, planet, species="H2O", abundance=-3.4, model_baseline=0., interp1D_kind='linear', convolve_to_resolution=0., scale=1.0, verbose=False, plot=False) :
    
    model = {}
    
    if model_source.endswith(".json") :
        db = model_source
        if verbose:
            print("Getting interpolated model from library database:",db)
        # get interpolated model from library
        interp_model = models_lib.get_interpolated_model(db, T_EQU=planet['Teq'].value, AB=abundance, R_PL=planet['params'].r.value, M_PL=planet['params'].mass.value, species=species, return_wl=True, return_emission=True)
    
        if verbose :
            print("Interpolated models:", interp_model['T_EQU_low_path'], " and ", interp_model['T_EQU_upp_path'])
    elif model_source.endswith(".fits")  :
        model_filename = model_source
        if verbose:
            print("Reading input model from file:",model_filename)
        interp_model = models_lib.load_spectrum_from_fits(model_filename)
    else :
        print("ERROR: model source file extension not supported",model_source)
        exit()

    # save wavelength vector
    #model['wl'] = interp_model['wl']
    model['min_wl'] = interp_model['wl'][0]
    model['max_wl'] = interp_model['wl'][-1]

    if 'transmission' in interp_model.keys() :
        jupiter_radius_rsun = 0.10049
        # Calculate minimum disk area, i.e., at maximum transmission
        planet_disk_area = planet['params'].r.value ** 2
        star_disk_area = (planet['params'].rstar.value / jupiter_radius_rsun)**2
        atmos_ring_area = (interp_model['transmission'] ** 2) - planet_disk_area
        
        norm_transm = 1.0 - scale * atmos_ring_area / star_disk_area
        
        # measure continuum
        model_cont, model_xbin, model_ybin = models_lib.continuum(interp_model['wl'], norm_transm, binsize=10000, overlap=1000, window=3, mode="max", use_linear_fit=True,telluric_bands=[[1323.,1565.]])
        
        # normalize transmission by continuum
        model_transmission = norm_transm / model_cont - 1.0 + model_baseline
        
        #plt.plot(interp_model['wl'], model_transmission, label='Atmosphere model')
        
        if convolve_to_resolution :
            # calculate pixel size in model, assuming lambda is evenly-spaced in velocity space
            mean_lambda = (interp_model['wl'][1] + interp_model['wl'][0])/2
            diff_lambda = np.abs(interp_model['wl'][1] - interp_model['wl'][0])
            model_pixel_size = mean_lambda / diff_lambda
            kernel_size = int(np.round(model_pixel_size / convolve_to_resolution))
            gauss_kernel = Gaussian1DKernel(kernel_size)
            #box_kernel = Box1DKernel(kernel_size)
            model_transmission = convolve(model_transmission, gauss_kernel)
        
        #plt.plot(interp_model['wl'], model_transmission, label='Atmosphere model')
        #plt.show()
        
        # save an interpolation model to allow evaluation of model at any wavelength
        model['transmission'] = interpolate.interp1d(interp_model['wl'],model_transmission,kind=interp1D_kind)

    if 'emission' in interp_model.keys() :
        # Emission model:
        emission = interp_model['emission']
        
        # measure continuum
        emi_cont, emi_xbin, emi_ybin = models_lib.continuum(interp_model['wl'], emission, binsize=3000, overlap=500, window=3, mode="max", use_linear_fit=True,telluric_bands=[[1323.,1565.]])

        # normalize emission by continuum
        model_emission = emission / emi_cont - 1.0 + model_baseline

        if convolve_to_resolution :
            emission_spectrum = {}
            emission_spectrum['wl'] = interp_model['wl']
            emission_spectrum['flux'] = model_emission
            emission_spectrum['fluxerr'] = np.zeros_like(model_emission)
            convolved_emission = models_lib.convolve_spectrum(emission_spectrum, convolve_to_resolution)
            model_emission = convolved_emission['flux']

        model['emission'] = interpolate.interp1d(interp_model['wl'],model_emission,kind=interp1D_kind)

    if plot :
        if 'transmission' in interp_model.keys() :
            plt.title(r"Transmission model for {0} in {1}".format(species, planet['params'].etdname))
            #plt.plot(interp_model['wl'], norm_transm, label='Atmosphere model')
            #plt.plot(interp_model['wl'], model_cont, "-", label='Continuum')
            #plt.plot(model_xbin, model_ybin , "o")
            plt.plot(interp_model['wl'], model['transmission'](interp_model['wl']), label='Atmosphere model')
            plt.legend()
            plt.xlabel(r"$\lambda$ (nm)")
            plt.ylabel(r"relative transmission")
            plt.show()
        
        if 'emission' in interp_model.keys() :
            plt.title(r"Emission model for {0} in {1}".format(species, planet['params'].etdname))
            #plt.plot(interp_model['wl'], emission, label='Atmosphere model')
            #plt.plot(interp_model['wl'], emi_cont, "-", label='Continuum')
            #plt.plot(emi_xbin, emi_ybin, "o")
            plt.plot(interp_model['wl'], model['emission'](interp_model['wl']), label='Atmosphere model')
            plt.legend()
            plt.xlabel(r"$\lambda$ (nm)")
            plt.ylabel(r"emission")
            plt.show()

    return model


def calculate_doppler_shifted_model(wl_grid, model, rv_shift, nsamples=5, interpolate=False) :

    speed_of_light_in_kps = constants.c / 1000.
    
    shifted_wl = wl_grid / (1.0 + rv_shift / speed_of_light_in_kps)
    
    if interpolate :
        shifted_model = model(shifted_wl)
    else :
        wl_diffs = (shifted_wl[1:] - shifted_wl[:-1]) / 2.

        if len(shifted_wl) :
            wl_diffs = np.append(wl_diffs[-1], wl_diffs)
    
        wl0 = shifted_wl - wl_diffs
        wlf = shifted_wl + wl_diffs
    
        shifted_model = np.full_like(shifted_wl,0.)

        for i in range(len(shifted_wl)) :
            x = np.linspace(wl0[i], wlf[i], nsamples, endpoint=True)
            y = model(x)
            shifted_model[i] = integrate.simps(y,x)

    return shifted_model


def build_exoatmos_model(model, template, spectra, planet, transit, wl0=950, wlf=2500, verbose=False, plot=False) :
    
    loc = {}
    
    bjd = spectra["bjds"]
    berv = spectra["bervs"]
    
    wl, transmission_model = [], []
    masks = []
    
    # calculate planet-to-star relative velocities in the line of sight "z" direction in km/s
    #planet_rvs = np.array(planet['orbit'].get_relative_velocity(bjd)[2].eval() * (696340./(24.*60.*60.)))
    planet_rvs = exoplanetlib.planet_rv(transit['phases'], planet['Kp'])

    for order in range(len(template)) :
        
        order_template = template[order]
        
        order_mask = order_template['wl'] > model['min_wl'] * (1.0 + planet["Kp"]*1000. / constants.c)
        order_mask &= order_template['wl'] > wl0
        order_mask &= order_template['wl'] < model['max_wl'] * (1.0 - planet["Kp"]*1000. / constants.c)
        order_mask &= order_template['wl'] < wlf

        order_wl = order_template['wl'][order_mask]
        
        if len(order_wl) :
            if verbose :
                print("Calculating model for order {0} in range: wl0={1:.2f} wlf={2:.2f}".format(order,order_wl[0],order_wl[-1]))

            masks.append(order_mask)
            wl.append(order_wl)
                
            order_transmission_models = []

            for i in range(len(bjd)) :
                
                # calculate velocity shift caused by the planet orbit and observer's movement (BERV)
                rv_shift = planet_rvs[i] - berv[i]
                #print(i, bjd[i], planet_rvs[i])
                
                if transit['window'][i] :
                    # apply Doppler shift to obtain the model in the planet frame
                    order_model = calculate_doppler_shifted_model(order_wl, model['transmission'], rv_shift, interpolate=True)
                    
                    # multiply model by window transit function
                    order_model *= transit['window'][i]

                else :
                    order_model = np.full_like(order_wl,0.)

                order_transmission_models.append(order_model)
            
            # cast order models into np.array
            order_transmission_models = np.array(order_transmission_models, dtype=float)
            
            if plot :
                print("Plotting transmission model for order {0}".format(order))
                plot_2d(order_wl, bjd, order_transmission_models, transit=transit, cmap="plasma")
                break
            
            transmission_model.append(order_transmission_models)

        else :
            if verbose :
                print("Skipping order {0}, empty arrays".format(order))

            masks.append(np.array([]))
            wl.append(np.array([]))
            transmission_model.append(np.array([]))

    loc['bjds'] = bjd
    loc['bervs'] = berv
    
    loc['wl_masks'] = masks
    loc['wl'] = wl
    
    loc['transmission_model'] = transmission_model

    return loc


def HeI_nIR_triplet_model(model_baseline=1.0, scale=1.0, plot=False) :
    
    loc = {}
    
    def model(wl) :
        #l1,l2,l3 = 1082.909114, 1083.025010, 1083.033977
        l1,l2,l3 = 1083.2057, 1083.3217, 1083.3306
        resolution=70000
        g1 = models.Gaussian1D(scale * 0.1, l1, l1/resolution)
        g2 = models.Gaussian1D(scale * 0.3, l2, l2/resolution)
        g3 = models.Gaussian1D(scale * 0.5, l3, l3/resolution)
        return model_baseline - (g1(wl) + g2(wl) + g3(wl))

    if plot :
        wlplot = np.geomspace(1082., 1084., 1000)
        #l1,l2,l3 = 1082.909114, 1083.025010, 1083.033977
        l1,l2,l3 = 1083.2057, 1083.3217, 1083.3306
        plt.plot(wlplot, model(wlplot))
        plt.vlines((l1, l2, l3), model_baseline-1, model_baseline, color = "r", ls=("--", "--", "--"))
        plt.show()
    
    loc['min_wl'] = 950.
    loc['max_wl'] = 2500.
    
    loc['transmission'] = model
    # save an interpolation model to allow evaluation of model at any wavelength

    return loc



def run_HeI_analysis(template, spectra, planet, transit, model_baseline=1.0, verbose=True, plot=True) :
    
    loc = {}
    
    bjd = spectra["bjds"]
    berv = spectra["bervs"]
    rvs = spectra["rvs"]
    rverrs = spectra["rverrs"]
    
    model = HeI_nIR_triplet_model(model_baseline=model_baseline, plot=False)
    
    sys_rv = planet['params'].gamma.value
    
    # calculate planet velocities in the line of sight "z" direction in km/s
    #planet_rvs = np.array(planet['orbit'].get_relative_velocity(bjd)[2].eval() * (696340./(24.*60.*60.)))
    planet_rvs = exoplanetlib.planet_rv(transit['phases'], planet['Kp'])

    # calculate star velocities in the line of sight "z" direction in km/s
    #star_rvs = np.array(planet['orbit'].get_star_velocity(bjd)[2].eval() * (696340./(24.*60.*60.)))
    star_rvs = exoplanetlib.planet_rv(transit['phases'], planet['Ks'])

    order = 8
    order_template = template[order]
    #wl0, wlf = 1050., 1100. # one can use these values to restrict the range of analysis
    wl0, wlf = 1082., 1085.
    order_mask = order_template['wl'] > wl0
    order_mask &= order_template['wl'] < wlf

    wl = order_template['wl'][order_mask]
    if len(wl) == 0 :
        print("ERROR: Empty length of data array, exiting...")
        exit()
            
    planet_models, star_models = [], []
    observed_star_spectra, observed_planet_spectra = [], []

    plt.plot(wl, order_template['flux'][order_mask], '-')
    plt.plot(wl, np.median(order_template['flux'][order_mask])*calculate_doppler_shifted_model(wl, model['transmission'], sys_rv-np.mean(berv), interpolate=True),'-')
    plt.show()
    exit()

    for i in range(len(bjd)) :
        #planet-to-star_rv = planet_rv - sys_rv - star_to_centre_rv, where sys_rv + star_to_centre_rv = rvs[i]
        planet_rv_shift = planet_rvs[i] - rvs[i] - berv[i]
        
        # star_rv  = rvs[i] = sys_rv + star_to_centre_rv
        star_rv_shift = rvs[i] - berv[i]
        
        print(i, bjd[i], star_rvs[i],  planet_rvs[i], berv[i], rvs[i], rverrs[i], sys_rv)
        
        star_model = calculate_doppler_shifted_model(wl, model['transmission'], star_rv_shift, interpolate=True)
        # Recover fluxes already shifted and re-scaled to match the template
        observed_star = order_template['flux_arr_sub'][i][order_mask] + order_template['flux'][order_mask] - model_baseline

        # Recover fluxes already shifted and re-scaled to match the template
        observed_planet = order_template['flux_arr_sub'][i][order_mask]

        if transit['window'][i] :
        # apply Doppler shift to obtain the model in the planet frame
            planet_model = calculate_doppler_shifted_model(wl, model['transmission'], planet_rv_shift, interpolate=True)
            # multiply model by window transit function
            planet_model *= transit['window'][i]
        else :
            planet_model = np.full_like(wl,model_baseline)
        
        planet_models.append(planet_model)
        star_models.append(star_model)
        observed_star_spectra.append(observed_star)
        observed_planet_spectra.append(observed_planet)

    # cast order models into np.array
    planet_models = np.array(planet_models, dtype=float)
    star_models = np.array(star_models, dtype=float)

    observed_star_spectra = np.array(observed_star_spectra, dtype=float)
    observed_planet_spectra = np.array(observed_planet_spectra, dtype=float)

    if plot :
        print("Plotting planet model for He I triplet")
        plot_2d(wl, bjd, planet_models, transit=transit, cmap="plasma")
        plot_2d(wl, bjd, observed_planet_spectra, transit=transit, cmap="gist_heat")
        #plot_2d(wl, bjd, observed_star_spectra, transit=transit, cmap="plasma")
        #plot_2d(wl, bjd, star_models, transit=transit, cmap="plasma")


    loc['bjds'] = bjd
    loc['bervs'] = berv
    loc['wl'] = wl

    loc['planet_model'] = planet_models
    loc['star_model'] = star_models

    return loc


def compute_ccf(model, template, spectra, transit, planet, v0=-20., vf=20., nvels=40, wl0=950, wlf=2500, use_observed_rvs=False, ref_frame='observer', kp=0, verbose=False, plot=False) :
    
    speed_of_light_in_kps = constants.c / 1000.
    
    loc = {}
    loc['wl0'] = wl0
    loc['wlf'] = wlf
    loc['use_observed_rvs'] = use_observed_rvs
    
    bjd = spectra["bjds"]
    berv = spectra["bervs"]
    if use_observed_rvs :
        rvs = spectra["rvs"]
    else :
        sys_rv = planet['params'].gamma.value
        rvs = np.full_like(bjd, sys_rv)

    if kp == 0 :
        kp = planet['Kp']
        if verbose :
            print("Kp=",kp,"km/s")
    loc['kp'] = kp

    planet_rvs = exoplanetlib.planet_rv(transit['phases'], kp)
    #planet_rvs = np.array(planet['orbit'].get_relative_velocity(bjd)[2].eval() * (696340./(24.*60.*60.)))

    if ref_frame == 'star' :
        ref_frame_rvs = rvs - berv
        model_rvs = planet_rvs
    elif ref_frame == 'planet' :
        ref_frame_rvs = rvs - berv + planet_rvs
        model_rvs = np.full_like(bjd, 0.)
    elif ref_frame == 'observer' :
        ref_frame_rvs = np.full_like(bjd, 0.)
        model_rvs = rvs - berv + planet_rvs
    else :
        print("ERROR: ref_frame {0} not supported, exiting ...".format(ref_frame))
        exit()
    loc['ref_frame'] = ref_frame
    loc['ref_frame_rvs'] = ref_frame_rvs
    loc['model_rvs'] = model_rvs

    if verbose :
        print("min/max BERV: ",np.min(berv), "/", np.max(berv), " km/s")
        print("min/max RVs: ",np.min(rvs), "/", np.max(rvs), " km/s")

    # build velocity array in km/s
    vels = np.linspace(v0,vf,nvels)
    xcorr = []
    
    # First loop over all orders to create masks to get only data at useful wavelength ranges
    masks, valid_orders = [], []

    for order in range(len(template)) :
        order_template = template[order]
        order_wl = order_template['wl']
        
        order_mask = order_template['wl'] > model['min_wl'] * (1.0 + vf / speed_of_light_in_kps)
        order_mask &= order_template['wl'] > wl0
        order_mask &= order_template['wl'] < model['max_wl'] * (1.0 + v0 / speed_of_light_in_kps)
        order_mask &= order_template['wl'] < wlf
        
        masks.append(order_mask)
        
        if len(order_wl[order_mask]) :
            if verbose:
                print("Using order ", order)
            valid_orders.append(order)


    for i in range(len(bjd)) :
        if verbose:
            print("Calculating cross-correlation for spectrum ", i)
    
        corr = np.zeros_like(vels)

        if transit['window'][i] :
            for order in valid_orders :
                order_template = template[order]
            
                order_mask = masks[order]
                order_mask &= ~np.isnan(order_template['flux_arr_sub'][i])
            
                order_wl = order_template['wl'][order_mask]
                observed_spectrum = order_template['flux_arr_sub'][i][order_mask]

                for j in range(len(vels)) :
                    # calculate Doppler shift
                    rv_shift = vels[j] + ref_frame_rvs[i]
                
                    # apply Doppler shift to model
                    order_model = calculate_doppler_shifted_model(order_wl, model['transmission'], rv_shift, interpolate=True)

                    # add cross-correlation between observed spectrum and shifted model
                    loc_corr = np.ma.corrcoef(observed_spectrum,order_model).data[0,1]
                    corr[j] += loc_corr * transit['window'][i]
                        #print(i, order, j, loc_corr)

        #plt.plot(vels,corr)s
        #plt.show()
        xcorr.append(corr)

    xcorr = np.array(xcorr,dtype=float)
    
    # calculate flattened CCF by summing all CCFs along axis
    flat_xcorr = np.nansum(xcorr, axis=0)

    loc['bjd'] = bjd
    loc['vels'] = vels
    loc['xcorr'] = xcorr
    loc['flatxcorr'] = flat_xcorr
    loc['valid_orders'] = valid_orders
    loc['masks'] = masks

    if plot :
        print("plotting CCF...")
        # calculate planet velocities in the line of sight "z" direction in km/s
        
        LAB   = [r"Velocity [km/s]",r"Time [BJD]",r"CCF"]
        plot_2d(vels, bjd, xcorr, model=model_rvs, LAB=LAB, transit=transit, cmap="coolwarm")
    
        imax = np.argmax(loc['flatxcorr'])
        plt.plot(vels, loc['flatxcorr'])
        plt.vlines(vels[imax],np.min(loc['flatxcorr']), np.max(loc['flatxcorr']), color = "r", ls="--")
        plt.xlabel(r"Velocity [km/s]")
        plt.ylabel(r"Cross-correlation")
        plt.show()
    
    return loc


def simulate_data(model, template, spectra, transit, planet, model_baseline = 0.0, scale=1.0, use_observed_rvs=False, ref_frame='observer', kp=0, plot=False, verbose=False) :
    
    bjd = spectra["bjds"]
    berv = spectra["bervs"]
    rvs = spectra["rvs"]

    #planet_rvs = np.array(planet['orbit'].get_relative_velocity(bjd)[2].eval() * (696340./(24.*60.*60.)))
    planet_rvs = exoplanetlib.planet_rv(transit['phases'], planet['Kp'])

    rv_shift = rvs + planet_rvs - berv
    
    simulated_template  = deepcopy(template)
    
    for order in range(len(template)) :
        if verbose :
            print("Simulating data for order {0}/{1} ".format(order, len(template)-1))
        
        order_template = template[order]
        
        order_mask = order_template['wl'] > model['min_wl'] * (1.0 + np.abs(np.max(rv_shift))*1000. / constants.c)
        #order_mask &= order_template['wl'] > wl0
        order_mask &= order_template['wl'] < model['max_wl'] * (1.0 - np.abs(np.max(rv_shift))*1000 / constants.c)
        #order_mask &= order_template['wl'] < wlf
        order_wl = order_template['wl'][order_mask]

        for i in range(len(bjd)) :
                
            # Recover fluxes already shifted and re-scaled to match the template
            observed_noise = order_template['fluxerr'][order_mask]
            
            if transit['window'][i] :
                # apply Doppler shift to obtain the model in the planet frame
                planet_model = calculate_doppler_shifted_model(order_wl, model['transmission'], rv_shift[i], interpolate=True)
                # multiply model by window transit function
                planet_model = (planet_model - model_baseline) * transit['window'][i] * scale + model_baseline
            else :
                planet_model = np.full_like(order_wl, model_baseline)
        
            simulated_noise = np.random.normal(0., observed_noise, np.shape(observed_noise))
            #simulated_order_flux = (simulated_noise + order_template['flux'][order_mask] * planet_model) / order_template['flux'][order_mask]
            simulated_order_flux = simulated_noise + planet_model
            
            simulated_template[order]['flux_arr_sub'][i] = np.full_like(order_template['flux_arr_sub'][i], model_baseline)
            
            simulated_template[order]['flux_arr_sub'][i][order_mask] = simulated_order_flux

    return simulated_template


def compute_ccf_kp_v0_map(model, template, spectra, transit, planet, wl0=950, wlf=2500, kp_range=75., n_kp=100, v0_range=100., n_v0=100, use_observed_rvs=False, plot=False, verbose=False) :

    loc = {}
    
    # Build kp vector
    kp_ini, kp_end = planet['Kp'] - kp_range/2., planet['Kp'] + kp_range/2.
    kp  = np.linspace(kp_ini, kp_end, n_kp)

    # Build v0 vector
    v0_ini, v0_end = - v0_range/2., v0_range/2.
    v0  = np.linspace(v0_ini, v0_end, n_v0)
    
    max_ccf = -1e20
    save_max_ccf = None
    
    out_ccf = []
    for i in range(len(kp)) :
        if verbose :
            print("Calculating CCF for  Kp={0:.1f} km/s {1}/{2}".format(kp[i], i, len(kp)-1))
        
        planet_rvs = exoplanetlib.planet_rv(transit['phases'], kp[i])
        
        ccf = compute_ccf(model, template, spectra, transit, planet, v0=v0_ini, vf=v0_end, nvels=n_v0, wl0=wl0, wlf=wlf, use_observed_rvs=use_observed_rvs, ref_frame='planet', kp=kp[i], verbose=False, plot=False)

        if np.nanmax(ccf['flatxcorr']) > max_ccf :
            max_ccf = np.nanmax(ccf['flatxcorr'])
            save_max_ccf = ccf
                
        out_ccf.append(ccf['flatxcorr'])
    
    out_ccf = np.array(out_ccf, dtype=float)
    
    loc['kp'] = kp
    loc['v0'] = v0
    loc['ccf'] = out_ccf

    if plot :
        print("plotting CCF [Kp x V0] map...")
        
        LAB   = [r"Velocity [km/s]",r"Kp [km/s]",r"CCF"]
        #plot_2d(v0, kp, out_ccf, model=np.full_like(v0,planet['Kp']), LAB=LAB)
        plot_2d(v0, kp, out_ccf, LAB=LAB, cmap="coolwarm")

        # plot maximum CCF :
        plt.title(r"Maximum CCF for Kp={0:.2f} km/s".format(save_max_ccf['kp']))
        imax = np.argmax(save_max_ccf['flatxcorr'])
        plt.plot(save_max_ccf['vels'], save_max_ccf['flatxcorr'])
        plt.vlines(save_max_ccf['vels'][imax],np.min(save_max_ccf['flatxcorr']), np.max(save_max_ccf['flatxcorr']), color = "r", ls="--")
        plt.xlabel(r"Velocity [km/s]")
        plt.ylabel(r"Cross-correlation")
        plt.show()

    return loc
