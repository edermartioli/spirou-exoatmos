# -*- coding: iso-8859-1 -*-
"""
    Created on Jul 06 2020
    
    Description: python library for handling SPIRou data
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy.io import ascii
from scipy import constants

import exoatmos_utils

import os,sys

import time

#--- Load a spirou spectrum from e.fits or t.fits file (which are the default products at CADC)
# This function preserves the spectral order structure
def load_spirou_AB_efits_spectrum(input, nan_pos_filter=True, preprocess=False, apply_BERV_to_preprocess=True, source_rv=0., remove_blaze=True, normalization_in_preprocess=True, normalize_blaze=True) :
    
    # open fits file
    hdu = fits.open(input)
    
    if input.endswith("e.fits") :
        WaveAB = hdu["WaveAB"].data
        FluxAB = hdu["FluxAB"].data
        #BlazeAB = hdu[9].data / np.median(hdu[9].data)
        if normalize_blaze :
            BlazeAB = hdu["BlazeAB"].data / np.nanmean(hdu["BlazeAB"].data)
        else :
            BlazeAB = hdu["BlazeAB"].data
        
        WaveC = hdu["WaveC"].data
        FluxC = hdu["FluxC"].data
        #BlazeC = hdu["BlazeC"].data / np.median(hdu["BlazeC"].data)
        BlazeC = hdu["BlazeC"].data / np.nanmean(hdu["BlazeC"].data)

    elif input.endswith("t.fits") :
        WaveAB = hdu["WaveAB"].data
        FluxAB = hdu["FluxAB"].data
        #BlazeAB = hdu[3].data / np.median(hdu[3].data)
        if normalize_blaze :
            BlazeAB = hdu["BlazeAB"].data / np.nanmean(hdu["BlazeAB"].data)
        else :
            BlazeAB = hdu["BlazeAB"].data
        Recon = hdu["Recon"].data
    else :
        print("ERROR: input file type not recognized")
        exit()

    WaveABout, FluxABout, BlazeABout = [], [], []
    WaveCout, FluxCout, BlazeCout = [], [], []
    Reconout = []
    for i in range(len(WaveAB)) :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(FluxAB[i]))
            # mask negative and zero values
            negmask = np.where(FluxAB[i][nanmask] > 0)

            if len(WaveAB[i][nanmask][negmask]) :
                WaveABout.append(WaveAB[i][nanmask][negmask])
                FluxABout.append(FluxAB[i][nanmask][negmask])
                BlazeABout.append(BlazeAB[i][nanmask][negmask])
                if input.endswith("e.fits") :
                    WaveCout.append(WaveC[i][nanmask][negmask])
                    FluxCout.append(FluxC[i][nanmask][negmask])
                    BlazeCout.append(BlazeC[i][nanmask][negmask])
                elif input.endswith("t.fits") :
                    Reconout.append(Recon[i][nanmask][negmask])
            else :
                WaveABout.append(np.array([]))
                FluxABout.append(np.array([]))
                BlazeABout.append(np.array([]))
                if input.endswith("e.fits") :
                    WaveCout.append(np.array([]))
                    FluxCout.append(np.array([]))
                    BlazeCout.append(np.array([]))
                elif input.endswith("t.fits") :
                    Reconout.append(np.array([]))

        else :
            WaveABout.append(WaveAB[i])
            FluxABout.append(FluxAB[i])
            BlazeABout.append(BlazeAB[i])
            if input.endswith("e.fits") :
                WaveCout.append(WaveC[i])
                FluxCout.append(FluxC[i])
                BlazeCout.append(BlazeC[i])
            elif input.endswith("t.fits") :
                Reconout.append(Recon[i])

    loc = {}
    loc['filename'] = input
    loc['header0'] = hdu[0].header
    loc['header1'] = hdu[1].header

    loc['WaveAB'] = WaveABout
    loc['FluxAB'] = FluxABout
    loc['BlazeAB'] = BlazeABout
    
    if input.endswith("e.fits") :
        loc['WaveC'] = WaveCout
        loc['FluxC'] = FluxCout
        loc['BlazeC'] = BlazeCout
        loc['headerC'] = hdu['FluxC'].header

    elif input.endswith("t.fits") :
        loc['Recon'] = Reconout

    if preprocess :
        # Pre-process spectrum to normalize data, remove nans and zeros, apply BERV correction if requested, etc
        loc = pre_process(loc, apply_BERV=apply_BERV_to_preprocess, source_rv=source_rv, remove_blaze=remove_blaze, normalize=normalization_in_preprocess, nan_pos_filter=nan_pos_filter)

    return loc


def pre_process(spectrum, apply_BERV=True, source_rv=0., remove_blaze=True, normalize=True, nan_pos_filter=True) :
    out_wl, out_flux, out_fluxerr = [], [], []
    if 'Recon' in spectrum.keys() :
        out_recon = []
    if normalize :
        norm_chunk = get_chunk_data(spectrum, 1054., 1058., 6, rv_overscan = 0., source_rv=0.0, apply_BERV=False, cal_fiber=False, remove_blaze=remove_blaze, normalize=False, nan_pos_filter=True, plot=False)
        spectrum["normalization_factor"] = np.median(norm_chunk['flux'])
    
    out_continuum = []
    
    for order in range(len(spectrum['WaveAB'])) :
        if len(spectrum['WaveAB'][order]) :
            wl0, wlf = spectrum['WaveAB'][order][0], spectrum['WaveAB'][order][-1]
            loc = get_chunk_data(spectrum, wl0, wlf, order, rv_overscan = 0., source_rv=source_rv, cal_fiber=False, apply_BERV=apply_BERV, remove_blaze=remove_blaze, normalize=normalize, nan_pos_filter=nan_pos_filter, plot=False)
            out_wl.append(loc['wl'])
            out_flux.append(loc['flux'])
            out_fluxerr.append(loc['fluxerr'])
            if 'Recon' in spectrum.keys() :
                out_recon.append(loc['Recon'])
        else :
            out_wl.append(np.array([]))
            out_flux.append(np.array([]))
            out_fluxerr.append(np.array([]))
            if 'Recon' in spectrum.keys() :
                out_recon.append(np.array([]))

    spectrum['wl'] = out_wl
    spectrum['flux'] = out_flux
    spectrum['fluxerr'] = out_fluxerr
    if 'Recon' in spectrum.keys() :
        spectrum['Recon'] = out_recon

    return spectrum

#### Function to get chunk data #########
def get_chunk_data(spectrum, wl0, wlf, order, rv_overscan = 100.0, source_rv=0.0, apply_BERV=True, cal_fiber=False, remove_blaze=True, normalize=True, nan_pos_filter=True, plot=False) :

    loc = {}
    loc['order'] = order

    loc['filename'] = spectrum['filename']
    loc['wl0'] = wl0
    loc['wlf'] = wlf

    wlc = (wl0 + wlf) / 2.
    
    # set overscan to avoid edge issues
     # in km/s
    loc['rv_overscan'] = rv_overscan

    #rv_overscan = 0.0 # in km/s
    dwl_1 = rv_overscan * wl0 / (constants.c / 1000.)
    dwl_2 = rv_overscan * wlf / (constants.c / 1000.)
    
    # get BERV from header
    if apply_BERV :
        BERV = spectrum['header1']['BERV']
    else :
        BERV = 0.

    # get DETECTOR GAIN and READ NOISE from header
    gain, rdnoise = spectrum['header0']['GAIN'], spectrum['header0']['RDNOISE']

    if cal_fiber :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(spectrum['FluxAB'][order]))
            # mask negative and zero values
            negmask = np.where(spectrum['FluxAB'][order][nanmask] > 0)
            # set calibration fiber flux and wavelength vectors
            if remove_blaze :
                flux = spectrum['FluxC'][order][nanmask][negmask] / spectrum['BlazeC'][order][nanmask][negmask]
                fluxerr = np.sqrt((spectrum['FluxAB'][order][nanmask][negmask] + (rdnoise * rdnoise / gain * gain) ) / spectrum['BlazeAB'][order][nanmask][negmask])
            else :
                flux = spectrum['FluxC'][order][nanmask][negmask]
                fluxerr = np.sqrt(spectrum['FluxAB'][order][nanmask][negmask] + (rdnoise * rdnoise / gain * gain))
            wave = spectrum['WaveC'][order][nanmask][negmask]
            # calculate flux variance
        else :
            # set calibration fiber flux and wavelength vectors
            if remove_blaze :
                flux = spectrum['FluxC'][order] / spectrum['BlazeC'][order]
                fluxerr = np.sqrt((spectrum['FluxAB'][order] + (rdnoise * rdnoise / gain * gain) ) / spectrum['BlazeAB'][order])
            else :
                flux = spectrum['FluxC'][order]
                fluxerr = np.sqrt(spectrum['FluxAB'][order] + (rdnoise * rdnoise / gain * gain))
            wave = spectrum['WaveC'][order]
            # calculate flux variance

    else :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(spectrum['FluxAB'][order]))
            # mask negative and zero values
            negmask = np.where(spectrum['FluxAB'][order][nanmask] > 0)
            # set science fiber flux and wavelength vectors
            if remove_blaze :
                flux = spectrum['FluxAB'][order][nanmask][negmask] / spectrum['BlazeAB'][order][nanmask][negmask]
                fluxerr = np.sqrt(spectrum['FluxAB'][order][nanmask][negmask] + (rdnoise * rdnoise / gain * gain)) / spectrum['BlazeAB'][order][nanmask][negmask]
            else :
                flux = spectrum['FluxAB'][order][nanmask][negmask]
                fluxerr = np.sqrt(spectrum['FluxAB'][order][nanmask][negmask] + (rdnoise * rdnoise / gain * gain))
            # apply BERV correction - Barycentric Earth Radial Velocity (BERV)
            wave = spectrum['WaveAB'][order][nanmask][negmask] * ( 1.0 + (BERV - source_rv) / (constants.c / 1000.) )
            # calculate flux variance
            if 'Recon' in spectrum.keys():
                recon = spectrum['Recon'][order][nanmask][negmask]
        else :
            # set science fiber flux and wavelength vectors
            if remove_blaze :
                flux = spectrum['FluxAB'][order] / spectrum['BlazeAB'][order]
                # calculate flux variance
                fluxerr = np.sqrt(spectrum['FluxAB'][order] + (rdnoise * rdnoise / gain * gain)) / spectrum['BlazeAB'][order]
            else :
                flux = spectrum['FluxAB'][order]
                fluxerr = np.sqrt(spectrum['FluxAB'][order] + (rdnoise * rdnoise / gain * gain))
            # apply BERV correction - Barycentric Earth Radial Velocity (BERV)
            wave = spectrum['WaveAB'][order] * ( 1.0 + (BERV - source_rv) / (constants.c / 1000.) )
            # get telluric absorption spectrum
            if 'Recon' in spectrum.keys():
                recon = spectrum['Recon'][order]
    
    # set wavelength masks
    wlmask = np.where(np.logical_and(wave > wl0 - dwl_1, wave < wlf + dwl_2))

    if len(flux[wlmask]) == 0 :
        loc['wl'] = np.array([])
        loc['flux'] = np.array([])
        loc['fluxerr'] = np.array([])
        if 'Recon' in spectrum.keys():
            loc['Recon'] = np.array([])
        return loc

    # measure continuum and normalize flux if nomalize=True
    if normalize == True:
        # Calculate normalization factor
        normalization_factor = spectrum['normalization_factor']

        loc['normalization_factor'] = normalization_factor

        # normalize flux
        flux = flux / normalization_factor
        fluxerr = fluxerr / normalization_factor

    # mask data
    flux, fluxerr, wave = flux[wlmask], fluxerr[wlmask], wave[wlmask]
    if 'Recon' in spectrum.keys():
        recon = recon[wlmask]

    if plot :
        plt.plot(wave,flux)
        plt.errorbar(wave,flux,yerr=fluxerr,fmt='.')
        if 'Recon' in spectrum.keys():
            plt.plot(wave,flux*recon,'-',linewidth=0.3)
        plt.show()

    loc['order'] = order
    loc['wl'] = wave
    if cal_fiber :
        loc['flux'] = flux / np.max(flux)
        loc['fluxerr'] = fluxerr / np.max(flux)
    else :
        loc['flux'] = flux
        loc['fluxerr'] = fluxerr
        if 'Recon' in spectrum.keys():
            loc['Recon'] = recon

    return loc
##-- end of function

def get_normalization_factor(flux_order, i_max, norm_window=50) :
    
    min_i = i_max - norm_window
    max_i = i_max + norm_window
    if min_i < 0 :
        min_i = 0
        max_i = min_i + 2 * norm_window
    if max_i >= len(flux_order) :
        max_i = len(flux_order) - 1

    # Calculate normalization factor as the median of flux within window around maximum signal
    normalization_factor = np.nanmedian(flux_order[min_i:max_i])

    return normalization_factor


# Functon below defines SPIRou spectral orders, useful wavelength ranges and the NIR bands
def spirou_order_mask():
    order_mask = [[0, 967, 980, 'Y'],
             [1, 977, 994, 'Y'],
             [2, 989, 1005.5,'Y'],
             [3, 1001., 1020,'Y'],
             [4, 1018, 1035,'Y'],
             [5, 1028, 1050,'Y'],
             [6, 1042.3, 1065,'Y'],
             [7, 1055, 1078,'Y'],
             [8, 1071.5, 1096,'Y'],
             [9, 1084.5, 1107,'Y'],
             [10, 1107, 1123,'J'],
             [11, 1123, 1140.2,'J'],
             [12, 1137.25, 1162,'J'],
             [13, 1150, 1178,'J'],
             [14, 1168, 1198,'J'],
             [15, 1186, 1216,'J'],
             [16, 1204, 1235,'J'],
             [17, 1223, 1255,'J'],
             [18, 1243, 1275,'J'],
             [19, 1263, 1297,'J'],
             [20, 1284, 1321,'J'],
             [21, 1306, 1344,'J'],
             [22, 1327.5, 1367.5,'J'],
             [23, 1350.1, 1392,'J'],
             [24, 1374.3, 1415,'J'],
             [25, 1399.7, 1443.6,'H'],
             [26, 1426, 1470.9,'H'],
             [27, 1453.5, 1499,'H'],
             [28, 1482, 1528.6,'H'],
             [29, 1512, 1557.5,'H'],
             [30, 1544, 1591.1,'H'],
             [31, 1576.6, 1623,'H'],
             [32, 1608.5, 1658.9,'H'],
             [33, 1643.5, 1695,'H'],
             [34, 1679.8, 1733,'H'],
             [35, 1718, 1772,'H'],
             [36, 1758, 1813.5,'H'],
             [37, 1800.7, 1856.5,'H'],
             [38, 1843.9, 1902, 'H'],
             [39, 1890, 1949.5,'H'],
             [40, 1938.4, 1999.5, 'H'],
             [41, 1989.5, 2052, 'K'],
             [42, 2043, 2108, 'K'],
             [43, 2100, 2166, 'K'],
             [44, 2160, 2228,'K'],
             [45, 2223.5, 2293.6,'K'],
             [46, 2291, 2363,'K'],
             [47, 2362, 2437.2,'K'],
             [48, 2439, 2510,'K']]
             
    outorders, wl0, wlf, colors = [], [], [], []
    for order in order_mask:
        outorders.append(order[0])
        wl0.append(order[1])
        wlf.append(order[2])
        colors.append(order[3])
    
    loc = {}
    loc['orders'] = outorders
    loc['wl0'] = wl0
    loc['wlf'] = wlf
    loc['colors'] = colors
    return loc

def read_rvdata(rvfile) :
    
    rvdata = ascii.read(rvfile,data_start=2)
    rvbjds = np.array(rvdata['rjd']) + 2400000.
    rvs, rverrs = np.array(rvdata["vrad"]), np.array(rvdata["svrad"])
    
    return rvbjds, rvs, rverrs


def load_array_of_spirou_spectra(inputdata, flatten_data_arrays=False, rvfile="", remove_blaze=False, verbose=False, plot=False) :

    loc = {}
    loc["input"] = inputdata
    loc["flatten_data_arrays"] = flatten_data_arrays
    # Read input RVs from file

    if rvfile == "":
        if verbose:
            print("Reading RV file...")
        loc["rvfile"] = rvfile
        rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
    else :
        rvbjds, rvs, rverrs = read_rvdata(rvfile)

        if len(rvs) != len(inputdata):
            print("WARNING: size of RVs is different than number of input *e.fits files")
            print("*** Ignoring input RVs ***")
            rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
    #---
    orders = spirou_order_mask()
    loc["orders"] = orders

    output = []
    spectra = []
    wlmin, wlmax = 1e20, -1e20
    speed_of_light_in_kps = constants.c / 1000.
    
    for i in range(len(inputdata)) :
        
        if verbose :
            print("Loadinng spectrum:",inputdata[i],"{0}/{1}".format(i,len(inputdata)-1))

        if inputdata[i].endswith('e.fits') :
            t_fits = False
            # create output CCF file name based on input spectrum file name
            outfilename = inputdata[i].replace("e.fits", "_s.fits")
        elif inputdata[i].endswith('t.fits') :
            t_fits = True
            #  create output CCF file name based on input spectrum file name
            outfilename = inputdata[i].replace("t.fits", "_s.fits")
        else :
            print("WARNING: File extension not supported, skipping ... ")
            continue

        output.append(outfilename)

        # Load SPIRou *e.fits spectrum file
        spectrum = load_spirou_AB_efits_spectrum(inputdata[i], nan_pos_filter=False, preprocess=True, apply_BERV_to_preprocess=False, remove_blaze=remove_blaze, normalization_in_preprocess=remove_blaze)

        # set source RVs
        spectrum['FILENAME'] = inputdata[i]
        spectrum["source_rv"] = rvs[i]
        spectrum["rvfile"] = rvfile
        spectrum['RV'] = rvs[i]
        spectrum['RVERR'] = rverrs[i]
        
        spectrum['DATE'] = spectrum['header0']["DATE"]

        spectrum['BJD_mid'] = spectrum['header1']["BJD"] + (spectrum['header0']['MJDEND'] - spectrum['header0']['MJD-OBS']) / 2.
        spectrum['BERV'] = float(spectrum['header1']['BERV'])
        spectrum['snr32'] = spectrum['header1']['SNR32']
        spectrum['airmass'] = spectrum['header0']['AIRMASS']
        spectrum['exptime'] = spectrum['header0']['EXPTIME']

        wl_mean = []
        if flatten_data_arrays :
            out_wl, out_flux, out_fluxerr, out_order = np.array([]), np.array([]), np.array([]), np.array([])
            wl_sf, vels = np.array([]), np.array([])
            if t_fits : out_recon = np.array([])
        else :
            out_wl, out_flux, out_fluxerr, out_order = [], [], [], []
            wl_sf, vels = [], []
            if t_fits : out_recon = []

        NORDERS = 0
        for order in range(len(orders['orders'])) :
            mask = ~np.isnan(spectrum['flux'][order])
            mask &= spectrum['wl'][order] > orders['wl0'][order]
            mask &= spectrum['wl'][order] < orders['wlf'][order]

            if len(spectrum['wl'][order][mask]) :
                wl, flux = spectrum['wl'][order][mask], spectrum['flux'][order][mask]
                fluxerr = spectrum['fluxerr'][order][mask]
                if t_fits :
                    recon = spectrum['Recon'][order][mask]
                
                wl0, wlf = np.min(wl), np.max(wl)
                if wl0 < wlmin :
                    wlmin=wl0
                if wlf > wlmax :
                    wlmax=wlf

                wlc = 0.5 * (wl0 + wlf)

                if plot :
                    p = plt.plot(wl, flux)
                    color = p[0].get_color()
                    plt.plot(spectrum['wl'][order], spectrum['flux'][order], color=color, lw=0.3, alpha=0.6)
                    if t_fits :
                        plt.plot(wl, flux * recon, color=color, lw=0.3, alpha=0.6)
                            
                order_vec = np.full_like(wl,float(NORDERS))

                vel_shift = spectrum['RV'] - spectrum['BERV']
                wl_stellar_frame = wl / (1.0 + vel_shift / speed_of_light_in_kps)
                vel = speed_of_light_in_kps * ( wl_stellar_frame / wlc - 1.)

                if flatten_data_arrays :
                    out_wl = np.append(out_wl, wl)
                    out_flux = np.append(out_flux, flux)
                    out_fluxerr = np.append(out_fluxerr, fluxerr)
                    if t_fits :
                        out_recon = np.append(out_recon, recon)
                    out_order = np.append(out_order, order_vec)
                    wl_sf = np.append(wl_sf, wl_stellar_frame)
                    vels = np.append(vels, vel)
                else :
                    out_wl.append(wl)
                    out_flux.append(flux)
                    out_fluxerr.append(fluxerr)
                    if t_fits :
                        out_recon.append(recon)
                    out_order.append(order_vec)
                    wl_sf.append(wl_stellar_frame)
                    vels.append(vel)

                wl_mean.append(wlc)

                NORDERS += 1

        if plot :
            plt.xlabel(r"wavelength [nm]")
            plt.xlabel(r"flux")
            plt.show()
            exit()

        spectrum['NORDERS'] = NORDERS
        spectrum['WLMIN'] = wlmin
        spectrum['WLMAX'] = wlmax
        spectrum['WLMEAN'] = np.array(wl_mean)

        spectrum['out_wl_sf'] = wl_sf
        spectrum['out_vels'] = vels
        
        spectrum['out_wl'] = out_wl
        spectrum['out_flux'] = out_flux
        spectrum['out_fluxerr'] = out_fluxerr
        spectrum['out_order'] = out_order
        if t_fits :
            spectrum['out_recon'] = out_recon

        spectra.append(spectrum)

    loc["output"] = output
    loc["spectra"] = spectra

    return loc


def get_spectral_data(array_of_spectra, ref_index=0, verbose=False) :
    
    if verbose:
        print("Loading data")
    
    input_flatten_data = array_of_spectra["flatten_data_arrays"]

    loc = {}

    loc["flatten_data_arrays"] = input_flatten_data

    spectra = array_of_spectra["spectra"]

    filenames, rvfiles, dates = [], [], []
    bjds, airmasses, rvs, rverrs, bervs = [], [], [], [], []
    wl0s, wlfs, wl_mean = [], [], []

    ref_spectrum = spectra[ref_index]
    n_valid_spectra = 0
    nspectra = len(spectra)

    snrs = []
    waves, waves_sf, vels = [], [], []
    fluxes, fluxerrs, orders = [], [], []
    recons = []
    
    if not input_flatten_data :
        
        for order in range(ref_spectrum['NORDERS']) :
            snrs.append([])

            waves.append([])
            waves_sf.append([])
            vels.append([])

            fluxes.append([])
            fluxerrs.append([])
            orders.append([])
            
            recons.append([])

    for i in range(nspectra) :
        
        spectrum = spectra[i]

        if verbose:
            print("Loading input spectrum {0}/{1} : {2}".format(i,nspectra-1,spectrum['FILENAME']))

        valid_spectrum = False
        
        if input_flatten_data and spectrum['NORDERS'] == ref_spectrum['NORDERS'] :
            if np.all(spectrum['out_wl']) == np.all(ref_spectrum['out_wl']) :
                valid_spectrum = True
        elif input_flatten_data == False and spectrum['NORDERS'] == ref_spectrum['NORDERS']:
            for order in range(spectrum['NORDERS']) :
                if np.any(spectrum['out_wl'][order]) != np.any(ref_spectrum['out_wl'][order]) :
                    valid_spectrum = False
                    break
                if order == spectrum['NORDERS'] - 1 :
                    valid_spectrum = True

        if valid_spectrum :
            
            filenames.append(spectrum['FILENAME'])
            rvfiles.append(spectrum['rvfile'])
            dates.append(spectrum['DATE'])
            
            bjds.append(spectrum['BJD_mid'])
            airmasses.append(spectrum['airmass'])
            rvs.append(spectrum['RV'])
            rverrs.append(spectrum['RVERR'])
            bervs.append(spectrum['BERV'])
            
            wl0s.append(spectrum['WLMIN'])
            wlfs.append(spectrum['WLMAX'])
            wl_mean.append(spectrum['WLMEAN'])

            if input_flatten_data :
                snrs.append(spectrum['snr32'])

                waves.append(spectrum['out_wl'])
                waves_sf.append(spectrum['out_wl_sf'])
                vels.append(spectrum['out_vels'])

                fluxes.append(spectrum['out_flux'])
                fluxerrs.append(spectrum['out_fluxerr'])
                orders.append(spectrum['out_order'])
                if 'out_recon' in spectrum.keys() :
                    recons.append(spectrum['out_recon'])
                        
            else :
                for order in range(spectrum['NORDERS']) :
                    mean_snr = np.nanmean(spectrum['out_flux'][order] / spectrum['out_fluxerr'][order])
                    snrs[order].append(mean_snr)
                    
                    waves[order].append(spectrum['out_wl'][order])
                    waves_sf[order].append(spectrum['out_wl_sf'][order])
                    vels[order].append(spectrum['out_vels'][order])

                    fluxes[order].append(spectrum['out_flux'][order])
                    fluxerrs[order].append(spectrum['out_fluxerr'][order])
                    orders[order].append(spectrum['out_order'][order])
                    if 'out_recon' in spectrum.keys() :
                        recons[order].append(spectrum['out_recon'][order])

            n_valid_spectra += 1

    loc["NSPECTRA"] = n_valid_spectra
    loc["NORDERS"] = ref_spectrum['NORDERS']
    
    bjds  = np.array(bjds, dtype=float)
    airmasses  = np.array(airmasses, dtype=float)
    rvs  = np.array(rvs, dtype=float)
    rverrs  = np.array(rverrs, dtype=float)
    bervs  = np.array(bervs, dtype=float)

    wl0s  = np.array(wl0s, dtype=float)
    wlfs  = np.array(wlfs, dtype=float)
    wl_mean  = np.array(wl_mean, dtype=float)

    if input_flatten_data :
        snrs  = np.array(snrs, dtype=float)

        waves  = np.array(waves, dtype=float)
        waves_sf  = np.array(waves_sf, dtype=float)
        vels  = np.array(vels, dtype=float)

        fluxes  = np.array(fluxes, dtype=float)
        fluxerrs  = np.array(fluxerrs, dtype=float)
        orders  = np.array(orders, dtype=float)
        if 'out_recon' in spectrum.keys() :
            recons = np.array(recons, dtype=float)

        loc["wl"] = waves[0]

    else :
        loc["wl"] = []
        for order in range(ref_spectrum['NORDERS']) :
            snrs[order] = np.array(snrs[order], dtype=float)
            
            waves[order]  = np.array(waves[order], dtype=float)
            waves_sf[order]  = np.array(waves_sf[order], dtype=float)
            vels[order]  = np.array(vels[order], dtype=float)

            fluxes[order]  = np.array(fluxes[order], dtype=float)
            fluxerrs[order]  = np.array(fluxerrs[order], dtype=float)
            orders[order]  = np.array(orders[order], dtype=float)
            if 'out_recon' in spectrum.keys() :
                recons[order]= np.array(recons[order], dtype=float)
            
            loc["wl"].append(waves[order][0])


    loc["bjd"] = bjds
    loc["airmass"] = airmasses
    loc["rv"] = rvs
    loc["rverr"] = rverrs
    loc["berv"] = bervs
    loc["wl0"] = wl0s
    loc["wlf"] = wlfs
    loc["wl_mean"] = wl_mean

    loc["snr"] = snrs

    loc["waves"] = waves
    loc["waves_sf"] = waves_sf
    loc["vels"] = vels
    
    loc["flux"] = fluxes
    loc["fluxerr"] = fluxerrs
    loc["order"] = orders
    if 'out_recon' in spectrum.keys() :
        loc["recon"] = recons

    return loc


def save_spectrum_to_fits(spectrum, output, wl0=0., wlf=1e50) :

    header = fits.Header()
    
    header.set('ORIGIN', "spiroulib.save_spectrum_to_fits()")
    header.set('FILENAME', output)
    header.set('UTCSAVED', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    
    header.set('BJDMID', spectrum['BJD_mid'], "Barycentric Julian Date at middle of exposure")
    header.set('BERV', spectrum['BERV'], "Barycentric Earth Radial Velocity correction (km/s)")
    header.set('SNR32', spectrum['snr32'], "Signal-to-Noise ratio for order 32")
    header.set('AIRMASS', spectrum['airmass'], "Airmass")
    header.set('RV', spectrum['RV'], "Radial velocity of source (km/s)")
    header.set('RVERR', spectrum['RVERR'], "Radial velocity of source (km/s)")
    header.set('RVFILE', spectrum['rvfile'], "Radial velocity data file")
    
    header.set('NORDERS', spectrum['NORDERS'], "Number of spectral orders")
    
    if wl0 > 0 :
        header.set('WL0', wl0, "Initial wavelength [nm]")
    if wlf < 1e50 :
        header.set('WLF', wlf, "Final wavelength [nm]")
    
    header.set('TTYPE1', "WAVE")
    header.set('TUNIT1', "NM")
    
    header.set('TTYPE2', "FLUX")
    header.set('TUNIT2', "COUNTS")
    
    header.set('TTYPE3', "FLUXERR")
    header.set('TUNIT3', "COUNTS")
    
    header.set('TTYPE4', "ORDER")
    header.set('TUNIT4', "NUMBER")

    wlmask = ((spectrum['out_wl'] > wl0) & (spectrum['out_wl'] < wlf))

    #minorder, maxorder = np.min(spectrum['out_order'][wlmask]), np.max(spectrum['out_order'][wlmask])
    #wlmask &= spectrum['out_order'] > minorder
    #wlmask &= spectrum['out_order'] < maxorder

    outhdulist = []
    
    primary_hdu = fits.PrimaryHDU(header=header)
    outhdulist.append(primary_hdu)
    
    hdu_wl = fits.ImageHDU(data=spectrum['out_wl'][wlmask], name="WAVE")
    outhdulist.append(hdu_wl)
    
    hdu_flux = fits.ImageHDU(data=spectrum['out_flux'][wlmask], name="FLUX")
    outhdulist.append(hdu_flux)
    
    hdu_fluxerr = fits.ImageHDU(data=spectrum['out_fluxerr'][wlmask], name="FLUXERR")
    outhdulist.append(hdu_fluxerr)
    
    hdu_order = fits.ImageHDU(data=spectrum['out_order'][wlmask], name="ORDER")
    outhdulist.append(hdu_order)
    
    mef_hdu = fits.HDUList(outhdulist)
    mef_hdu.writeto(output, overwrite=True)


def save_spirou_spectra_to_fits(dataset, wl0=0., wlf=1e50, overwrite=False, verbose=False) :

    for i in range(len(dataset["input"])) :
        
        output = dataset["output"][i]
        
        if os.path.exists(output) and not overwrite :
            print("File",output," already exists, skipping ...")
            continue
        
        if verbose :
            print("Saving spectrum:",dataset["input"][i]," to output=", output, "{0}/{1}".format(i,len(dataset["input"])-1))

        spectrum = dataset["spectra"][i]

        save_spectrum_to_fits(spectrum, output, wl0=wl0, wlf=wlf)


def load_array_of_1D_spectra(inputdata, ref_index=0, verbose=False, plot=False):
    
    if verbose:
        print("Loading data")

    loc = {}
    nspectra = len(inputdata)

    filenames, rvfiles, dates = [], [], []
    bjds, airmasses, rvs, rverrs, bervs, snrs = [], [], [], [], [], []
    wl0s, wlfs, wl_mean = [], [], []
    norders = []

    waves, fluxes, fluxerrs, orders = [], [], [], []
    waves_sf, vels = [], []

    # Take wavelength vector of reference spectrum
    ref_hdu = fits.open(inputdata[ref_index])
    ref_waves = ref_hdu["WAVE"].data
    n_valid_spectra = 0

    speed_of_light_in_kps = constants.c / 1000.

    for i in range(nspectra) :
        if verbose:
            print("Loading input file {0}/{1} : {2}".format(i,nspectra-1,inputdata[i]))
        # open FITS file
        hdu = fits.open(inputdata[i])
        # load main header
        hdr = hdu[0].header

        if np.all(hdu["WAVE"].data) == np.all(ref_waves) :
            # append quantities:
            filenames.append(hdr["FILENAME"])
            rvfiles.append(hdr["RVFILE"])
            dates.append(hdr["DATE"])
        
            bjds.append(hdr["BJDMID"])
            airmasses.append(hdr["AIRMASS"])
            rvs.append(hdr["RV"])
            rverrs.append(hdr["RVERR"])
            bervs.append(hdr["BERV"])
            snrs.append(hdr["SNR32"])

            norders.append(hdr["NORDERS"])
            wl0s.append(hdr["WL0"])
            wlfs.append(hdr["WLF"])

            waves.append(hdu["WAVE"].data)
            
            wlc = 0.5 * (np.min(hdu["WAVE"].data) + np.max(hdu["WAVE"].data))
            wl_mean.append(wlc)

            vel_shift = hdr["RV"] - hdr["BERV"]

            wl_sf = hdu["WAVE"].data / (1.0 + vel_shift / speed_of_light_in_kps)
            waves_sf.append(wl_sf)

            vel = speed_of_light_in_kps * ( wl_sf / wlc - 1.)
            vels.append(vel)
            
            fluxes.append(hdu["FLUX"].data)
            fluxerrs.append(hdu["FLUXERR"].data)
            orders.append(hdu["ORDER"].data)

            n_valid_spectra += 1
    
    loc["NSPECTRA"] = n_valid_spectra

    bjds  = np.array(bjds, dtype=float)
    airmasses  = np.array(airmasses, dtype=float)
    rvs  = np.array(rvs, dtype=float)
    rverrs  = np.array(rverrs, dtype=float)
    bervs  = np.array(bervs, dtype=float)
    snrs  = np.array(snrs, dtype=float)

    norders  = np.array(norders, dtype=float)
    wl0s  = np.array(wl0s, dtype=float)
    wlfs  = np.array(wlfs, dtype=float)
    wl_mean  = np.array(wl_mean, dtype=float)

    waves  = np.array(waves, dtype=float)
    waves_sf  = np.array(waves_sf, dtype=float)
    vels  = np.array(vels, dtype=float)

    fluxes  = np.array(fluxes, dtype=float)
    fluxerrs  = np.array(fluxerrs, dtype=float)
    orders  = np.array(orders, dtype=float)
    
    loc["bjd"] = bjds
    loc["airmass"] = airmasses
    loc["rv"] = rvs
    loc["rverr"] = rverrs
    loc["berv"] = bervs
    loc["snr"] = snrs
    loc["norder"] = norders
    loc["wl0"] = wl0s
    loc["wlf"] = wlfs
    loc["wl_mean"] = wl_mean
    
    loc["wl"] = waves[0]
    
    loc["waves"] = waves
    loc["waves_sf"] = waves_sf
    loc["vels"] = vels
    
    loc["flux"] = fluxes
    loc["fluxerr"] = fluxerrs
    loc["order"] = orders
    
    if plot :
        exoatmos_utils.plot_2d(loc["wl"], loc["bjd"], loc["flux"])
    
    return loc


def calculate_shift_vels(spectra):
    """
        Calculate wavelengths in the stellar rest frame then convert wavelengths
        into velocities
    """
    speed_of_light_kps= (constants.c / 1000.)

    loc["waves"] = waves

    V_shift     = (self.rv_s - self.berv).reshape((len(self.date),1))
    W_corr      = W / (1.0 + V_shift / c0)
    V_corr      = c0 * ( W_corr / self.W_mean - 1.)

    return spectra
