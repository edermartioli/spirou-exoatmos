# -*- coding: utf-8 -*-


# -*- coding: iso-8859-1 -*-
"""
    Created on Feb 7 2021
    
    Description: utilities for spirou data reduction
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from copy import deepcopy

import astropy.io.fits as fits
from astropy.io import ascii
from scipy import constants

from scipy.interpolate import interp1d


def check_efits(inputdata, verbose) :
    inputedata = []
    for i in range(len(inputdata)) :
        epath = inputdata[i].replace("t.fits","e.fits")
        if os.path.exists(epath):
            inputedata.append(epath)
    efits_ok = False
    if len(inputedata) == len(inputdata) :
        efits_ok = True
    
    if verbose :
        if efits_ok :
            print("All corresponding e.fits data are ok")
        else :
            print("WARNING: missing e.fits data, ignoring ...")

    return inputedata, efits_ok


def read_rvdata(rvfile) :
    
    rvdata = ascii.read(rvfile,data_start=2)
    rvbjds = np.array(rvdata['rjd']) + 2400000.
    rvs, rverrs = np.array(rvdata["vrad"]), np.array(rvdata["svrad"])
    
    return rvbjds, rvs, rverrs


def load_array_of_spirou_spectra(inputdata, rvfile="", correct_blaze=True, apply_berv=True, silent=True, verbose=False, plot=False, plot_diagnostics=False) :

    loc = {}
    loc["input"] = inputdata

    if silent :
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    if rvfile == "":
        #print("WARNING: RV file has not been provided ...")
        #print("*** Setting all source RVs equal to zero ***")
        rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
    else :
        loc["rvfile"] = rvfile

        if verbose :
            print("Reading RVs from file:",rvfile)
        
        rvbjds, rvs, rverrs = read_rvdata(rvfile)
        
        if len(rvs) != len(inputdata):
            if verbose :
                print("WARNING: size of RVs is different than number of input *t.fits files")
                print("*** Ignoring input RVs and setting all source RVs equal to zero ***")
            rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
        else :
            for i in range(len(rvs)):
                hdr = fits.getheader(inputdata[i],1)
                print(i, inputdata[i], hdr['BJD'], rvbjds[i], rvs[i], rverrs[i])
    #---
    loc["source_rv"] = np.nanmedian(rvs)

    spectra = []
    speed_of_light_in_kps = constants.c / 1000.

    if plot_diagnostics :
        bjd, snr, airmass, berv = [], [], [], []
    
    for i in range(len(inputdata)) :
        
        #if verbose :
        #    print("Loading spectrum:",inputdata[i],"{0}/{1}".format(i,len(inputdata)-1))

        if not (inputdata[i].endswith('t.fits') or inputdata[i].endswith('e.fits')) :
            if verbose:
                print("WARNING: File extension not supported, skipping ... ")
            continue

        spectrum = {}

        # set source RVs
        spectrum['FILENAME'] = inputdata[i]
        spectrum["source_rv"] = rvs[i]
        spectrum["rvfile"] = rvfile
        spectrum['RV'] = rvs[i]
        spectrum['RVERR'] = rverrs[i]

        wl_mean = []
        out_wl, out_flux, out_fluxerr, out_order = [], [], [], []
        out_recon, out_blaze = [], []
        wl_sf, vels = [], []

        hdu = fits.open(inputdata[i])

        hdr = deepcopy(hdu[0].header + hdu[1].header)

        spectrum["header"] = hdr

        # get DETECTOR GAIN and READ NOISE from header
        gain, rdnoise = hdr['GAIN'], hdr['RDNOISE']

        WaveAB = deepcopy(hdu["WaveAB"].data)
        FluxAB = deepcopy(hdu["FluxAB"].data)
        BlazeAB = deepcopy(hdu["BlazeAB"].data)
        if inputdata[i].endswith('t.fits') :
            Recon = deepcopy(hdu["Recon"].data)

        spectrum['DATE'] = hdr["DATE"]
        spectrum['BJD_mid'] = hdr["BJD"] + (hdr['MJDEND'] - hdr['MJD-OBS']) / 2.
        spectrum['BERV'] = float(hdr['BERV'])

        spectrum['airmass'] = hdr['AIRMASS']
        spectrum['exptime'] = hdr['EXPTIME']

        # Estimate signal-to-noise
        max_flux = []
        for order in range(49) :
            finite = np.isfinite(FluxAB[order])
            if len(FluxAB[order][finite]) :
                max_flux.append(np.nanmax(FluxAB[order][finite]))
        mean_flux = np.nanmean(np.array(max_flux))
        maxsnr = mean_flux / np.sqrt(mean_flux + (rdnoise * rdnoise / gain * gain))

        spectrum['SNR'] = maxsnr
        
        if plot_diagnostics :
            if i == 0 :
                objectname = hdr['OBJECT']
            bjd.append(spectrum['BJD_mid'])
            snr.append(maxsnr)
            airmass.append(spectrum['airmass'])
            berv.append(spectrum['BERV'])

        if verbose :
            print("Spectrum: {0} {1}/{2} BJD={3:.6f} SNR={4:.1f} BERV={5:.3f} km/s".format(inputdata[i],i,len(inputdata)-1,spectrum['BJD_mid'],maxsnr,spectrum['BERV']))
        for order in range(49) :
            
            wl = deepcopy(WaveAB[order])
            flux = deepcopy(FluxAB[order])
            
            if inputdata[i].endswith('t.fits') :
                recon = deepcopy(Recon[order])
                fluxerr = np.sqrt(flux *  + (rdnoise * rdnoise / gain * gain))
            else :
                fluxerr = np.sqrt(flux + (rdnoise * rdnoise / gain * gain))

            blaze = deepcopy(BlazeAB[order])

            if correct_blaze :
                nblaze = blaze / np.nanpercentile(blaze,90)
                flux /= nblaze
                fluxerr /= nblaze

            wlc = 0.5 * (wl[0] + wl[-1])

            if plot :
                p = plt.plot(wl, flux)
                color = p[0].get_color()
                plt.plot(wl, flux, color=color, lw=0.3, alpha=0.6)
                if inputdata[i].endswith('t.fits') :
                    plt.plot(wl, flux * recon, color=color, lw=0.3, alpha=0.6)
            
            order_vec = np.full_like(wl,float(order))

            if apply_berv :
                vel_shift = spectrum['RV'] - spectrum['BERV']
            else :
                vel_shift = spectrum['RV']

            # relativistic calculation
            wl_stellar_frame = wl * np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
            #wl_stellar_frame = wl / (1.0 + vel_shift / speed_of_light_in_kps)
            vel = speed_of_light_in_kps * ( wl_stellar_frame / wlc - 1.)

            out_wl.append(wl)
            out_flux.append(flux)
            out_fluxerr.append(fluxerr)
            if inputdata[i].endswith('t.fits') :
                out_recon.append(recon)
                del recon
            out_blaze.append(blaze)
            out_order.append(order_vec)
            wl_sf.append(wl_stellar_frame)
            vels.append(vel)
            wl_mean.append(wlc)


        if plot :
            plt.xlabel(r"wavelength [nm]")
            plt.xlabel(r"flux")
            plt.show()
            exit()

        spectrum['wlmean'] = np.array(wl_mean)

        spectrum['wl_sf'] = wl_sf
        spectrum['vels'] = vels

        spectrum['order'] = out_order
        spectrum['wl'] = out_wl
        spectrum['flux'] = out_flux
        spectrum['fluxerr'] = out_fluxerr
        spectrum['blaze'] = out_blaze
        if inputdata[i].endswith('t.fits') :
            spectrum['recon'] = out_recon

        spectra.append(spectrum)

        hdu.close()

    loc["spectra"] = spectra

    if plot_diagnostics :
        bjd = np.array(bjd)
        snr = np.array(snr)
        airmass = np.array(airmass)
        berv = np.array(berv)
        
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('{} spectra of {}'.format(len(inputdata), objectname))
        axs[0].plot(bjd, snr, '-', color="orange",label="SNR")
        axs[0].set_ylabel('SNR')
        axs[0].legend()

        axs[1].plot(bjd, airmass, '--', color="olive",label="Airmass")
        axs[1].set_ylabel('Airmass')
        axs[1].legend()

        axs[2].plot(bjd, berv, ':', color="darkblue",label="BERV")
        axs[2].set_xlabel('BJD')
        axs[2].set_ylabel('BERV [km/s]')
        axs[2].legend()
        
        plt.show()
        #exit()

    return loc


def get_spectral_data(array_of_spectra, ref_index=0, edge_size=100, verbose=False) :
    
    if verbose :
        print("Loading data")
    
    loc = {}

    spectra = array_of_spectra["spectra"]

    filenames, rvfiles, dates = [], [], []
    bjds, airmasses, rvs, rverrs, bervs = [], [], [], [], []
    wl_mean = []

    ref_spectrum = spectra[ref_index]

    nspectra = len(spectra)
    loc['nspectra'] = nspectra
    snrs = []
    waves, waves_sf, vels = [], [], []
    fluxes, fluxerrs, orders = [], [], []
    blazes, recons = [], []
    wl_out, wlsf_out, vel_out = [], [], []
    hdr = []

    for order in range(49) :
        snrs.append([])
        orders.append([])
        waves.append([])
        waves_sf.append([])
        vels.append([])
        fluxes.append([])
        fluxerrs.append([])
        blazes.append([])
        recons.append([])

    for i in range(nspectra) :
        
        spectrum = spectra[i]

        if verbose:
            print("Loading input spectrum {0}/{1} : {2}".format(i,nspectra-1,spectrum['FILENAME']))
        
        filenames.append(spectrum['FILENAME'])
        hdr.append(spectrum['header'])
        rvfiles.append(spectrum['rvfile'])
        dates.append(spectrum['DATE'])
        
        bjds.append(spectrum['BJD_mid'])
        airmasses.append(spectrum['airmass'])
        rvs.append(spectrum['RV'])
        rverrs.append(spectrum['RVERR'])
        bervs.append(spectrum['BERV'])
        
        wl_mean.append(spectrum['wlmean'])
        
        for order in range(49) :
            mean_snr = np.nanmean(spectrum['flux'][order] / spectrum['fluxerr'][order])
            
            snrs[order].append(mean_snr)

            orders[order].append(spectrum['order'][order])

            waves[order].append(spectrum['wl'][order])
            waves_sf[order].append(spectrum['wl_sf'][order])
            vels[order].append(spectrum['vels'][order])
            if i==0 :
                wl_out.append(spectrum['wl'][order])
                wlsf_out.append(spectrum['wl_sf'][order])
                vel_out.append(spectrum['vels'][order])
            
            fluxes[order].append(spectrum['flux'][order])
            fluxerrs[order].append(spectrum['fluxerr'][order])
            blazes[order].append(spectrum['blaze'][order])
            if 'recon' in spectrum.keys():
                recons[order].append(spectrum['recon'][order])

    bjds  = np.array(bjds, dtype=float)
    airmasses  = np.array(airmasses, dtype=float)
    rvs  = np.array(rvs, dtype=float)
    rverrs  = np.array(rverrs, dtype=float)
    bervs  = np.array(bervs, dtype=float)
    wl_mean  = np.array(wl_mean, dtype=float)

    for order in range(49) :
        snrs[order] = np.array(snrs[order], dtype=float)
        
        orders[order]  = np.array(orders[order], dtype=float)

        waves[order]  = np.array(waves[order], dtype=float)
        waves_sf[order]  = np.array(waves_sf[order], dtype=float)
        vels[order]  = np.array(vels[order], dtype=float)

        fluxes[order]  = np.array(fluxes[order], dtype=float)
        fluxerrs[order]  = np.array(fluxerrs[order], dtype=float)
        blazes[order] = np.array(blazes[order], dtype=float)
        if 'recon' in spectrum.keys():
            recons[order]= np.array(recons[order], dtype=float)


    loc["header"] = hdr
    loc["filenames"] = filenames

    loc["bjds"] = bjds
    loc["airmasses"] = airmasses
    loc["rvs"] = rvs
    loc["rverrs"] = rverrs
    loc["bervs"] = bervs
    loc["wl_mean"] = wl_mean

    loc["snrs"] = snrs
    loc["orders"] = orders
    loc["waves"] = waves
    loc["waves_sf"] = waves_sf
    loc["vels"] = vels

    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs
    loc["blazes"] = blazes
    if 'recon' in spectrum.keys():
        loc["recons"] = recons

    # set base wavelength from first spectrum
    loc["wl"] = np.array(wl_out)
    loc["wl_sf"] = np.array(wlsf_out)
    loc["vel"] = np.array(vel_out)

    loc = get_wlmin_wlmax(loc, edge_size=edge_size)

    return loc


def get_wlmin_wlmax(spectra, edge_size=100) :

    speed_of_light_in_kps = constants.c / 1000.
    
    # find minimum and maximum wavelength for valid (not NaN) data
    wlmin, wlmax = np.full(49,-1e20), np.full(49,+1e20)

    for order in range(49) :
        for i in range(spectra['nspectra']) :
            minwl_sf = np.nanmin(spectra["waves_sf"][order][i])
            maxwl_sf = np.nanmax(spectra["waves_sf"][order][i])

            if minwl_sf > wlmin[order] :
                wlmin[order] = minwl_sf
            
            if maxwl_sf < wlmax[order] :
                wlmax[order] = maxwl_sf

    spectra["wlmin"] = wlmin
    spectra["wlmax"] = wlmax

    return spectra


def get_gapfree_windows(spectra, max_vel_distance=3.0, min_window_size=120., fluxkey="fluxes", velkey="vels", wavekey="waves", verbose=False) :
    
    windows = []
    
    for order in range(49) :
        windows.append([])
    
    for order in range(49) :
    #for order in range(38,39) :
        if verbose :
            print("Calculating windows with size > {0:.0f} km/s and with gaps < {1:.1f} km/s for order={2}".format(min_window_size,max_vel_distance, order))

        for i in range(spectra['nspectra']) :
        #for i in range(0,1) :
        
            nanmask = ~np.isnan(spectra[fluxkey][order][i])
            vels = spectra[velkey][order][i]
            wl = spectra[wavekey][order][i]

            if len(vels[nanmask]) > min_window_size / 4.0 :

                dv = np.abs(vels[nanmask][1:] - vels[nanmask][:-1])
            
                gaps = dv > max_vel_distance
            
                window_v_ends = np.append(vels[nanmask][:-1][gaps],vels[nanmask][-1])
                window_v_starts = np.append(vels[nanmask][0],vels[nanmask][1:][gaps])

                window_size = np.abs(window_v_ends - window_v_starts)
                good_windows = window_size > min_window_size

                window_wl_ends = np.append(wl[nanmask][:-1][gaps],wl[nanmask][-1])
                window_wl_starts = np.append(wl[nanmask][0],wl[nanmask][1:][gaps])

                loc_windows = np.array((window_wl_starts[good_windows],window_wl_ends[good_windows])).T
            else :
                loc_windows = np.array([])

            windows[order].append(loc_windows)

    # save window function
    spectra["windows"] = windows

    return spectra


def mask_fluxes_out_of_windows(spectra, fluxkey="fluxes", velkey="vels", wavekey="waves", verbose=False) :

    for order in range(49) :
        if verbose:
            print("Masking out-of-window data for order ", order )
        
        wl = spectra[wavekey][order]
        
        for i in range(spectra['nspectra']) :
            
            windows = spectra["windows"][order][i]
            
            if len(windows) :
                mask = ((wl[i] >= windows[0][0]) & (wl[i] <= windows[0][1]))
                for j in range(1,len(windows)) :
                    mask ^= ((wl[i] >= windows[j][0]) & (wl[i] <= windows[j][1]))
        
                spectra[fluxkey][order][i][~mask] = np.nan

    return spectra


def set_common_wl_grid(spectra, vel_sampling=2.0, verbose=False) :

    if "wlmin" not in spectra.keys() or "wlmax" not in spectra.keys():
        print("ERROR: function set_common_wl_grid() requires keywords wlmin and wlmax in input spectra, exiting.. ")
        exit()
    
    common_wl, common_vel = [], []
    speed_of_light_in_kps = constants.c / 1000.
    drv = 1.0 + vel_sampling / speed_of_light_in_kps
    drv_neg = 1.0 - vel_sampling / speed_of_light_in_kps

    np_min = 1e50
    
    for order in range(49) :
        
        wlmin = spectra["wlmin"][order]
        wlmax = spectra["wlmax"][order]
        
        wl_array = []
        wl = wlmin
        while wl < wlmax * drv_neg :
            wl *= drv
            wl_array.append(wl)
        wl_array = np.array(wl_array)
        
        wlc = (wl_array[0]+wl_array[-1])/2

        vels = speed_of_light_in_kps * ( wl_array / wlc - 1.)
        
        common_vel.append(vels)
        common_wl.append(wl_array)
    
        if len(wl_array) < np_min :
            np_min = len(wl_array)

    for order in range(49) :
        diff_size = len(common_wl[order]) - np_min
        half_diff_size = int(diff_size/2)
        
        if diff_size :
            common_vel[order] = common_vel[order][half_diff_size:np_min+half_diff_size]
            common_wl[order] = common_wl[order][half_diff_size:np_min+half_diff_size]

    spectra["common_vel"] = np.array(common_vel, dtype=float)
    spectra["common_wl"] = np.array(common_wl, dtype=float)

    return spectra


# function to interpolate spectrum
def interp_spectrum(wl_out, wl_in, flux_in, good_windows, kind='cubic') :

    flux_out = np.full_like(wl_out, np.nan)

    for w in good_windows :

        mask = wl_in >= w[0]
        mask &= wl_in <= w[1]

        wl_in_copy = deepcopy(wl_in)
        flux_in_copy = deepcopy(flux_in)

        # create interpolation function for input data
        f = interp1d(wl_in_copy[mask], flux_in_copy[mask], kind=kind)

        wl1, wl2 = w[0], w[1]

        if wl1 < wl_in[mask][0] :
            wl1 = wl_in[mask][0]
        if wl2 > wl_in[mask][-1] :
            wl2 = wl_in[mask][-1]

        out_mask = wl_out > wl1
        out_mask &= wl_out < wl2

        # interpolate data
        flux_out[out_mask] = f(wl_out[out_mask])

    return flux_out


def align_spectra(spectra, star_frame=True, interp_kind='cubic', plot=False, verbose=False) :

    if "common_wl" not in spectra.keys() :
        print("ERROR: function align_spectra() requires keyword common_wl in input spectra, exiting.. ")
        exit()
    
    aligned_waves, aligned_waves_sf = [], []
    aligned_fluxes, aligned_fluxerrs = [], []
    aligned_blazes, aligned_recons = [], []

    for order in range(49) :
        aligned_waves.append([])
        aligned_waves_sf.append([])
        aligned_fluxes.append([])
        aligned_fluxerrs.append([])
        aligned_blazes.append([])
        aligned_recons.append([])

    for order in range(49) :
    #for order in range(38,39) :
        if verbose :
            print("Aligning all spectra to a common wavelength grid for order=", order)

        common_wl = spectra['common_wl'][order]
        
        for i in range(spectra['nspectra']) :
        #for i in range(0,1) :
            if "windows" in spectra.keys() :
                windows = spectra["windows"][order][i]
            else :
                windows = [[common_wl[0],common_wl[-1]]]
            nanmask = ~np.isnan(spectra["fluxes"][order][i])

            flux = spectra["fluxes"][order][i][nanmask]
            fluxerr = spectra["fluxerrs"][order][i][nanmask]
            blaze = spectra["blazes"][order][i][nanmask]

            aligned_waves[order].append(common_wl)

            if star_frame :
                wl = spectra["waves_sf"][order][i][nanmask]
            else :
                wl = spectra["waves"][order][i][nanmask]

            wl_rest = spectra["waves"][order][i][nanmask]

            aligned_flux = interp_spectrum(common_wl, wl, flux, windows, kind=interp_kind)
            aligned_fluxerr = interp_spectrum(common_wl, wl, fluxerr, windows, kind=interp_kind)
            aligned_blaze = interp_spectrum(common_wl, wl, blaze, windows, kind=interp_kind)

            aligned_fluxes[order].append(aligned_flux)
            aligned_fluxerrs[order].append(aligned_fluxerr)
            aligned_blazes[order].append(aligned_blaze)

            if "recons" in spectra.keys() :
                recon = spectra["recons"][order][i][nanmask]
                aligned_recon = interp_spectrum(common_wl, wl_rest, recon, windows, kind=interp_kind)
                aligned_recons[order].append(aligned_recon)

            if plot :
                p = plt.plot(wl,flux, ":", lw=0.3, alpha=0.6)
                color = p[0].get_color()
                plt.plot(wl, flux, ":", color=color, lw=0.3, alpha=0.6)
                plt.plot(common_wl, aligned_flux, '-', color=color)
                for w in windows:
                    plt.vlines(w, [np.min(flux),np.min(flux)], [np.max(flux),np.max(flux)], color = "r", ls="--")
    if plot :
        plt.show()

    spectra["aligned_fluxes"] = aligned_fluxes
    spectra["aligned_fluxerrs"] = aligned_fluxerrs
    spectra["aligned_blazes"] = aligned_blazes
    if "recons" in spectra.keys() :
        spectra["aligned_recons"] = aligned_recons
    spectra["aligned_waves"] = aligned_waves

    return spectra


def reduce_spectra(spectra, mask=[], nsig_clip=0.0, combine_by_median=False, subtract=True,  output="", fluxkey="fluxes", fluxerrkey="fluxerrs", wavekey="wl", update_spectra=False, plot=False, verbose=False) :
    
    signals, ref_snrs, noises,  orders = [], [], [], []
    rel_noises = []
    snrs, snrs_err = [], []
    template = []

    if subtract :
        sub_flux_base = 0.0
    else :
        sub_flux_base = 1.0
    
    for order in range(49) :
        
        if verbose:
            print("Processing order {0} / 48 ...".format(order))
        
        # 1st pass - to build template for each order and subtract out all spectra by template
        order_template = calculate_template(spectra[fluxkey][order], wl=spectra[wavekey][order], mask=mask, fit=True, median=combine_by_median, subtract=True, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # Recover fluxes already shifted and re-scaled to match the template
        fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base

        # 2nd pass - to build template from calibrated fluxes
        order_template = calculate_template(fluxes, wl=spectra[wavekey][order], mask=mask, fit=True, median=combine_by_median, subtract=subtract, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # apply sigma-clip using template and median dispersion in time as clipping criteria
        # bad values can either be replaced by the template values, by interpolated values or by NaNs
        if nsig_clip > 0 :
            order_template = sigma_clip(order_template, nsig=nsig_clip, interpolate=False, replace_by_model=False, sub_flux_base=sub_flux_base, plot=False)
            #order_template = sigma_clip_remove_bad_columns(order_template, nsig=nsig_clip, plot=False)

        # Recover fluxes already shifted and re-scaled to match the template
        if subtract :
            fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base
        else:
            fluxes = order_template["flux_arr_sub"] * order_template["flux"]

        # 3rd pass - Calculate a final template combined by the mean
        order_template = calculate_template(fluxes, wl=spectra[wavekey][order], mask=mask, fit=True, median=False, subtract=False, sub_flux_base=sub_flux_base, verbose=False, plot=plot)

        # save number of spectra in the time series
        order_template['nspectra'] = spectra['nspectra']

        # save order flag
        order_template['order'] = order

        mean_signal, noise = [], []
        rel_noise = []
        for i in range(spectra['nspectra']) :
            mean_signal.append(np.nanmean(spectra[fluxkey][order][i]))
            noise.append(np.nanstd(order_template["flux_residuals"][i]))
            rel_noise.append(np.nanstd(order_template["flux_arr_sub"][i]))

        mean_signal, noise = np.array(mean_signal), np.array(noise)
        rel_noise = np.array(rel_noise)
        
        m_signal = np.nanmean(mean_signal)
        m_ref_snr = np.nanmean(spectra["snrs"][order])
        m_noise = np.nanmean(noise)
        m_snr = np.nanmean(mean_signal/noise)
        sig_snr = np.nanstd(mean_signal/noise)
        m_rel_noise = np.nanmedian(rel_noise)

        if verbose :
            print("Order {0}: ref_snr={1:.0f} snr={2:.0f}+-{3:.0f} sigma={4:.2f}%".format(order, m_ref_snr, m_snr, sig_snr, m_rel_noise*100.))

        signals.append(m_signal)
        noises.append(m_noise)
        ref_snrs.append(m_ref_snr)
        snrs.append(m_snr)
        snrs_err.append(sig_snr)
        orders.append(order)

        order_template["mean_signal"] = mean_signal
        order_template["ref_snr"] = spectra["snrs"][order]
        order_template["noise"] = noise
        order_template["mean_snr"] = mean_signal/noise
        order_template["rel_noise"] = m_rel_noise

        template.append(order_template)
    
        if update_spectra :
            # Recover fluxes already shifted and re-scaled to match the template
            if subtract :
                fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base
                fluxerr = order_template["fluxerr"]
            else:
                fluxes = order_template["flux_arr_sub"] * order_template["flux"]
                fluxerr = order_template["flux_arr_sub"] * order_template["fluxerr"]

            for i in range(spectra['nspectra']) :
                spectra[fluxkey][order][i] = fluxes[i]
                spectra[fluxerrkey][order][i] = fluxerr[i]
    
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


#################################################################################################
def calculate_template(flux_arr, wl=[], mask=[], fit=False, median=True, subtract=False, sub_flux_base=1.0, min_npoints=100, verbose=False, plot=False, pfilename=""):
    """
        Compute the mean/median template spectrum along the time axis and divide/subtract
        each exposure by the mean/median
        
        Inputs:
        - flux_arr: 2D flux matrix (N_exposures, N_wavelengths)
        - wl: 1D wavelength array (N_wavelengths)
        - mask: define spectra that will be used to build template, use all if =None
        - fit: boolean to fit median spectrum to each observation before normalizing it
        - median: boolean to calculate median instead of mean
        - subtract: boolean to subtract instead of dividing out spectra by the mean/median template

        Outputs:
        - loc: python dict containing all products
    """
    
    loc = {}

    loc["fit"] = fit
    loc["mask"] = mask
    loc["median"] = median
    loc["subtract"] = subtract
    loc["pfilename"] = pfilename

    if wl == []:
        x = np.arange(len(flux_arr[0]))
    else :
        x = wl

    if verbose :
        print("Calculating template out of {0} input spectra".format(len(flux_arr)))
    
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
        flux_calib = []
        flux_fit = []
        
        shift_arr = []
        scale_arr = []
        quadratic_arr = []

        def flux_model (coeffs, template, wave):
            outmodel = coeffs[2] * wave * wave + coeffs[1] * template + coeffs[0]
            return outmodel
        
        def errfunc (coeffs, fluxes, xx) :
            nanmask = ~np.isnan(fluxes)
            residuals = fluxes[nanmask] - flux_model (coeffs, flux_template[nanmask], xx[nanmask])
            return residuals

        for i in range(len(flux_arr)):
            
            nanmask = ~np.isnan(flux_arr[i])
            
            if len(flux_arr[i][nanmask]) > min_npoints :
                #guess = [0.0001, 1.001]
                guess = [0.0001, 1.001, 0.0000001]
                pfit, success = optimize.leastsq(errfunc, guess, args=(flux_arr[i], x))
            else :
                pfit = [0.,1.,0.]

            flux_template_fit = flux_model(pfit, flux_template, x)
            flux_fit.append(flux_template_fit)

            shift_arr.append(pfit[0])
            scale_arr.append(pfit[1])
            quadratic_arr.append(pfit[2])

            #flux_calib_loc = (flux_arr[i] - pfit[0]) / pfit[1]
            flux_calib_loc = (flux_arr[i] - pfit[2] * x * x - pfit[0]) / pfit[1]
            flux_calib.append(flux_calib_loc)

        loc["shift"] = np.array(shift_arr, dtype=float)
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
        else :
            # Compute median on all spectra along the time axis
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
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
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
    loc["wl"] = x
    loc["flux_arr_sub"] = flux_arr_sub
    loc["flux_residuals"] = residuals
    loc["snr"] = flux_arr / flux_template_medsig

    loc["template_source"] = "data"
    
    template_nanmask = ~np.isnan(flux_template)
    template_nanmask &= ~np.isnan(flux_template_medsig)
    
    if len(flux_template_medsig[template_nanmask]) :
        loc["fluxerr_model"] = fit_continuum(x, flux_template_medsig, function='polynomial', order=5, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux error", plot_fit=False,silent=True)
    else :
        loc["fluxerr_model"] = np.full_like(x,np.nan)

    if plot :
        plot_template_products(loc, pfilename=pfilename)
    
    return loc


def plot_2d(x, y, z, model=[], LIM=None, LAB=None, z_lim=None, title="", pfilename="", cmap="gist_heat"):
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


def plot_template_products(template, pfilename="") :

    wl = template["wl"]

    for i in range(len(template["flux_arr"])) :
        
        flux = template["flux_arr"][i]
        resids = template["flux_residuals"][i]

        if i == len(template["flux_arr"]) - 1 :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="SPIRou data")
            plt.plot(wl, resids,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="Residuals")
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


def plot_2Dtime_series(template, bjd, wl0=1083.0, wlf=1083.8) :

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

    plot_2d(wl[wlsorted], bjd, fluxes, z_lim = [1.-2*mediansig,1.+2*mediansig], cmap="coolwarm")

    #plot_2d(wl, bjd, fluxes, z_lim = [0.98,1.02], cmap="gist_heat")


def fit_continuum(wav, spec, function='polynomial', order=3, nit=5, rej_low=2.0,
    rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,
                  min_points=10, xlabel="", ylabel="", plot_fit=True, verbose=False, silent=False):
    """
    Continuum fitting re-implemented from IRAF's 'continuum' function
    in non-interactive mode only but with additional options.

    :Parameters:
    
    wav: array(float)
        abscissa values (wavelengths, velocities, ...)

    spec: array(float)
        spectrum values

    function: str
        function to fit to the continuum among 'polynomial', 'spline3'

    order: int
        fit function order:
        'polynomial': degree (not number of parameters as in IRAF)
        'spline3': number of knots

    nit: int
        number of iteractions of non-continuum points
        see also 'min_points' parameter

    rej_low: float
        rejection threshold in unit of residul standard deviation for point
        below the continuum

    rej_high: float
        same as rej_low for point above the continuum

    grow: int
        number of neighboring points to reject

    med_filt: int
        median filter the spectrum on 'med_filt' pixels prior to fit
        improvement over IRAF function
        'med_filt' must be an odd integer

    percentile_low: float
        reject point below below 'percentile_low' percentile prior to fit
        improvement over IRAF function
        "percentile_low' must be a float between 0. and 100.

    percentile_high: float
        same as percentile_low but reject points in percentile above
        'percentile_high'
        
    min_points: int
        stop rejection iterations when the number of points to fit is less than
        'min_points'

    plot_fit: bool
        if true display two plots:
            1. spectrum, fit function, rejected points
            2. residual, rejected points

    verbose: bool
        if true fit information is printed on STDOUT:
            * number of fit points
            * RMS residual
    """
    if silent :
        import warnings
        warnings.simplefilter('ignore', np.RankWarning)
    
    mspec = np.ma.masked_array(spec, mask=np.zeros_like(spec))
    # mask 1st and last point: avoid error when no point is masked
    # [not in IRAF]
    mspec.mask[0] = True
    mspec.mask[-1] = True
    
    mspec = np.ma.masked_where(np.isnan(spec), mspec)
    
    # apply median filtering prior to fit
    # [opt] [not in IRAF]
    if int(med_filt):
        fspec = signal.medfilt(spec, kernel_size=med_filt)
    else:
        fspec = spec
    # consider only a fraction of the points within percentile range
    # [opt] [not in IRAF]
    mspec = np.ma.masked_where(fspec < np.percentile(fspec, percentile_low),
        mspec)
    mspec = np.ma.masked_where(fspec > np.percentile(fspec, percentile_high),
        mspec)
    # perform 1st fit
    if function == 'polynomial':
        coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
        cont = np.poly1d(coeff)(wav)
    elif function == 'spline3':
        knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
        spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
        cont = sint.splev(wav, spl)
    else:
        raise(AttributeError)
    # iteration loop: reject outliers and fit again
    if nit > 0:
        for it in range(nit):
            res = fspec-cont
            sigm = np.std(res[~mspec.mask])
            # mask outliers
            mspec1 = np.ma.masked_where(res < -rej_low*sigm, mspec)
            mspec1 = np.ma.masked_where(res > rej_high*sigm, mspec1)
            # exlude neighbors cf IRAF's continuum parameter 'grow'
            if grow > 0:
                for sl in np.ma.clump_masked(mspec1):
                    for ii in range(sl.start-grow, sl.start):
                        if ii >= 0:
                            mspec1.mask[ii] = True
                    for ii in range(sl.stop+1, sl.stop+grow+1):
                        if ii < len(mspec1):
                            mspec1.mask[ii] = True
            # stop rejection process when min_points is reached
            # [opt] [not in IRAF]
            if np.ma.count(mspec1) < min_points:
                if verbose:
                    print("  min_points %d reached" % min_points)
                break
            mspec = mspec1
            if function == 'polynomial':
                coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
                cont = np.poly1d(coeff)(wav)
            elif function == 'spline3':
                knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
                spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
                cont = sint.splev(wav, spl)
            else:
                raise(AttributeError)
    # compute residual and rms
    res = fspec-cont
    sigm = np.std(res[~mspec.mask])
    if verbose:
        print("  nfit=%d/%d" %  (np.ma.count(mspec), len(mspec)))
        print("  fit rms=%.3e" %  sigm)
    # compute residual and rms between original spectrum and model
    # different from above when median filtering is applied
    ores = spec-cont
    osigm = np.std(ores[~mspec.mask])
    if int(med_filt) and verbose:
        print("  unfiltered rms=%.3e" %  osigm)
    # plot fit results
    if plot_fit:
        # overplot spectrum and model + mark rejected points
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(wav[~mspec.mask], spec[~mspec.mask],
            c='tab:blue', lw=1.0)
        # overplot median filtered spectrum
        if int(med_filt):
            ax1.plot(wav[~mspec.mask], fspec[~mspec.mask],
                c='tab:cyan', lw=1.0)
        ax1.scatter(wav[mspec.mask], spec[mspec.mask], s=20., marker='d',
        edgecolors='tab:gray', facecolors='none', lw=0.5)
        ax1.plot(wav, cont, ls='--', c='tab:orange')
        if nit > 0:
            # plot residuals and rejection thresholds
            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.axhline(0., ls='--', c='tab:orange', lw=1.)
            ax2.axhline(-rej_low*sigm, ls=':')
            ax2.axhline(rej_high*sigm, ls=':')
            ax2.scatter(wav[mspec.mask], res[mspec.mask],
                s=20., marker='d', edgecolors='tab:gray', facecolors='none',
                lw=0.5)
            ax2.scatter(wav[~mspec.mask], ores[~mspec.mask],
                marker='o', s=10., edgecolors='tab:blue', facecolors='none',
                lw=.5)
            # overplot median filtered spectrum
            if int(med_filt):
                ax2.scatter(wav[~mspec.mask], res[~mspec.mask],
                    marker='s', s=5., edgecolors='tab:cyan', facecolors='none',
                    lw=.2)
        if xlabel != "" :
            plt.xlabel(xlabel)
        if ylabel != "" :
            plt.ylabel(ylabel)
        plt.show()
    return cont



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


def sigma_clip_remove_bad_columns(template, nsig=2.0, plot=False) :
    
    template_sigma = np.nanstd(np.abs(template["flux_residuals"]),axis=0)

    good_wl_channels = template_sigma < nsig * template["fluxerr_model"]

    clean_template = {}
    clean_template["fit"] = template["fit"]
    clean_template["median"] = template["median"]
    clean_template["subtract"] = template["subtract"]
    clean_template["pfilename"] = template["pfilename"]
    
    clean_template["shift"] = template["shift"]
    clean_template["scale"] = template["scale"]
    clean_template["quadratic"] = template["quadratic"]

    flux_arr = np.full_like(template["flux_arr"], np.nan)
    flux_arr_sub = np.full_like(template["flux_arr_sub"], np.nan)
    flux_residuals = np.full_like(template["flux_residuals"], np.nan)
    snr = np.full_like(template["snr"], np.nan)

    for i in range(len(template["flux_residuals"])) :
        flux_arr[i][good_wl_channels] = template["flux_arr"][i][good_wl_channels]
        flux_arr_sub[i][good_wl_channels] = template["flux_arr_sub"][i][good_wl_channels]
        flux_residuals[i][good_wl_channels] = template["flux_residuals"][i][good_wl_channels]
        snr[i][good_wl_channels] = template["snr"][i][good_wl_channels]
        if plot :
            plt.plot(template["wl"][good_wl_channels], template["flux_residuals"][i][good_wl_channels], '.', color='g')
            plt.plot(template["wl"][~good_wl_channels], template["flux_residuals"][i][~good_wl_channels], '.', color='r')

    clean_template["fluxerr_model"] = np.full_like(template["fluxerr_model"], np.nan)
    clean_template["flux"] = np.full_like(template["flux"], np.nan)
    clean_template["fluxerr"] = np.full_like(template["fluxerr"], np.nan)
    clean_template["wl"] = template["wl"]

    clean_template["fluxerr_model"][good_wl_channels] =  template["fluxerr_model"][good_wl_channels]
    clean_template["flux"][good_wl_channels] = template["flux"][good_wl_channels]
    clean_template["fluxerr"][good_wl_channels] = template["fluxerr"][good_wl_channels]

    clean_template["flux_arr"] = flux_arr
    clean_template["flux_arr_sub"] = flux_arr_sub
    clean_template["flux_residuals"] = flux_residuals
    clean_template["snr"] =  snr

    if plot :
        plt.plot(template["wl"], nsig * template["fluxerr_model"], '-')
        plt.show()

    return clean_template


def normalize_spectra(spectra, template, fluxkey="fluxes", fluxerrkey="fluxerrs", plot=False) :
    
    continuum_fluxes = []
    
    for order in range(len(template)) :
        order_template = template[order]

        wl = order_template["wl"]
        flux = order_template["flux"]
        fluxerr = order_template["fluxerr"]
        
        nanmask = np.isfinite(flux)
        nanmask &= np.isfinite(wl)

        if len(flux[nanmask]) > 10 :
            continuum = fit_continuum(wl, flux, function='polynomial', order=3, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux", plot_fit=False, silent=True)
        else :
            continuum = np.full_like(wl, np.nan)
        
        if plot :
            plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', lw=0.3, alpha=0.3, zorder=1)
            #plt.scatter(wl, flux, marker='o', s=10., edgecolors='tab:blue', facecolors='none', lw=.5)
            plt.plot(wl, continuum, '-', lw=2, zorder=2)
    
        for i in range(spectra['nspectra']) :
            spectra[fluxkey][order][i] /= continuum
            spectra[fluxerrkey][order][i] /= continuum
        
        template[order]["flux"] /= continuum
        template[order]["fluxerr"] /= continuum
        template[order]["fluxerr_model"] /= continuum
        
        for j in range(len(template[order]["flux_arr"])) :
            template[order]["flux_arr"][j] /= continuum
            template[order]["flux_residuals"][j] /= continuum

        continuum_fluxes.append(continuum)

    if plot :
        plt.show()

    spectra["continuum_fluxes"] = np.array(continuum_fluxes, dtype=float)

    return spectra, template


def calculate_weights(spectra, template, normalize_weights=True, use_err_model=True, plot=False) :

    weights = []

    normfactor = []

    for order in range(len(template)) :
        order_template = template[order]
        
        wl = order_template["wl"]
        flux = order_template["flux"]

        if use_err_model :
            fluxerr = order_template["fluxerr_model"]
        else :
            fluxerr = order_template["fluxerr"]

        nanmask = np.isfinite(flux)
        nanmask &= np.isfinite(fluxerr)
        nanmask &= flux > 0
        
        loc_weights = np.full_like(fluxerr, np.nan)
        
        loc_weights[nanmask] = 1. / (fluxerr[nanmask] * fluxerr[nanmask])

        weights.append(loc_weights)
        
        if len(loc_weights[nanmask]) :
            normfactor.append(np.nanmedian(loc_weights[nanmask]))
        else :
            normfactor.append(np.nan)

    normfactor = np.array(normfactor)

    if normalize_weights :
        median_norm_factor = np.nanmax(normfactor)
    else :
        median_norm_factor = 1.0

    for order in range(len(template)) :
        
        order_template = template[order]
        
        wl = order_template["wl"]
        flux = order_template["flux"]

        weights[order] /= median_norm_factor
        
        if plot :
            
            plt.ylim(-0.5,3.0)
            
            plt.scatter(wl, weights[order], marker='o', s=10., edgecolors='tab:red', facecolors='none', lw=.5)
            
            plt.plot(wl, flux, '-')

    if plot :
        plt.show()

    spectra["weights"] = np.array(weights, dtype=float)

    return spectra


def get_fraction_of_nans(spectra, label="", fluxkey="fluxes", fluxerrkey="fluxerrs", debug=False, verbose=False, plot=False) :

    nnans = 0
    npixs = 0
    
    fnans, orders = np.array([]), np.array([])
    vel_diff, orig_dv = np.array([]), np.array([])
    
    for order in range(49) :
        nnans_per_order = 0
        npix_per_order = 0
        
        vel_diff_loc = np.array([])
        orig_dv_loc = np.array([])
        for i in range(spectra['nspectra']) :
            finitemask = np.isfinite(spectra[fluxkey][order][i])*np.isfinite(spectra[fluxerrkey][order][i])

            nnans_per_order += len(spectra[fluxkey][order][i][~finitemask])
            npix_per_order += len(spectra[fluxkey][order][i])
        
            vels = spectra["vels"][order][i][finitemask]
            vels_orig = spectra["vels"][order][i]
        
            vel_diff_loc = np.append(vel_diff_loc, np.nanmedian(np.abs(vels[1:] - vels[:-1])))
            orig_dv_loc = np.append(orig_dv_loc, np.nanmedian(np.abs(vels_orig[1:] - vels_orig[:-1])))
    
        vel_diff = np.append(vel_diff, np.nanmedian(vel_diff_loc) )
        orig_dv = np.append(orig_dv, np.nanmedian(orig_dv_loc) )

        if debug :
            print("Order={0} Npix={1} NNaNs={2} fraction={3:.2f}%".format(order,npix_per_order,nnans_per_order,100. * nnans_per_order/npix_per_order))
        fnans = np.append(fnans, 100. * nnans_per_order/npix_per_order)
        orders = np.append(orders, order)

        npixs += npix_per_order
        nnans += nnans_per_order

    if plot :
        plt.plot(orders, fnans, "bo")
        plt.plot(orders, fnans, "b-")
        plt.xlabel("order number")
        plt.ylabel("fraction of NaNs (%)")
        plt.show()
    
    loc = {}
    loc["nnans"] = nnans
    loc["npixs"] = npixs
    loc["totfnans"] = nnans/npixs
    loc["fnans"] = fnans
    loc["orders"] = orders
    loc["dv"] = vel_diff
    loc["orig_dv"] = orig_dv

    if verbose :
        print("Fraction of NaNs {0}: {1} / {2} = {3:.2f} %  dv = {4:.2f} km/s".format(label, loc["nnans"], loc["npixs"], loc["totfnans"]*100, np.nanmedian(vel_diff)))

    return loc

