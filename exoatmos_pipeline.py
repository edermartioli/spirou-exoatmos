# -*- coding: iso-8859-1 -*-
"""
    Created on July 5 2020
    
    Description: Pipeline to process exoplanet atmosphere detection in SPIRou data
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python exoatmos_pipeline.py --pattern=Data/HD189733/*e.fits --rvfile="/Users/eder/spirou-tools/spirou-exoatmos/Data/HD189733/HD189733.rdb" --exoplanet="HD 189733 b" --model_source="/Users/eder/spirou-tools/spirou-exoatmos/hd189733_atmoslib/db.json"    
    
    python exoatmos_pipeline.py --pattern=../Data/HD189733_transit-01/*e.fits --rvfile=../Data/HD189733_transit-01/HD189733_t1-K2mask-filtered_ccf.rdb --exoplanet="HD 189733 b" --model_source="/Users/eder/spirou-tools/spirou-exoatmos/hd189733_atmoslib/db.json" --mask_type="out_transit" -pv
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import numpy as np
import matplotlib.pyplot as plt
import spiroulib
import exoatmos_utils
import exoplanetlib
import glob

parser = OptionParser()
parser.add_option("-i", "--pattern", dest="pattern", help="Input data pattern",type='string',default="")
parser.add_option("-r", "--rvfile", dest="rvfile", help="RV file (.rdb)",type='string',default="")
parser.add_option("-e", "--exoplanet", dest="exoplanet", help="Input exoplanet identifier",type='string',default="")
parser.add_option("-s", "--model_source", dest="model_source", help="Input model source (either a .db or .fits filename)",type='string',default="")
parser.add_option("-t", "--mask_type", dest="mask_type", help="Stellar template mask type (all or out_transit)",type='string',default="out_transit")

parser.add_option("-m", action="store_true", dest="median_combine", help="verbose", default=False)
parser.add_option("-a", action="store_true", dest="airmass_detrend", help="verbose", default=False)
parser.add_option("-p", action="store_true", dest="pca_cleaning", help="verbose", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with exoatmos_pipeline.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input data pattern: ', options.pattern)
    print('Input exoplanet identifier: ', options.exoplanet)
    print('RV file (.rdb): ', options.rvfile)
    print('Input model source: ', options.model_source)
    print('Stellar template mask type: ', options.mask_type)

plot2D = False
# Read spirou data

# Make list of data given an input pattern (e.g. *.fits)
inputdata = sorted(glob.glob(options.pattern))

print("******************************")
print("STEP 1: Load SPIRou data ...")
print("******************************")
# Load data from list of input spectra
#spectra = spiroulib.load_array_of_1D_spectra(inputdata, verbose=False)
array_of_spectra = spiroulib.load_array_of_spirou_spectra(inputdata, flatten_data_arrays=False, rvfile=options.rvfile, remove_blaze=True, verbose=True, plot=False)
spectra = spiroulib.get_spectral_data(array_of_spectra, verbose=True)


print("******************************")
print("STEP 2: Load exoplanet parameters and calculate transit window function ...")
print("******************************")
# load exoplanet parameters
planet = exoplanetlib.exoplanet(exoplanet=options.exoplanet)
#sample_of_hotjupiters = exoplanetlib.sample_of_exoplanets(min_mass=0.2, max_mass=10., min_per=0.5, max_per=15., plot=True)

# Calculate transit window function
transit = exoplanetlib.calculate_transit_window(planet, spectra["bjd"], exptime=array_of_spectra['spectra'][0]['exptime'], verbose=False, plot_flux=False, plot=False)
print("Transit starts in exp {0} and ends in exp {1}".format(transit["ind_ini"],transit["ind_end"]))


print("******************************")
print("STEP 3: Reduce SPIRou spectra (remove star, telluric, sky, and instrumental effects) ...")
print("******************************")
#mask_type='all' or 'out_transit'
mask_type=options.mask_type
#mask_type='all'
template = exoatmos_utils.reduce_spectra(spectra, transit, mask_type=mask_type, fit_type="quadratic", nsig_clip=4.0, combine_by_median=options.median_combine, airmass_detrend=options.airmass_detrend, pca_cleaning=options.pca_cleaning, subtract=True, verbose=True, plot=False, output="")

# The process above may still retain some stellar signal, since it's not taking into
# account the rv shifts due to stellar movements with respect to the observer

print("******************************")
print("STEP 4: Running analysis on each species ...")
print("******************************")

HeI_analysis = False
if HeI_analysis :
    print("******************************")
    print("STEP 4.1: He I triplet ...")
    print("******************************")
    # generate HeI nIR triplet model
    HeI_model = exoatmos_utils.HeI_nIR_triplet_model(model_baseline=1.0, plot=False)
    
    # generate simulated data
    HeI_simulated_data = exoatmos_utils.simulate_data(HeI_model, template, spectra, transit, planet, model_baseline = 1.0, scale=0.1, plot=False, verbose=True)
    
    # plot simulated data
    exoatmos_utils.plot_2Dtime_series(HeI_simulated_data, spectra["bjd"], wl0=1083.0, wlf=1083.8, transit=transit)
    
    exoatmos_utils.plot_2Dtime_series(template, spectra["bjd"], wl0=1083.0, wlf=1083.8, transit=transit)
    
    # run ccf on simulated data
    ccf_HeI_simulated = exoatmos_utils.compute_ccf(HeI_model, HeI_simulated_data, spectra, transit, planet, v0=-100., vf=100., nvels=201, wl0=1082.0, wlf=1084., use_observed_rvs=True, ref_frame='star', plot=True)
    
    # run ccf on observed data
    ccf_HeI = exoatmos_utils.compute_ccf(HeI_model, template, spectra, transit, planet, v0=-100., vf=100., nvels=201, wl0=1082.0, wlf=1084., use_observed_rvs=True, ref_frame='planet', plot=True)
    
    kp_v0_ccf_HeI = exoatmos_utils.compute_ccf_kp_v0_map(HeI_model, template, spectra, transit, planet, wl0=1082.0, wlf=1084., kp_range=100., n_kp=20, v0_range=80., n_v0=121, use_observed_rvs=True, plot=True, verbose=True)
    #exit()
    #HeI_model = exoatmos_utils.run_HeI_analysis(template, spectra, planet, transit, model_baseline=1.0, verbose=True, plot=True)

H2O_analysis = True

if H2O_analysis :
    print("******************************")
    print("STEP 4.2: H2O ...")
    print("******************************")

    model = exoatmos_utils.retrieve_exoatmos_model(options.model_source, planet, species="H2O", abundance=-3.0, model_baseline=1., convolve_to_resolution=70000., scale=1., verbose=True, plot=False)

    # generate simulated data
    #simulated_data = exoatmos_utils.simulate_data(model, template, spectra, transit, planet, model_baseline=1.0, plot=False, verbose=False)

    # all-bands
    #wl0, wlf = 950., 2450.
    
    # Y-band
    #wl0, wlf = 968, 1084
    # J-band
    #wl0, wlf = 1163, 1327
    # H-band
    wl0, wlf = 1558, 1717
    #wl0, wlf = 1810, 1890
    # K-band
    #wl0, wlf = 2100., 2290.

    # plot simulated data
    #exoatmos_utils.plot_2Dtime_series(simulated_data, spectra["bjd"], wl0=wl0, wlf=wlf, transit=transit)
    
    # plot simulated data
    exoatmos_utils.plot_2Dtime_series(template, spectra["bjd"], wl0=wl0, wlf=wlf, transit=transit)
    
    # run CCF on simulated data
    #simul_ccf = exoatmos_utils.compute_ccf(model, simulated_data, spectra, transit, planet, v0=-40., vf=40., nvels=121, wl0=wl0, wlf=wlf, ref_frame='star', verbose=True, plot=True)

    # run CCF on observed data
    observ_ccf = exoatmos_utils.compute_ccf(model, template, spectra, transit, planet, v0=-40., vf=40., nvels=121, wl0=wl0, wlf=wlf, ref_frame='planet', verbose=True, plot=True)

    #simul_kp_v0_ccf = exoatmos_utils.compute_ccf_kp_v0_map(model, simulated_data, spectra, transit, planet, wl0=wl0, wlf=wlf, kp_range=100., n_kp=20, v0_range=80., n_v0=121, use_observed_rvs=True, plot=True, verbose=True)

    #observ_kp_v0_ccf = exoatmos_utils.compute_ccf_kp_v0_map(model, template, spectra, transit, planet, wl0=wl0, wlf=wlf, kp_range=100., n_kp=20, v0_range=80., n_v0=121, use_observed_rvs=True, plot=True, verbose=True)


# To do:

# Fix stellar cleaning by including RV shifts during sequence
# Optimize H2O analysis -- include noise, weight/mask regions
# Write analysis for other species CO2, CH4, CO, NH4, K, Na, etc
# Organize codes to obtain detectability
# Organize codes to calculate simulations for any planet in the catalogue
# Include simulations for Emission spectra
# Try to detect atmosphere features on other datasets

exit()

print("******************************")
print("CHECK STEP: Plot time series")
print("******************************")
plot_lines = False
if plot_lines :
    #plot He I line
    exoatmos_utils.plot_2Dtime_series(template, spectra["bjd"], wl0=1083.0, wlf=1083.8, transit=transit)
    #plot K strong lines
    exoatmos_utils.plot_2Dtime_series(template, spectra["bjd"], wl0=1168.8, wlf=1169.4, transit=transit)
    exoatmos_utils.plot_2Dtime_series(template, spectra["bjd"], wl0=1176.4, wlf=1177.6, transit=transit)
    # Carbon band heads
    exoatmos_utils.plot_2Dtime_series(template, spectra["bjd"], wl0=2290.0, wlf=2293.5, transit=transit)
