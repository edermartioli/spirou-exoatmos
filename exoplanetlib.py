# -*- coding: iso-8859-1 -*-
"""
    Created on Jul 06 2020
    
    Description: python library for exoplanetary quantities
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import exoplanet as xo
import json

import batman

import os

from astropy.utils.data import download_file
from astropy.io import ascii
import astropy.units as u
from astropy.time import Time

from scipy import interpolate
from scipy import constants

import matplotlib.pyplot as plt

__all__ = ['PlanetParams']

# Set global variables
EXOPLANETS_CSV_URL = 'http://exoplanets.org/csv-files/exoplanets.csv'
EXOPLANETS_TABLE = None
PARAM_UNITS = None
TIME_ATTRS = {'TT': 'jd', 'T0': 'jd'}
BOOL_ATTRS = ('ASTROMETRY', 'BINARY', 'EOD', 'KDE', 'MICROLENSING', 'MULT',
              'SE', 'TIMING', 'TRANSIT', 'TREND')

def exoplanets_table(cache=True, show_progress=True):
    global EXOPLANETS_TABLE

    if EXOPLANETS_TABLE is None:
        table_path = download_file(EXOPLANETS_CSV_URL, cache=cache,
                                   show_progress=show_progress)
        EXOPLANETS_TABLE = ascii.read(table_path)

        # Store column of lowercase names for indexing:
        lowercase_names = [i.lower() for i in EXOPLANETS_TABLE['NAME'].data]
        EXOPLANETS_TABLE['NAME_LOWERCASE'] = lowercase_names
        EXOPLANETS_TABLE.add_index('NAME_LOWERCASE')

    return EXOPLANETS_TABLE


def parameter_units():
    """
    Dictionary of exoplanet parameters and their associated units.
    """
    global PARAM_UNITS

    if PARAM_UNITS is None:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        units_file = open(os.path.join(pkg_dir, 'exoplanets', 'param_units.json'))
        PARAM_UNITS = json.load(units_file)

    return PARAM_UNITS


class PlanetParams(object):
    """
    Exoplanet system parameters.
    Caches a local copy of the http://exoplanets.org table, and queries
    for a planet's properties. Unitful quantities are returned whenever
    possible.
    """
    def __init__(self, exoplanet_name, cache=True, show_progress=True):
        """
        Parameters
        ----------
        exoplanet_name : str
            Name of exoplanet (case insensitive)
        cache : bool (optional)
            Cache exoplanet table to local astropy cache? Default is `True`.
        show_progress : bool (optional)
            Show progress of exoplanet table download (if no cached copy is
            available). Default is `True`.
        """
        # Load exoplanets table, corresponding units
        table = exoplanets_table(cache=cache, show_progress=show_progress)
        param_units = parameter_units()

        if not exoplanet_name.lower().strip() in table['NAME_LOWERCASE'].data:
            raise ValueError('Planet "{0}" not found in exoplanets.org catalog')

        row = table.loc[exoplanet_name.lower().strip()]
        for column in row.colnames:
            value = row[column]

            # If param is unitful quantity, make it a `astropy.units.Quantity`
            if column in param_units:
                parameter = u.Quantity(value, unit=param_units[column])

            # If param describes a time, make it a `astropy.time.Time`
            elif column in TIME_ATTRS:
                parameter = Time(value, format=TIME_ATTRS[column])

            elif column in BOOL_ATTRS:
                parameter = bool(value)

            # Otherwise, simply set the parameter to its raw value
            else:
                parameter = value

            # Attributes should be all lowercase:
            setattr(self, column.lower(), parameter)

    def __repr__(self):
        return ('<{0}: name="{1}" from exoplanets.org>'
                .format(self.__class__.__name__, self.name))


def exoplanet(exoplanet) :
    loc = {}
    
    exoplanet = PlanetParams(exoplanet)
    
    loc['params'] = exoplanet
    
    if exoplanet.transit :
        orbit = xo.orbits.KeplerianOrbit(period=exoplanet.per.value, t0=exoplanet.tt.jd, incl=exoplanet.i.value, ecc=exoplanet.ecc.value, omega=exoplanet.om.value, m_planet=exoplanet.mass.value, m_star=exoplanet.mstar.value, r_star=exoplanet.rstar.value)
    
        loc['Ks'], loc['Kp'] = rvstar_semi_amplitude(exoplanet.per.value, exoplanet.mass.value, exoplanet.mstar.value, inc=exoplanet.i.value, ecc=exoplanet.ecc.value)

    else :
        orbit = xo.orbits.KeplerianOrbit(period=exoplanet.per.value, t_periastron=exoplanet.t0.jd, ecc=exoplanet.ecc.value, omega=exoplanet.om.value, m_planet=exoplanet.msini.value, m_star=exoplanet.mstar.value, r_star=exoplanet.rstar.value)
        
        loc['Ks'], loc['Kp'] = rvstar_semi_amplitude(exoplanet.per.value, exoplanet.msini.value, exoplanet.mstar.value, ecc=exoplanet.ecc.value)

    loc['orbit'] = orbit

    albedo = 0.1 # geometric albedo
    f = 1/2 # uniform heat redistribution
    #f = 2/3 # no heat redistribution

    loc['albedo'] = albedo
    loc['f'] = f

    Teq = exoplanet.teff.value * np.sqrt((exoplanet.rstar.value / 215.032) * f / exoplanet.a.value) * (1.0 - albedo)**(0.25)

    loc['Teq'] = Teq * u.K

    return loc


def calculate_transit_window(planet, bjd, exptime=0., verbose=False, plot_flux=False, plot=False) :
    
    loc = {}
    
    # Save exposure time
    # Note (E. Martioli 14/09/2020) in future versions exposure time
    # should be used to calculate the integrated flux over integration
    # for now it's just used in the plot, so one can see the time span
    # of each exposure
    
    loc["exptime"] = exptime
    loc["exptimes"] = np.full_like(bjd,exptime/(24.*60*60))
    
    planet_params = planet['params']
    
    ### See https://www.cfa.harvard.edu/~lkreidberg/batman/
    ### Init parameters
    params  = batman.TransitParams()
    
    params.rp        = planet_params.r.value / (planet_params.rstar.value / 0.10049)
    params.inc       = planet_params.i.value
    params.t0        = planet_params.tt.jd
    params.a         = planet_params.ar.value
    params.per       = planet_params.per.value
    params.ecc       = planet_params.ecc.value
    params.w         = planet_params.om.value
    
    #### Here I have to find a tool to retrieve the limb darkening
    #### either from a database or from a good empirical model (e.g. Claret)
    
    params.limb_dark = "nonlinear"      #limb darkening model
    params.u         = [0.7601,-0.5528,0.9838,-0.4226]        #Hayek+2012

    p = planet_params.per.value

    ##############################
    ## Create highly sampled model
    ##############################
    # Build a high sampling model for plotting and for determining tini,tcen,tend
    timespan = bjd[-1] - bjd[0]
    t0 = bjd[0] - timespan
    tf = bjd[-1] + timespan
    
    # create time array with 0.5 s sampling and wider coverage
    hs_times = np.arange(t0, tf, 1/(2*24*60*60))
    
    # create model
    highSamplingTransitModel = batman.TransitModel(params, hs_times)
    high_sampling_flux = highSamplingTransitModel.light_curve(params)
    high_sampling_phases = (hs_times - planet_params.tt.jd + 0.5 * p) % p - 0.5 * p
    high_sampling_window = (1.0 - high_sampling_flux) / np.max(1.0 - high_sampling_flux)
    
    in_mask = high_sampling_flux < 1.
    hsindices = np.where(in_mask)[0]
    tini = hs_times[hsindices[0]]
    tend = hs_times[hsindices[-1]]
    tcen = tini + (tend - tini)/2
    
    loc["tini"] = tini
    loc["tend"] = tend
    loc["tcen"] = tcen
    
    loc["phaseini"] = (tini - planet_params.tt.jd + 0.5 * p) % p - 0.5 * p
    loc["phaseend"] = (tend - planet_params.tt.jd + 0.5 * p) % p - 0.5 * p
    
    loc['window_func'] = interpolate.interp1d(hs_times,high_sampling_window,kind='linear')

    # end of highly sampled model
    ##############################

    ### Create transit light curve model from input parameters
    TransitModel = batman.TransitModel(params, bjd)
    
    ### Build window function
    flux = TransitModel.light_curve(params)
    window = (1.0 - flux) / np.max(1.0 - flux)

    phases = (bjd - planet_params.tt.jd + 0.5 * p) % p - 0.5 * p

    loc["params"] = params
    loc["TransitModel"] = TransitModel
    
    loc["flux"] = flux
    loc["window"] = window
    loc["phases"] = phases
    
    # create mask for all data
    loc["all"] = bjd > 0

    #in_transit = window > 0
    in_transit = bjd > loc["tini"] - loc["exptimes"]/2
    in_transit &= bjd < loc["tend"] + loc["exptimes"]/2
    loc["in_transit"] = in_transit
    
    #out_transit = window == 0
    out_transit = bjd <= loc["tini"] - loc["exptimes"]/2
    out_transit ^= bjd >= loc["tend"] + loc["exptimes"]/2
    loc["out_transit"] = out_transit

    indices = np.where(in_transit)[0]
    ind_ini=indices[0]
    ind_end=indices[-1]
    
    loc["ind_ini"] = ind_ini
    loc["ind_end"] = ind_end
    
    if plot :
        #plt.vlines((loc["phaseini"], 0, loc["phaseend"]), 0, 1, color = "r", ls=("--", ":", "--"))
        if plot_flux :
            plt.plot(high_sampling_phases, high_sampling_flux, "-", color="olive", lw=2)

            if exptime :
                plt.errorbar(phases, flux, xerr=loc["exptimes"]/2, fmt=".", color="C0", lw=2)
            else :
                plt.plot(phases, flux, "o", color="C0", lw=1)

            plt.ylabel("relative flux")
        else:
            plt.plot(high_sampling_phases, high_sampling_window, "-", color="olive", lw=2)

            if exptime :
                plt.errorbar(phases, window, xerr=loc["exptimes"]/2, fmt=".", color="C0", lw=2)
            else :
                plt.plot(phases, window, "o", color="C0", lw=1)

            plt.ylabel("relative flux")

        plt.xlabel("time [days]")
        plt.show()

    return loc

def rvstar_semi_amplitude(per, mp, mstar, inc=90, ecc=0.) :
    # per in days
    # mpsini and mp in jupiter mass
    # mstar in solar mass
    
    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = per * 24. * 60. * 60. # in s
    
    mjup = 1.898e27 # mass of Jupiter in Kg
    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    mp_kg = mp*mjup

    inc_rad = inc * np.pi/180. # inclination in radians
    
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p2 = np.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
    p3 = 1./np.sqrt(1 - ecc*ecc)
    
    ks = mp_kg * p1*p2*p3
    kp = mstar_kg * p1*p2*p3

    # return semi-amplitude in km/s
    return ks/1000., kp/1000.


def planet_rv(phases, kp):
    
    """
        Return planet radial velocities assuming circular orbit
        """
    rvs = kp * np.sin(2. * np.pi * phases)
    
    return rvs


def sample_of_exoplanets(min_mass=0., max_mass=100., min_per=0., max_per=10000., plot=False) :

    sample = {}

    table = exoplanets_table(cache=True, show_progress=True)
    
    mp, porb = [], []
    mp_out, porb_out = [], []
    
    for planet_name in table['NAME_LOWERCASE'] :
        
        try :
            planet = exoplanet(planet_name)
            if planet['params'].transit == True :
                if min_mass < planet['params'].mass.value < max_mass and min_per < planet['params'].per.value < max_per :
                
                    porb.append(planet['params'].per.value)
                    mp.append(planet['params'].mass.value)
                    #planet['params'].mstar.value
                    #planet['params'].rstar.value
                    sample[planet_name] = planet
                    #print("Selected planet: {0}".format(planet_name))
                else :
                    porb_out.append(planet['params'].per.value)
                    mp_out.append(planet['params'].mass.value)
        except:
            #print("WARNING: cannot read exoplanet {}, skipping ...".format(planet_name))
            continue
                
    mp, porb = np.array(mp), np.array(porb)
    mp_out, porb_out = np.array(mp_out), np.array(porb_out)
    
    if plot :
        plt.hlines([min_mass,max_mass], [min_per,min_per], [max_per,max_per], colors="k", linestyles='--')
        plt.vlines([min_per,max_per], [min_mass,min_mass], [max_mass,max_mass], colors="k", linestyles='--')
        plt.loglog(porb, mp, "o", color="blue", label="selected sample")
        plt.loglog(porb_out, mp_out, "o", color="orange", label="excluded sample")
        plt.ylabel(r"Mass [MJup]")
        plt.xlabel(r"Orbital period [d]")
        plt.legend()
        plt.show()
                    
    return sample
