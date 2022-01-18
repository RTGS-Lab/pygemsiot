'''
Copyright University of Minnesota 2020
Authors: Mohana Krishna, Bryan C. Runck
'''

import math


# Formulas specified here can be found in the following document:
# https://www.mesonet.org/images/site/ASCE_Evapotranspiration_Formula.pdf
# Page number of each formula is supplied with each function.

def get_delta(temp):
    """
    Reference page number: 28-29
    Parameters
    ------------------------------
    temp: (``float``)
        The air temperature in degrees Celcius
    Returns
    ------------------------------
    delta: (``float``)
        The slope of the saturation vapor pressure-temperature curve in kPa/C
    """
    numerator = 2503 * math.exp((17.27 * temp) / (temp + 237.3))
    denominator = math.pow(temp + 237.3, 2)
    delta = numerator / denominator
    return delta


def get_flux_density(r_n_metric, r_n, os):
    """
    Reference page number: 44
    Currently, nighttime is defined as solar radiation values less than or equal to 5
    Parameters
    ------------------------------
    r_n_metric: (``float``)
        Solar radiation in W/m^2
    r_n: (``float``)
        Solar radiation in MJ/hm2
    os: (``bool``)
        Boolean which indicates whether to calculate G for short reference or tall reference
    Returns
    ------------------------------
    G: (``float``)
        Soil heat flux density MJ/m^2 h
    """
    G = None
    daytime = r_n_metric > 5

    if os:
        if daytime:
            G = 0.1 * r_n
        else:
            G = 0.5 * r_n
    else:
        if daytime:
            G = 0.04 * r_n
        else:
            G = 0.2 * r_n
    return G


def get_gamma(p):
    """
    Reference page number: 28
    Parameters
    ------------------------------
    p: (``float``)
        Barometric pressure in kPa
    Returns
    ------------------------------
    gamma: (``float``)
        Gamma (psychrometric constant) in kPa/C
    """
    gamma = 0.000665 * p
    return gamma


def get_cn(r_n_metric, os):
    """
    Reference page number: 5
    Parameters
    ------------------------------
    r_n_metric: (``float``)
        Solar radiation in W/m^2
    os: (``bool``)
        Boolean which indicates whether to calculate G for short reference or tall reference
    Returns
    ------------------------------
    cn: (``int``)
        Numerator constant
    """
    cn = None
    daytime = r_n_metric > 5
    if os:
        if daytime > 5:
            cn = 37
            return cn
        else:
            cn = 37
            return cn
    else:
        if daytime > 5:
            cn = 66
            return cn
        else:
            cn = 66
            return cn


def get_cd(r_n_metric, os):
    """
    Reference page number: 5
    Parameters
    ------------------------------
    r_n_metric: (``float``)
        Solar radiation in W/m^2
    os: (``bool``)
        Boolean which indicates whether to calculate G for short reference or tall reference
    Returns
    ------------------------------
    cd: (``float``)
        Denominator constant
    """
    cd = None
    daytime = r_n_metric > 5

    if os:
        if daytime > 5:
            cd = 0.24
            return cd
        else:
            cd = 0.96
            return cd
    else:
        if daytime > 5:
            cd = 0.25
            return cd
        else:
            cd = 1.7
            return cd


def get_es(temp):
    """
    Reference page number: 29
    Parameters
    ------------------------------
    temp: (``float``)
        Air temperature in degrees Celcius
    Returns
    ------------------------------
    es: (``float``)
        The saturation vapour pressure
    """
    es = 0.6108 * math.exp((17.27 * temp) / (temp + 237.3))
    return es


def get_ea(temp, rh):
    """
    Reference page number: 31-32
    Parameters
    ------------------------------
    temp: (``float``)
        Air temperature in degrees Celcius
    rh: (``float``)
        Relative humidity
    Returns
    ------------------------------
    ea: (``float``)
        The actual vapour pressure
    """
    es = get_es(temp)
    ea = (rh / 100) * es
    return ea


def solar_rad_metric_to_campbell(rad):
    """
    Parameters
    ------------------------------
    rad: (``float``)
        Solar radiation in W/m2
    Returns
    ------------------------------
    campbell_rad: (``float``)
        Solar radiation in MJ/hm2
    """
    campbell_rad = rad * (3600 / math.pow(10, 6))
    return campbell_rad


def solar_rad_campbell_to_metric(rad):
    """
    Parameters
    ------------------------------
    rad: (``float``)
        Solar radiation in MJ/hm2
    Returns
    ------------------------------
    metric_rad: (``float``)
        Solar radiation in W/m2
    """
    metric_rad = rad * (math.pow(10, 6) / 3600)
    return metric_rad
