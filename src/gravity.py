"""
gravitational accel models    
"""

import numpy as np
from .const import MU_SUN, MU_EARTH
from .earth import earth_position

def sun_gravity(r):
    """
    grav accel due to the sun
    param:
    r = ndarray, shape(3,)
        pos vector (au)
    
    returns:
    a = ndarray
        accel vector (au/day^2)    
    """
    
    norm_r = np.linalg.norm(r)
    return -MU_SUN * r / norm_r**3

def get_total_accel(r, t):
    """
    total count for all accel also containing distractions from main planets
    
    param:
    r = ndarray
        object pos vector (au)
    t = float
        time (day)
    """
    
    # main accel from sun and earth distraction
    accel = sun_gravity(r)
    r_earth = earth_position(t)
    accel += _calculate_perturbation(r, r_earth, MU_EARTH)
    
    # this was only focusing on sun and earth first, can add more later perturbations from other planets
    
    return accel

def _calculate_perturbation(r_obj, r_planet, mu_planet):
    """
    third-body gravitational perturbation formula
    """
    r_rel = r_obj - r_planet
    dist_rel = np.linalg.norm(r_rel)
    dist_planet = np.linalg.norm(r_planet)
    
    # direct and indirect component (coor. center)
    term1 = -mu_planet * r_rel / dist_rel**3 # direct
    term2 = mu_planet * r_planet / dist_planet**3 # indirect
    
    return term1 + term2
