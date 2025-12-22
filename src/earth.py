"""
simple earth orbit model

assumptions?
circular orbit
radius = 1 au
period = 365.25 day
orbit in x-y plane
"""

import numpy as np

# orbit period and angular velo of earth
EARTH_PERIOD = 365.25
OMEGA_EARTH = 2.0 * np.pi / EARTH_PERIOD

def earth_position(t):
    """
    compute earth heliocentric position at time t
    
    param:
    t : float
        time(day)
        
    returns:
    r_earth : ndarray, shape(3,)
        earth pos vector (au)
    """
    x = np.cos(OMEGA_EARTH * t)
    y = np.sin(OMEGA_EARTH * t)
    z = 0.0
    
    return np.array([x,y,z])

def earth_positions(times):
    """
    compute earth heliocentric positions at array of times
    
    param:
    times : array-like
        times in days
        
    returns:
    positions : ndarray, shape(N,3)
        earth pos vectors (au)
    """
    
    return np.array([earth_position(t) for t in times])