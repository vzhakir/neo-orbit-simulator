"""
gravitational accel models    
"""

import numpy as np
from .const import MU_SUN

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