"""
module for orbit elements conversions
units:
a in au
angles in radians
output pos in au
output velo in au/day
"""

import numpy as np
from .const import MU_SUN

def solve_master(M,e,tol=1e-10,max_iter=100):
    """
    solve kepler's equation using newton-raphson method
    M = E - e sin(E)
    """
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e*np.cos(E)
        delta = -f / f_prime
        E += delta
        if abs(delta) < tol:
            break
    return E

def orbit_elem_to_statvec(a,e,i,raan,argp,M):
    """
    convert orbital elements to state vectors such as pos and velo
    param:
    a: float
        semi-major axis(au)
    e: float
        eccentricity
    i: float
        inclination (rad)
    argp: float
        argument of perigee (rad)
    M: float
        mean anomaly (rad)
    
    return:
    r : ndarray, shape(3,)
        pos vector (au)
    v : ndarray, shape(3,)
        velo vector (au/day)
    """
    
    # solve kepler's equation
    E = solve_master(M,e)
    # distance r
    r_norm = a * (1-e*np.cos(E))
    # position in orbit plane/pqw frame
    r_pqw = np.array([
        a * (np.cos(E)-e),
        a * np.sqrt(1-e**2) * np.sin(E),
        0.0
    ])
    
    # velo in orbit plane/pqw frame
    v_factor = np.sqrt(MU_SUN * a) / r_norm
    v_pqw = v_factor * np.array([
        -np.sin(E),
        np.sqrt(1-e**2) * np.cos(E),
        0.0
    ])
    
    # rotation matrix from pqw to ijk
    cos_0, sin_0 = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(argp), np.sin(argp)
    
    rz_0 = np.array([
        [cos_0, -sin_0, 0],
        [sin_0, cos_0, 0],
        [0, 0, 1]
    ])
    
    rx_i = np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i]
    ])
    
    rx_w = np.array([
        [cos_w, -sin_w, 0],
        [sin_w, cos_w, 0],
        [0, 0, 1]
    ])
    
    rotation = rz_0 @ rx_i @ rx_w
    r = rotation @ r_pqw
    v = rotation @ v_pqw
    
    return r,v