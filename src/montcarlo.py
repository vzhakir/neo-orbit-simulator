"""
monte carlo simulation module
used for uncertainty of NEO orbits analysis
"""

import numpy as np
from .orbit_elem import orbit_elem_to_statvec
from .integrator import prop_orbit
from .gravity import get_total_accel

def run_montecarlo(base_elements, sigmas, iter=50, t_span=(0,365), dt=1.0):
    """
    monte carlo simulation for object position distribution prediction
    
    param:
    base_elements: list
        init orbital elements [a,e,i,raan,argp,M]
    sigmas: list
        std dev for each orbit element
    iter: int
        total of monte carlo iterations
    t_span: tuple
        time span for simulation (day)
    dt: float
        time step (day)
        
    return:
    results: list
        list of last position of every iteration
    """
    
    all_finalpos = []
    print(f"running monte carlo sim with {iter} iterations...")
    for i in range(iter):
        # add gaussian noise to each orbit element
        noisy_elements = [
            np.random.normal(val,s) if s > 0 else val
            for val,s in zip(base_elements,sigmas)
        ]
        # convert random element to state vectors
        r0,v0 = orbit_elem_to_statvec(*noisy_elements)
        # propagate orbit with gravitation disturbances
        # using get_total_accel for calc planet pos for each time
        _, positions, _ = prop_orbit(r0, v0, get_total_accel, t_span, dt)
        # save last post for dist analysis (or cloud point)
        all_finalpos.append(positions[-1])
        
        if (i+1) % 10 == 0:
            print(f"  completed {i+1} / {iter} iterations")
        return np.array(all_finalpos)
    
def analyze_risk(final_pos, target_pos, threshold_au):
    """
    prob calc for obj that comes within certain distance from target
    """
    distances = np.linalg.norm(final_pos - target_pos, axis=1)
    hits = np.sum(distances < threshold_au)
    prob = (hits / len(final_pos)) * 100.0
    return prob, distances
