"""
    numerical integration for orbit integration.
    unit: au, day, solar mass
"""

import numpy as np

def rk4_step(r, v, t, accel_func, dt):
    """
    perform a single runge-kutta 4th order step.
    param:
    r = ndarray, shape(3,)
        pos vector (au)
    v = ndarray, shape(3,)
        velo vector (au/day)
    accel_func : callable
        function that takes pos r and returns accel a
    dt : float
        time step (day)
        
    returns:
    r_new, v_new : ndarray
        with updated position and velocity
    """
    
    k1_r = v
    k1_v = accel_func(v)

    k2_r = v+0.5*dt*k1_v
    k2_v = accel_func(r+0.5*dt*k1_r,t+0.5*dt)
    
    k3_r = v+0.5*dt*k2_v
    k3_v = accel_func(r+0.5*dt*k2_r,t+ 0.5*dt)
    
    k4_r = v+dt*k3_v
    k4_v = accel_func(r+dt*k3_r,t+dt)
    
    r_new = r + (dt/6.0)*(k1_r+2.0*k2_r+2.0*k3_r+k4_r)
    v_new = v + (dt/6.0)*(k1_v+2.0*k2_v+2.0*k3_v+k4_v)
    return r_new, v_new

def prop_orbit(r0, v0, accel_func, t_span, dt):
    """
    propagate orbit using rk4 integrator.
    param:
    r0 = ndarray, shape(3,)
        initial pos vector (au)
    v0 = ndarray, shape(3,)
        initial velo vector (au/day)
    accel_func : callable
        function that takes pos r and returns accel a
    t_span : tuple
        total time to propagate (day)
    dt : float
        time step (day)
        
    returns:
    times, positions, velocities : ndarray
        arrays of position and velocity overtime (N, 3)
    """
    
    t0,tf = t_span
    times = np.arange(t0,tf+dt,dt)
    r = np.array(r0,dtype=float)
    v = np.array(v0,dtype=float)
    positions = []
    velocities = []
    
    for t in times:
        positions.append(r.copy())
        velocities.append(v.copy())
        r,v = rk4_step(r,v,t,accel_func,dt)
    
    return np.array(times), np.array(positions), np.array(velocities)

def calculate_distances(neo_positions, times):
    """
    calculate distance between NEO and earth over simulation period.
    """
    from .earth import earth_positions
    
    e_positions = earth_positions(times) # get earth pos at same times
    distances = np.linalg.norm(neo_positions - e_positions, axis=1) # euclidean
    return distances
