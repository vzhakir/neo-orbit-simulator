"""
    this module contains physical constants used in the simulation.
    
    units:
    distance = au
    time = day
    mass = solar mass
"""

# gaussian gravitational constant (au^1.5/day)
K_GAUSS = 0.01720209895

# gravitational constant in au^3/day
MU_SUN = K_GAUSS**2
MU_JUPITER = MU_SUN * 9.5479193842e-4
MU_VENUS = MU_SUN * 2.4478383e-6
MU_MARS = MU_SUN * 3.2271514e-7

# earth radius
R_EARTH = 4.26352e-5 # au
R_SUN = 0.00465047 # au
