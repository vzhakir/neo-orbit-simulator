import numpy as np

def compute_distance(neo_positions: np.ndarray, earth_positions: np.ndarray) -> np.ndarray:
    """
    Compute distance between NEO and Earth at each timestep.

    Parameters
    ----------
    neo_positions : (N, 3) array
        NEO positions in AU
    earth_positions : (N, 3) array
        Earth positions in AU

    Returns
    -------
    distances : (N,) array
        Distance at each timestep in AU
    """
    neo_positions = np.asarray(neo_positions, dtype=float)
    earth_positions = np.asarray(earth_positions, dtype=float)

    if neo_positions.shape != earth_positions.shape:
        raise ValueError(f"Shape mismatch: neo {neo_positions.shape} vs earth {earth_positions.shape}")
    if neo_positions.ndim != 2 or neo_positions.shape[1] != 3:
        raise ValueError(f"Expected (N,3) arrays, got {neo_positions.shape}")

    diff = neo_positions - earth_positions
    distances = np.linalg.norm(diff, axis=1)
    return distances


def find_minimum_distance(times: np.ndarray, distances: np.ndarray):
    """
    Find minimum distance and the time when it occurs.

    Parameters
    ----------
    times : (N,) array
        Times in days
    distances : (N,) array
        Distances in AU

    Returns
    -------
    d_min : float
        Minimum distance (AU)
    t_min : float
        Time at closest approach (days)
    idx : int
        Index of closest approach
    """
    times = np.asarray(times, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if times.shape != distances.shape:
        raise ValueError(f"Shape mismatch: times {times.shape} vs distances {distances.shape}")

    idx = int(np.argmin(distances))
    return float(distances[idx]), float(times[idx]), idx
