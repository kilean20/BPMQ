import re
import io
import os
import warnings
import contextlib
from typing import List, Optional, Tuple, Dict, Union, Callable
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

def _warn(message, *args, **kwargs):
    return 'warning: ' +str(message) + '\n'
#     return _warn(x,stacklevel=2)  

warnings.formatwarning = _warn
def warn(x):
    return warnings.warn(x)

@contextlib.contextmanager    
def suppress_outputs():
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

_name_conversions =(
    (':PSQ_' , ':Q_'  ),
    (':PSQ_' , ':QV_' ),
    (':PSQ_' , ':QH_' ),
    (':PSC2_', ':DCH_'),
    (':PSC1_', ':DCV_'),
)

def calculate_Brho(E_MeV_u: float, mass_number: float, charge_number: float, **kwargs) -> float:
    c = 299792458  # Speed of light in m/s
    Ma = 931494320  # Atomic mass unit in eV/c^2
    E = E_MeV_u * 1e6 + Ma  # Total energy in eV
    p = mass_number * np.sqrt(E**2 - Ma**2) / c  # Momentum p in eV/c
    return p / charge_number  # Brho

# Calculate beta-gamma
def calculate_betagamma(E_MeV_u: float, mass_number: float, **kwargs) -> float:
    c = 299792458  # Speed of light in m/s
    Ma = 931494320  # Atomic mass unit in eV/c^2
    E = E_MeV_u * 1e6 + Ma  # Total energy in eV
    return np.sqrt(E**2 - Ma**2) / Ma  # Beta-gamma
    
# Exponential linear unit (ELU) function
def elu(x: float) -> float:
    return x if x > 0 else np.exp(x) - 1


# Calculate cyclic distance
def cyclic_distance(x: float, y: float, Lo: float, Hi: float) -> float:
    x_ang = 2 * np.pi * (x - Lo) / (Hi - Lo)
    y_ang = 2 * np.pi * (y - Lo) / (Hi - Lo)
    return np.arccos(np.cos(y_ang - x_ang)) / np.pi * 0.5 * (Hi - Lo)

# Calculate cyclic mean
def cyclic_mean(x: List[float], Lo: float, Hi: float) -> float:
    x_ = np.array(x)
    if x_.ndim == 1 and len(x_) < 1:
        return x_
    mean = np.mod(np.angle(np.mean(np.exp(1j * 2 * np.pi * (x_ - Lo) / (Hi - Lo)), axis=0)), 2 * np.pi) / (2 * np.pi) * (Hi - Lo) + Lo
    return mean
    
# Calculate cyclic mean and variance
def cyclic_mean_var(x: List[float], Lo: float, Hi: float) -> Tuple[float, float]:
    x_ = np.array(x)
    if x_.ndim == 1 and len(x_) < 1:
        return x_, np.zeros(x_.shape)
    mean = cyclic_mean(x, Lo, Hi)
    return mean, np.mean(cyclic_distance(x, mean, Lo, Hi) ** 2)
    
# Calculate cyclic difference
def cyclic_difference(x: float, y: float, Lo: float, Hi: float) -> float:
    x_ang = 2 * np.pi * (x - Lo) / (Hi - Lo)
    y_ang = 2 * np.pi * (y - Lo) / (Hi - Lo)
    distance = cyclic_distance(x, y, Lo, Hi)
    return distance * np.sign(np.sin(y_ang - x_ang))

# Get the middle row of a DataFrame group
def get_middle_row_of_group(group: pd.DataFrame) -> pd.Series:
    mid_i = len(group) // 2
    return group.iloc[mid_i]

            
# Nelder-Mead optimization
def NelderMead(
    loss_ftn: Callable,
    x0: np.ndarray,
    simplex_size: float = 0.05,
    bounds: Optional[List[Tuple[float, float]]] = None,
    tol: float = 1e-4
) -> optimize.OptimizeResult:
    n = len(x0)
    initial_simplex = np.vstack([x0] * (n + 1))

    if bounds is None:
        for i in range(n):
            initial_simplex[i + 1, i] += simplex_size
    else:
        bounds = np.array(bounds)
        assert np.all(x0 <= bounds[:, 1]) and np.all(bounds[:, 0] <= x0)
        for i in range(n):
            dx = simplex_size * (bounds[i, 1] - bounds[i, 0])
            initial_simplex[i + 1, i] = np.clip(x0[i] + dx, bounds[i, 0], bounds[i, 1])

    result = optimize.minimize(
        loss_ftn, x0, method='Nelder-Mead', bounds=bounds, tol=tol,
        options={'initial_simplex': initial_simplex}
    )

    return result


# Check if input is a list of lists
def is_list_of_lists(input_list: Union[list, np.ndarray]) -> bool:
    if not isinstance(input_list, list):
        return False
    return all(isinstance(item, list) for item in input_list)


# Convert list of dicts to pandas DataFrame
def from_listdict_to_pd(data: List[Dict]) -> pd.DataFrame:
    all_keys = set().union(*data)
    dict_of_lists = {key: [d.get(key, np.nan) for d in data] for key in all_keys}
    max_length = max(len(lst) for lst in dict_of_lists.values())
    for key in dict_of_lists:
        dict_of_lists[key] += [np.nan] * (max_length - len(dict_of_lists[key]))
    return pd.DataFrame(dict_of_lists)


def get_Dnum_from_pv(pv: str) -> int or None:
    """
    Extracts the D number from a PV string.
    Args:
        pv (str): The PV string.
    Returns:
        int or None: The extracted D number or None if not found.
    """
    try:
        match = re.search(r"_D(\d{4})", pv)
        if match:
            return int(match.group(1))
        else:
            return None
    except AttributeError:
        return None
    

def split_name_field_from_PV(PV: str, 
                           return_device_name: bool =True) -> tuple:
    """
    Splits the PV into name and key components.

    Args:
        PV (str): The PV string.

    Returns:
        tuple: A tuple containing the name and key components.
    """
    # Find the index of the first colon
    first_colon_index = PV.find(':')

    if first_colon_index == -1:
        print(f"Name of PV: {PV} is not found")
        return None, None
    
    if return_device_name:
        for dev_name, phys_name in _name_conversions:
            PV = PV.replace(phys_name,dev_name)

    second_colon_index = PV.find(':', first_colon_index + 1)
    if second_colon_index != -1:
        return PV[:second_colon_index], PV[second_colon_index + 1:]
    else:
        return PV, None

    
def sort_by_Dnum(strings):
    """
    Sort a list of PVs by dnum.
    """
    # Define a regular expression pattern to extract the 4-digit number at the end of each string
    pattern = re.compile(r'\D(\d{4})$')

    # Define a custom sorting key function that extracts the 4-digit number using the regex pattern
    def sorting_key(s):
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        return 0  # Default value if no match is found

    # Sort the strings based on the custom sorting key
    sorted_strings = sorted(strings, key=sorting_key)
    return sorted_strings
    
    
# Calculate mismatch factor between two Courant-Snyder parameters
def calculate_mismatch_factor(cs_ref: Tuple[float, float, float], cs_test: Tuple[float, float, float]) -> float:
    alpha_ref, beta_ref, nemit_ref = cs_ref
    alpha, beta, nemit = cs_test
    gamma_ref = (1 + alpha_ref**2) / beta_ref
    gamma = (1 + alpha**2) / beta
    R = beta_ref * gamma + beta * gamma_ref - 2 * alpha_ref * alpha
    Mx = max(0.5 * (R + max(R**2 - 4, 0)**0.5), 1)**0.5 - 1
    return max((nemit / nemit_ref) - 1, (nemit_ref / nemit) -1) * Mx
    
    
# Generate ellipse points based on Courant-Snyder parameters
def generate_ellipse(alpha: float, beta: float, nemit: float, bg: float) -> np.ndarray:
    gamma = (1 + alpha**2) / beta
    cov_matrix = np.array([
        [nemit * beta, -nemit * alpha * 1e-3],
        [-nemit * alpha * 1e-3, nemit * gamma * 1e-6]
    ]) / bg * 1e6
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    ellipse = np.linalg.cholesky(cov_matrix).dot(circle)
    return ellipse
    

# Plot beam ellipse
def plot_beam_ellipse(alpha: float, beta: float, nemit: float, bg: float, direction: str = 'x',
                      ax: Optional[plt.Axes] = None, fig=None, **kwargs):
    ellipse = generate_ellipse(alpha, beta, nemit, bg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.3))
    ax.plot(ellipse[0, :], ellipse[1, :] * 1e3, **kwargs)
    ax.set_xlabel(f"{direction}  (mm)")
    ax.set_ylabel(f"{direction}' (mrad)")
    ax.grid(True)
    

# Plot beam ellipse from covariance matrix
def plot_beam_ellipse_from_cov(cov: np.ndarray, direction: str = 'x',
                               ax: Optional[plt.Axes] = None, fig=None, **kwargs):
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    cov = cov.copy()
    fail = True
    while fail:
        try:
            ellipse = np.linalg.cholesky(cov).dot(circle)
            fail = False
        except np.linalg.LinAlgError:
            cov[0, 0] += 1e-6
            cov[1, 1] += 1e-6

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.3))
    ax.plot(ellipse[0, :] * 1e3, ellipse[1, :] * 1e3, **kwargs)
    ax.set_xlabel(f"{direction}  (mm)")
    ax.set_ylabel(f"{direction}' (mrad)")
    ax.grid(True)