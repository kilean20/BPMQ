# Standard Libraries
# import re
# import os
import pickle
from datetime import datetime
import time
import warnings
from typing import List, Dict, Optional, Tuple, Callable, Any
from copy import deepcopy as copy
import concurrent
from dataclasses import dataclass, field

try:
    from IPython.display import display as _display
except ImportError:
    _display = print
def display(obj):
    try:
        _display(obj)
    except:
        print(obj)

# Third-Party Libraries
import numpy as np
import pandas as pd
from math import ceil
import torch
import matplotlib.pyplot as plt

# Local Libraries
from .torch_helper import run_torch_optimizer
from .construct_machineIO import Evaluator_wBPMQ
from .construct_machineIO import phantasy_fetch_data_orig as fetch_data
from .LinearControl import LinearControl
from .machine_portal_helper import get_MPelem_from_PVnames
from .utils import calculate_Brho, calculate_betagamma, sort_by_Dnum, calculate_mismatch_factor, calculate_MMD4D, \
                   plot_beam_ellipse_from_cov, plot_beam_ellipse, get_ISAAC_preset, proximal_ordered_init_sampler, select_n_most_distant_mmd4d_covs




# Ignore specific user warnings for tensor copying
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")

# Type Definitions
_dtype = torch.float32

_E_MeV_u = 130
_mass_number = 18
_charge_number = 8
_Brho = calculate_Brho(_E_MeV_u,_mass_number,_charge_number)
_bg = calculate_betagamma(_E_MeV_u,_mass_number)
_types = {'quadrupole':['QUAD','quad','quadrupole','Qaudrupole'],
          'drift':['Drift','drift','drif']}
# _xalpha,_xbeta,_xnemit = 0.0, 4.0, 0.15*1e-6
# _yalpha,_ybeta,_ynemit = 0.0, 4.0, 0.15*1e-6
_xalpha,_xbeta,_xnemit = 0.0, 5.0, 0.16*1e-6
_yalpha,_ybeta,_ynemit = 0.0, 5.0, 0.16*1e-6
_cs_ref = [_xalpha,_xbeta,_xnemit,_yalpha,_ybeta,_ynemit]

PSQ_D5501 = {"name":"BDS_BTS:PSQ_D5501",
             "type":"quadrupole",
             "B2"  : -8.847796226101002,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aper": 0.025}
DRIFT_D5502 = {"name":"BDS_BTS:DRIFT_D5502",
               "type":"drift",
               "L"   :0.489, 
               "aper": 0.1 }
PSQ_D5509 = {"name":"BDS_BTS:PSQ_D5509",
             "type":"quadrupole",
             "B2"  : 9.423499969160641,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aper": 0.025}
DRIFT_D5510 = {"name":"BDS_BTS:DRIFT_D5510",
               "type":"drift",
               "L"   :0.268080814, 
               "aper": 0.1} 
BPM_D5513 = {"name":"BDS_BTS:BPM_D5513",
             "type":"drift",
             "L"   : 0.145282 +0.705851186 +0.25710829999999996*10 +0.356979, 
             "aper": 0.1 }
PSQ_D5552 = {"name":"BDS_BTS:PSQ_D5552",
             "type":"quadrupole",
             "B2"  : -14.624756233974265,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aper": 0.025}
DRIFT_D5553 = {"name":"BDS_BTS:DRIFT_D5553",
               "type":"drift",
               "L"   :0.489, 
               "aper": 0.1}
PSQ_D5559 = {"name":"BDS_BTS:PSQ_D5559",
             "type":"quadrupole",
             "B2"  : 17.174836398354252,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aper": 0.025}
DRIFT_D5560 = {"name":"BDS_BTS:DRIFT_D5560",
               "type":"drift",
               "L"   :0.2195 + 0.234072, 
               "aper": 0.1 }
BPM_D5565 = {"name":"BDS_BTS:BPM_D5565",
             "type":"drift",
             "L"   : 0.145282, 
             "aper": 0.1}
PM_D5567 = {"name":"BDS_BTS:PM_D5567",
            "type":"drift",
            "L"   : 0.0, 
            "aper": 0.1}
            
BDS_dicts_f5501_t5567 = [
    PSQ_D5501,DRIFT_D5502,PSQ_D5509,DRIFT_D5510,BPM_D5513,
    PSQ_D5552,DRIFT_D5553,PSQ_D5559,DRIFT_D5560,BPM_D5565,PM_D5567
]


def noise2cs(
    noise: torch.Tensor,
    xalpha: float = _xalpha, xbeta: float = _xbeta, xnemit: float = _xnemit,
    yalpha: float = _yalpha, ybeta: float = _ybeta, ynemit: float = _ynemit
) -> torch.Tensor:
    """
    Convert batch of noise values into Twiss parameters (cs).
    """
    # x0, x1, x2, x3, x4, x5 = noise[:, 0], noise[:, 1], noise[:, 2], noise[:, 3], noise[:, 4], noise[:, 5]
    # xalpha_term = xalpha + 1.5*x0
    # xbeta_term = xbeta * torch.exp(x1 * 0.6)
    # xnemit_term = xnemit * torch.exp(x2 * 0.3)
    # yalpha_term = yalpha + 1.5*x3
    # ybeta_term = ybeta * torch.exp(x4 * 0.6)
    # ynemit_term = ynemit * torch.exp(x5 * 0.3)
    # return torch.stack([xalpha_term, xbeta_term, xnemit_term, yalpha_term, ybeta_term, ynemit_term], dim=1)

    # weights = torch.tensor([1.5, 0.6, 0.3, 1.5, 0.6, 0.3], device=noise.device)
    weights = torch.tensor([1.5, 0.8, 0.3, 1.5, 0.8, 0.3], device=noise.device)
    scaled = noise * weights
    return torch.stack([
        xalpha + scaled[:, 0],
        xbeta * torch.exp(scaled[:, 1]),
        xnemit * torch.exp(scaled[:, 2]),
        yalpha + scaled[:, 3],
        ybeta * torch.exp(scaled[:, 4]),
        ynemit * torch.exp(scaled[:, 5]),
    ], dim=1)

            
def noise2covar(
    noise: torch.Tensor,
    xalpha: float = _xalpha, xbeta: float = _xbeta, xnemit: float = _xnemit,
    yalpha: float = _yalpha, ybeta: float = _ybeta, ynemit: float = _ynemit,
    bg: float = _bg
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the cs values
    cs = noise2cs(noise, 
                  xalpha=xalpha, xbeta=xbeta, xnemit=xnemit,
                  yalpha=yalpha, ybeta=ybeta, ynemit=ynemit)
    # Preallocate covariance matrices
    xcov = torch.empty(noise.size(0), 2, 2, dtype=_dtype)
    ycov = torch.empty(noise.size(0), 2, 2, dtype=_dtype)
    # Fill in xcov
    xcov[:, 0, 0] = cs[:, 1]
    xcov[:, 0, 1] = -cs[:, 0]
    xcov[:, 1, 0] = -cs[:, 0]
    xcov[:, 1, 1] = (cs[:, 0]**2 + 1) / cs[:, 1]
    # xcov[:, 0, 0].copy_(cs[:, 1])
    # xcov[:, 0, 1].copy_(-cs[:, 0])
    # xcov[:, 1, 0].copy_(-cs[:, 0])
    # xcov[:, 1, 1].copy_((cs[:, 0]**2 + 1) / cs[:, 1])
    # Fill in ycov
    ycov[:, 0, 0] = cs[:, 4]
    ycov[:, 0, 1] = -cs[:, 3]
    ycov[:, 1, 0] = -cs[:, 3]
    ycov[:, 1, 1] = (cs[:, 3]**2 + 1) / cs[:, 4]
    # ycov[:, 0, 0].copy_(cs[:, 4])
    # ycov[:, 0, 1].copy_(-cs[:, 3])
    # ycov[:, 1, 0].copy_(-cs[:, 3])
    # ycov[:, 1, 1].copy_((cs[:, 3]**2 + 1) / cs[:, 4])
    # Scale by cs[:,2] / bg and cs[:,5] / bg respectively
    return xcov * (cs[:, 2] / bg).unsqueeze(-1).unsqueeze(-1), ycov * (cs[:, 5] / bg).unsqueeze(-1).unsqueeze(-1)
    
    
def covar2cs(xcov, ycov, bg=_bg):
    # Compute xnemit, xbeta, xalpha
    xnemit = torch.sqrt(xcov[:, 0, 0] * xcov[:, 1, 1] - xcov[:, 0, 1]**2) * bg
    xbeta = xcov[:, 0, 0] * bg / xnemit
    xalpha = -xcov[:, 0, 1] * bg / xnemit
    # Compute ynemit, ybeta, yalpha
    ynemit = torch.sqrt(ycov[:, 0, 0] * ycov[:, 1, 1] - ycov[:, 0, 1]**2) * bg
    ybeta = ycov[:, 0, 0] * bg / ynemit
    yalpha = -ycov[:, 0, 1] * bg / ynemit
    # Stack the results into a single tensor
    return torch.stack([xalpha, xbeta, xnemit, yalpha, ybeta, ynemit], dim=1)
    
    
def drift_maps_2x2(L,**kwarg):
    M = torch.tensor([[1, L], [0, 1]],dtype=_dtype)
    return [M,M]
    
    
def quadrupole_maps_2x2(L,B2,**kwarg):
    if not isinstance(B2,torch.Tensor):
        B2 = torch.tensor(B2,dtype=_dtype)
    if 'Brho' in kwarg:
        k = B2/kwarg['Brho']
    else:
        k = B2/calculate_Brho(**kwarg)
    kr2    = torch.abs(k)**0.5
    coskL  = torch.cos(kr2*L)
    sinkL  = torch.sin(kr2*L)
    coshkL = torch.cosh(kr2*L)
    sinhkL = torch.sinh(kr2*L)
    
    M1 = torch.stack([
        torch.stack([     coskL, sinkL/kr2]),
        torch.stack([-kr2*sinkL, coskL])
    ])
    M2 = torch.stack([
        torch.stack([     coshkL, sinhkL/kr2]),
        torch.stack([ kr2*sinhkL, coshkL])
    ])
    
    if k.item() > 0:
        return M1,M2
    else:
        return M2,M1
        
        
class Element:
    def __init__(self, name: str, type: str, aper: float,
                 map_generator: Callable = None, **properties):
        """
        Represents a lattice element (e.g., drift, quadrupole).

        Args:
            name (str): Name of the element.
            type (str): Type of the element (e.g., 'quadrupole', 'drift').
            aper (float): Aperture of the element.
            map_generator (Callable, optional): Function to generate element's map.
            **properties: Additional properties for the element.
        """
        self.name = name
        for t,lt in _types.items():
            if type in lt:
                type = t
                break
        self.type = type
        self.aper = aper
        self.properties = copy(properties)

        self.map_generator = map_generator or self._default_map_generator()
        self.map = self.map_generator(**self.properties)
        
    def _default_map_generator(self) -> Callable:
        """
        Returns the default map generator for the element based on its type.
        """
        if self.type == 'quadrupole':
            return quadrupole_maps_2x2
        elif self.type == 'drift':
            return drift_maps_2x2
        else:
            raise ValueError(f"Unknown element type: {self.type}")
            
    def reconfigure(self,**properties):
        self.properties.update(properties)
        self.map = self.map_generator(**self.properties)
        
    def to_dict(self):
        """
        Returns a dictionary representation of the element, including 
        the name, type, aper, map_generator, and any additional properties.
        """
        element_dict = {
            'name': self.name,
            'type': self.type,
            'aper': self.aper,
        }
        element_dict.update(self.properties)
        return copy(element_dict)

        
# LatticeMap class definition
class LatticeMap:
    def __init__(self, elem_dicts: List[Dict]):
        """
        Represents a collection of elements forming a lattice.
        """
        self.elements = [Element(**edict) for edict in elem_dicts]
    
    def get_ifrom_ito_map(self, i_from: int, i_to: int) -> Tuple[torch.Tensor, torch.Tensor]:
        Mh = self.elements[i_from].map[0]  # horizontal matrix map
        Mv = self.elements[i_from].map[1]  # vertical matrix map
        for i in range(i_from+1,i_to):
            Mh = self.elements[i].map[0]@Mh 
            Mv = self.elements[i].map[1]@Mv
        return Mh,Mv
    
    def get_maps_ibtw(self, indices: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.get_ifrom_ito_map(indices[i], indices[i+1]) for i in range(len(indices) - 1)]
    
    def get_maps_btw(self, elem_names: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices = [i for i, elem in enumerate(self.elements) if elem.name in elem_names]
        return self.get_maps_ibtw(indices)
    
    def get_expanded_maps_ibtw(self, i_monitors, batch_size):
        return [[M2[0].unsqueeze(0).expand(batch_size, -1, -1),
                 M2[1].unsqueeze(0).expand(batch_size, -1, -1)] 
                 for M2 in self.get_maps_ibtw(i_monitors)]
        
    def elem_dicts(self):
        return [elem.to_dict() for elem in self.elements]
        

class EnvelopeEnsembleModel:
    def __init__(self,
                 E_MeV_u: float,
                 mass_number: int,
                 charge_number: int,
                 lattice_dicts = BDS_dicts_f5501_t5567,
                 quads_to_scan: list = None,  # List of quadrupole names for scanning
                 B2min: list = None,     # Min bounds for quadrupole strength (T/m)
                 B2max: list = None,     # Max bounds for quadrupole strength (T/m)
                 xcovs: torch.Tensor = None,
                 ycovs: torch.Tensor = None,
                 beamloss_sig_level = 4,
                 dtype  = _dtype,
                 cs_ref =_cs_ref): 
        """
        quads_to_scan : list
            Names of quadrupoles to scan. Must match the order in the lattice.
        B2min, B2max : list
            Min and max bounds for quadrupole strengths (T/m).
        xcovs, ycovs : torch.Tensor
            Beam covariance matrices in x and y directions.
        cs_ref : list
            Reference Courant-Snyder parameters.
        """
        self.Brho = calculate_Brho(E_MeV_u,mass_number,charge_number)
        for dic in lattice_dicts:
            if 'Brho' in dic:
                dic['Brho'] = self.Brho
        self.lattice_dicts = lattice_dicts
        self.latmap = LatticeMap(lattice_dicts)
        self.bg   = calculate_betagamma(E_MeV_u,mass_number)
        self.dtype = dtype
        self.cs_ref = torch.tensor(cs_ref,dtype=self.dtype)
        self.beamloss_sig_level = beamloss_sig_level
        self.selected_cov_index = None

        if xcovs is None:
            self.xcovs, self.ycovs = noise2covar(torch.zeros(1,6,dtype=dtype),*cs_ref,bg=self.bg)
        else:
            self.xcovs = torch.tensor(xcovs,dtype=dtype)
            self.ycovs = torch.tensor(ycovs,dtype=dtype)
            
        self._initialize_lattice_indices(quads_to_scan)        
        self._initialize_B2_bounds(B2min, B2max)
                        
    def _initialize_lattice_indices(self,quads_to_scan=None):
        self.quads, self.i_quads = [], []
        self.bpms, self.i_bpms, self.bpm_names = [], [], []
        self.pms, self.i_pms, self.pm_names = [], [], []
        
        for i, elem in enumerate(self.latmap.elements):
            if elem.type == 'quadrupole':
                self.quads.append(elem)
                self.i_quads.append(i)
            elif 'BPM' in elem.name:
                self.bpms.append(elem)
                self.i_bpms.append(i)
                self.bpm_names.append(elem.name)
            elif 'PM' in elem.name:
                self.pms.append(elem)
                self.i_pms.append(i)
                self.pm_names.append(elem.name)
                
        if quads_to_scan is None:
            self.quads_to_scan, self.i_quads_to_scan = self.quads, self.i_quads
        else:
            quads_to_scan = sort_by_Dnum(quads_to_scan)
            self.quads_to_scan, self.i_quads_to_scan = [], []
            for i,elem in enumerate(self.latmap.elements):
                if elem.type == 'quadrupole':
                    if elem.name in quads_to_scan:
                        self.quads_to_scan.append(elem)
                        self.i_quads_to_scan.append(i)
                
    def _initialize_B2_bounds(self, B2min=None, B2max=None):
        if B2min is None:
            B2min, B2max = [], []
            for iq in self.i_quads_to_scan:
                B2 = self.latmap.elements[iq].properties['B2']
                B2min.append(2 if B2 >= 0 else -20)
                B2max.append(20 if B2 >= 0 else -2)
        self.B2min = torch.tensor(B2min, dtype=self.dtype)
        self.B2max = torch.tensor(B2max, dtype=self.dtype)
        
        
    def noise2covar(self,noise):
        if hasattr(self,'best_noise_ensemble_cs_ref'):
            cs_ref = self.best_noise_ensemble_cs_ref
        else:
            cs_ref = self.cs_ref
        return noise2covar(noise,*cs_ref,bg=self.bg)
    
    def noise2cs(self,noise):
        if hasattr(self,'best_noise_ensemble_cs_ref'):
            cs_ref = self.best_noise_ensemble_cs_ref
        else:
            cs_ref = self.cs_ref
        return noise2cs(noise,*cs_ref)
    
    def covar2cs(self,xcovs,ycovs):
        return covar2cs(xcovs,ycovs,bg=self.bg)
        
    def reconfigure_quadrupole_strengths(self,lB2,Brho=None):
        Brho = Brho or self.Brho
        for i,b2 in enumerate(lB2):
            self.quads_to_scan[i].reconfigure(B2=b2,Brho=Brho)
    
    def simulate_beam_covars(self,xcovs,ycovs,i_monitors):
        '''
        xcovs.shape = (batch_size,2,2) 
        i_monitors : list of index (must be sorted) of lattice elements at which beam covariances will be calculated.
        '''

        
        l_xcovs = torch.empty(len(i_monitors),*xcovs.shape,dtype=self.dtype)
        l_ycovs = torch.empty(len(i_monitors),*ycovs.shape,dtype=self.dtype)
        
        start_idx = 0
        if i_monitors[0] == 0:
            l_xcovs[0], l_ycovs[0] = xcovs, ycovs
            start_idx = 1
        else:
            i_monitors = [0] + i_monitors
            
        maps = self.latmap.get_maps_ibtw(i_monitors)
        batch_size = xcovs.shape[0]
        
        for idx,M in enumerate(maps):
            Mx = M[0].unsqueeze(0).expand(batch_size, -1, -1)
            My = M[1].unsqueeze(0).expand(batch_size, -1, -1)
            xcovs = torch.bmm(torch.bmm(Mx,xcovs),Mx.transpose(2, 1))
            ycovs = torch.bmm(torch.bmm(My,ycovs),My.transpose(2, 1))
            l_xcovs[start_idx + idx] = xcovs
            l_ycovs[start_idx + idx] = ycovs
        return l_xcovs, l_ycovs
    

    def simulate_beam_covars_with_cached_expanded_maps(self, xcovs, ycovs, i_monitors, expanded_maps):
        """
        Simulate beam covariances using precomputed maps.
        
        Args:
            xcovs: Tensor of shape (batch_size, 2, 2) - initial horizontal covariances
            ycovs: Tensor of shape (batch_size, 2, 2) - initial vertical covariances
            i_monitors: List of sorted indices where covariances are calculated
            maps: List of (Mh, Mv) tuples - precomputed transfer maps
        
        Returns:
            l_xcovs, l_ycovs: Tensors of shape (len(i_monitors), batch_size, 2, 2)
        """
        l_xcovs = torch.empty(len(i_monitors), *xcovs.shape, dtype=self.dtype)
        l_ycovs = torch.empty(len(i_monitors), *ycovs.shape, dtype=self.dtype)
        
        start_idx = 0
        if i_monitors[0] == 0:
            l_xcovs[0], l_ycovs[0] = xcovs, ycovs  # Store initial covariances
            start_idx = 1

        
        for idx, (Mx, My) in enumerate(expanded_maps):
                xcovs = torch.bmm(torch.bmm(Mx, xcovs), Mx.transpose(2, 1))
                ycovs = torch.bmm(torch.bmm(My, ycovs), My.transpose(2, 1))
                l_xcovs[start_idx + idx] = xcovs
                l_ycovs[start_idx + idx] = ycovs
        
        return l_xcovs, l_ycovs

    # def simulate_beam_covars(self, xcovs, ycovs, i_monitors):
    #     '''
    #     xcovs.shape = (batch_size,2,2) 
    #     i_monitors : list of index (must be sorted) of lattice elements at which beam covariances will be calculated.
    #     '''
    #     l_xcovs = torch.empty(len(i_monitors), *xcovs.shape, dtype=self.dtype)
    #     l_ycovs = torch.empty(len(i_monitors), *ycovs.shape, dtype=self.dtype)
        
    #     start_idx = 0
    #     if i_monitors[0] == 0:
    #         l_xcovs[0], l_ycovs[0] = xcovs, ycovs
    #         start_idx = 1
    #     else:
    #         i_monitors = [0] + i_monitors
            
    #     maps = self.latmap.get_maps_ibtw(i_monitors)
    #     batch_size = xcovs.shape[0]
        
    #     temp_xcovs = xcovs.clone()  # One-time copy
    #     temp_ycovs = ycovs.clone()

    #     for idx, M in enumerate(maps):
    #         Mx = M[0].unsqueeze(0).expand(batch_size, -1, -1)
    #         My = M[1].unsqueeze(0).expand(batch_size, -1, -1)
            
    #         temp_xcovs = torch.bmm(Mx, temp_xcovs)
    #         temp_xcovs = torch.bmm(temp_xcovs, Mx.transpose(2, 1))
    #         temp_ycovs = torch.bmm(My, temp_ycovs)
    #         temp_ycovs = torch.bmm(temp_ycovs, My.transpose(2, 1))
            
    #         l_xcovs[start_idx + idx] = temp_xcovs
    #         l_ycovs[start_idx + idx] = temp_ycovs

    #     return l_xcovs, l_ycovs



        
    def backward_simulate_beam_covars(self,xcovs,ycovs,i_monitors):
        '''
        xcovs.shape = (batch_size,2,2) 
        i_monitors : list of index (must be sorted) of lattice elements at which beam covariances will be calculated.
            ***i_monitors[-1] must be the starting point for backtracking***
        '''
        l_xcovs = torch.empty(len(i_monitors),*xcovs.shape,dtype=self.dtype)
        l_ycovs = torch.empty(len(i_monitors),*ycovs.shape,dtype=self.dtype)
        l_xcovs[-1], l_ycovs[-1] = xcovs, ycovs
            
        maps = self.latmap.get_maps_ibtw(i_monitors)
        batch_size = xcovs.shape[0]
        
        for idx,M in enumerate(maps[::-1]):
            invMx = torch.inverse(M[0])
            invMy = torch.inverse(M[1])
            invMx = invMx.expand(batch_size, -1, -1)
            invMy = invMy.expand(batch_size, -1, -1)
            xcovs = torch.bmm(torch.bmm(invMx,xcovs),invMx.transpose(2, 1))
            ycovs = torch.bmm(torch.bmm(invMy,ycovs),invMy.transpose(2, 1))
            l_xcovs[-2-idx] = xcovs
            l_ycovs[-2-idx] = ycovs
        return l_xcovs, l_ycovs
        

    def multi_reconfigure_simulate_beam_covars(self,llB2,xcovs,ycovs,i_monitors):
        ll_xcovs, ll_ycovs = [], []
        for lB2 in llB2:
            self.reconfigure_quadrupole_strengths(lB2)
            l_xcovs, l_ycovs = self.simulate_beam_covars(xcovs,ycovs,i_monitors)
            ll_xcovs.append(l_xcovs)
            ll_ycovs.append(l_ycovs)
        return torch.stack(ll_xcovs,dim=0),torch.stack(ll_ycovs,dim=0)   # shape of len(llB2), len(i_monitors), batch_size, 2, 2

    def multi_reconfigure_simulate_beam_covars_with_cached_maps(self, xcovs, ycovs, i_monitors, multiconf_expanded_maps):
        ll_xcovs, ll_ycovs = [], []
        for expanded_maps in multiconf_expanded_maps:
            l_xcovs, l_ycovs = self.simulate_beam_covars_with_cached_expanded_maps(
                xcovs, ycovs, i_monitors, expanded_maps
            )
            ll_xcovs.append(l_xcovs)
            ll_ycovs.append(l_ycovs)
        return torch.stack(ll_xcovs, dim=0), torch.stack(ll_ycovs, dim=0)


    def _calculate_beam_loss(self, apers, ll_xvars, ll_yvars):
        # apers: tensor, shape : (n_monitors)
        # ll_xcovs : tensor, shape : (n_scan, n_monitors, batch_size, 2, 2)
        max_rms  = (torch.max(ll_xvars, ll_yvars))**0.5 # (meter)
        return torch.relu((self.beamloss_sig_level * max_rms / apers[None,:,None] - 1))**2  # shape of (n_scan, n_monitors, batch_size) 
        
    def simulate_beam_loss(self,xcovs,ycovs,i_extra_aper=None):
        if i_extra_aper is None:
            i_apers = self.i_quads
        else:
            i_apers = sorted(set(self.i_quads).union(i_extra_aper))
        apers = torch.tensor([self.latmap.elements[idx].aper for idx in i_apers], dtype=self.dtype)
        l_xcovs, l_ycovs = self.simulate_beam_covars(xcovs,ycovs,i_monitors=i_apers)
        l_xvars, l_yvars = l_xcovs[:,:,0,0], l_ycovs[:,:,0,0]
        return torch.amax(self._calculate_beam_loss(apers, l_xvars.unsqueeze(0), l_yvars.unsqueeze(0)), dim=(0, 1))
    
    def _get_cs_reconst_loss_ftn(self,
        batch_size,
        iBPMQ, BPMQ_llB2, BPMQ_targets, BPMQ_tolerances, 
        iPM=None, PM_llB2=None, PM_xrms_targets=None, PM_yrms_targets=None, PM_xrms_tolerances=None, PM_yrms_tolerances=None, 
        fit_err=False,
        xnemit_target=None,
        ynemit_target=None,
        compute_beam_loss = True,
        i_extra_aper = None,
        ):  
        '''
        BPMQ_llB2.shape : (n_scan, n_quad)
        BPMQ_targets.shape : (n_scan, n_monitors, 1)
        BPMQ_tolerances.shape : (1, n_monitors, 1) or (n_scan, n_monitors, 1)
        '''

        if compute_beam_loss:
            if i_extra_aper is None:
                i_apers = self.i_quads
            else:
                i_apers = sorted(set(self.i_quads).union(i_extra_aper))
        else:
            i_apers = []
            
        iBPMQ = sorted(iBPMQ)
        i_apers_wBPMQ = sorted(set(iBPMQ).union(i_apers))
        apers_wBPMQ = torch.tensor([self.latmap.elements[idx].aper for idx in i_apers_wBPMQ], dtype=self.dtype)
        arg_iBPMQ = [i for i, imon in enumerate(i_apers_wBPMQ) if imon in iBPMQ]
        if iPM is not None and PM_llB2 is not None:
            # if len(iPM) > 0:
            iPM = sorted(iPM)
            # i_apers_wPM = sorted(set(iPM).union(i_apers))
            i_apers_wPM = iPM #sorted(set(iPM).union(i_apers))
            arg_iPM = [i for i, imon in enumerate(i_apers_wPM) if imon in iPM]
            apers_wPM = torch.tensor([self.latmap.elements[idx].aper for idx in i_apers_wPM], dtype=self.dtype)

        n_bpmQ_err = 0
        n_pm_err = 0
        if fit_err:
            n_scan, n_bpm, _ = BPMQ_targets.shape
            n_bpmQ_err = n_scan*n_bpm
            if PM_llB2 is not None:
                n_pm_scan, n_pm, _ = PM_xrms_targets.shape
                n_pm_err = n_pm_scan*n_pm
#         print("n_bpmQ_err",n_bpmQ_err)        
#         print("n_pm_err",n_pm_err)

        # pre-computed cached maps

        BPMQ_llMaps = []
        for lB2 in BPMQ_llB2:
            self.reconfigure_quadrupole_strengths(lB2)
            if i_apers_wBPMQ[0]==0:
                BPMQ_llMaps.append(self.latmap.get_expanded_maps_ibtw(i_apers_wBPMQ, batch_size))
            else:
                BPMQ_llMaps.append(self.latmap.get_expanded_maps_ibtw([0]+i_apers_wBPMQ, batch_size))

        if PM_llB2 is not None:
            PM_llMaps = []
            for lB2 in PM_llB2:
                self.reconfigure_quadrupole_strengths(lB2)
                if i_apers_wPM[0]==0:
                    PM_llMaps.append(self.latmap.get_expanded_maps_ibtw(i_apers_wPM, batch_size))
                else:
                    PM_llMaps.append(self.latmap.get_expanded_maps_ibtw([0]+i_apers_wPM, batch_size))
        
        def loss_fun(x):
            '''
            x.shape : (batch_size, 6+n_scan*n_bpm + 2*n_pm_scan*n_pm)
            '''
            xcovs, ycovs = self.noise2covar(x[:,:6])
            # batch_size = x.shape[0]
            assert batch_size == x.shape[0]
            fitloss_bpmQ = None
            fitloss_PM = None
            regloss_beamloss = None
            regloss_PM_err = None
            regloss_bpmQ_err = None
            regloss_emitprior = None

            if BPMQ_llB2 is not None:
                ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars_with_cached_maps(xcovs,ycovs,i_apers_wBPMQ, BPMQ_llMaps)
                # ll_xcovs_, ll_ycovs_ = self.multi_reconfigure_simulate_beam_covars(BPMQ_llB2,xcovs,ycovs,i_apers_wBPMQ)
                # print("ll_xcovs[0,0,0,:,:]",ll_xcovs[0,0,0,:,:])
                # print("ll_xcovs_[0,0,0,:,:]",ll_xcovs_[0,0,0,:,:])
                # print("diff",torch.abs(ll_xcovs-ll_xcovs_).mean())
                # ll_xcovs: tensor, shape: (n_scan, n_monitors, batch_size, 2, 2)
                ll_xvars, ll_yvars = ll_xcovs[:,:,:,0,0], ll_ycovs[:,:,:,0,0]
                BPMQ_sim = (ll_xvars[:,arg_iBPMQ,:] - ll_yvars[:,arg_iBPMQ,:])*1e6            
                if fit_err:
                    #bpmQ_err = x[:,6:6+n_bpmQ_err].view(n_scan,n_bpm,batch_size)
                    bpmQ_err = x[:,6:6+n_bpmQ_err].reshape(n_scan,n_bpm,batch_size)
                    target = BPMQ_targets + bpmQ_err
                    regloss_bpmQ_err = torch.mean(torch.abs(bpmQ_err), dim=[0,1])
                else:
                    target = BPMQ_targets
                # print("BPMQ_sim.shape, BPMQ_targets.shape",BPMQ_sim.shape, BPMQ_targets.shape)

                fitloss_bpmQ = torch.mean(torch.abs(BPMQ_sim - target) / BPMQ_tolerances, dim=[0,1])  # shape of batch_size
                regloss_beamloss = torch.amax(self._calculate_beam_loss(apers_wBPMQ, ll_xvars, ll_yvars), dim=(0, 1)) 
                
            if PM_llB2 is not None:
                # ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars(PM_llB2,xcovs,ycovs,i_apers_wPM)
                ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars_with_cached_maps(xcovs,ycovs,i_apers_wPM,PM_llMaps)
                # print("ll_xcovs[0,0,0,:,:]",ll_xcovs[0,0,0,:,:])
                # print("ll_xcovs_[0,0,0,:,:]",ll_xcovs_[0,0,0,:,:])
                # print("diff",torch.abs(ll_xcovs-ll_xcovs_).mean())
                ll_xvars, ll_yvars = ll_xcovs[:,:,:,0,0], ll_ycovs[:,:,:,0,0]
                xPM_sim = ll_xvars[:,arg_iPM,:]**0.5*1e3
                yPM_sim = ll_yvars[:,arg_iPM,:]**0.5*1e3
                if fit_err:
                    xPM_err = x[:,6+n_bpmQ_err:6+n_bpmQ_err+n_pm_err].reshape(n_pm_scan,n_pm,batch_size)
                    xPM_target = PM_xrms_targets + xPM_err
                    yPM_err = x[:,6+n_bpmQ_err+n_pm_err:6+n_bpmQ_err+2*n_pm_err].reshape(n_pm_scan,n_pm,batch_size)
                    yPM_target = PM_yrms_targets + yPM_err
                    regloss_PM_err = 0.5*torch.sum(torch.abs(xPM_err) + torch.abs(yPM_err), dim=[0,1])
                else:
                    target = BPMQ_targets
                    xPM_target = PM_xrms_targets
                    yPM_target = PM_yrms_targets
                fitloss_PM = torch.sum(torch.abs(xPM_sim/xPM_target -1.0) / PM_xrms_tolerances
                                      + torch.abs(yPM_sim/yPM_target -1.0) / PM_yrms_tolerances, dim=[0,1])
                regloss_beamloss = regloss_beamloss + torch.amax(self._calculate_beam_loss(apers_wPM, ll_xvars, ll_yvars), dim=(0, 1)) 
                
            if xnemit_target is not None:
                cs = self.noise2cs(x[:,:6])
                xnemit_sim_ratio = cs[:,2]/xnemit_target
                ynemit_sim_ratio = cs[:,5]/ynemit_target
                regloss_emitprior = (torch.relu(torch.abs(xnemit_sim_ratio - 1) - 0.2)**2 +
                                     torch.relu(torch.abs(ynemit_sim_ratio - 1) - 0.2)**2)
            
            # make each regloss > 0 and < 1 for tolerable region to work with torch_helper.run_torch_optimizer
            return {'fitloss_bpmQ': fitloss_bpmQ,
                    'fitloss_PM': fitloss_PM,
                    'regloss_beamloss': regloss_beamloss,
                    'regloss_PM_err': regloss_PM_err,
                    'regloss_bpmQ_err': regloss_bpmQ_err,
                    'regloss_emitprior': regloss_emitprior}
        
        return loss_fun


    def _bootstrap_data(self, BPMQ_llB2: torch.Tensor, l_BPMQ_targets_like: List[torch.Tensor], min_data_points=12):
        """ Perform bootstrapping on the provided data. """
        BPMQ_targets = l_BPMQ_targets_like[0]
        n_scan = BPMQ_targets.shape[0]
        n_monitor = BPMQ_targets.shape[1]
        if (n_scan - 1) * n_monitor > min_data_points:
            # 70% subsample
            sub_sample_rate = 0.7
            num_samples = ceil(sub_sample_rate * n_scan)
            min_samples_needed = ceil(min_data_points / n_monitor)
            num_samples = max(num_samples, min_samples_needed)  
            indices = torch.randperm(n_scan)[:num_samples]
            l_bootstraped = []
            for v in l_BPMQ_targets_like:
                if v is None:
                    l_bootstraped.append(v)
                elif v.shape[0]==n_scan:
                    l_bootstraped.append(v[indices])
                else:
                    l_bootstraped.append(v)
            return BPMQ_llB2[indices], l_bootstraped
        else:
            return BPMQ_llB2, l_BPMQ_targets_like
    
    def cs_reconstruct(self, 
                       BPMQ_i_monitors: List[int], 
                       BPMQ_llB2      : torch.Tensor, 
                       BPMQ_targets   : torch.Tensor, 
                       BPMQ_tolerances: torch.Tensor = None, 
                       BPMQ_model_err : torch.Tensor = None, 
                       BPMQ_weight    : Optional[float] = None,
                       PM_i_monitors  : Optional[List[int]] = None, 
                       PM_llB2        : torch.Tensor = None, 
                       PM_xrms_targets: Optional[torch.Tensor] = None, 
                       PM_yrms_targets: Optional[torch.Tensor] = None, 
                       PM_xrms_tolerances: Optional[torch.Tensor] = None, 
                       PM_yrms_tolerances: Optional[torch.Tensor] = None, 
                       PM_weight         : Optional[float] = None,
                       fit_err: Optional[bool] = False,
                       i_extra_aper = None,
                       xnemit_target: Optional[float] = None, 
                       ynemit_target: Optional[float] = None,
                       batch_size: int = 8,
                       n_batch_padding_factor: int = 8,
                       lr: float = 0.2,
                       max_iter: int = 300,
                       num_restarts: int = 4,
                       bootstrap = False,
                       plot_history = True,
                       sample_model_err = False,
                       retrun_loss_ftn_4_debug = False,
                       ):    

        print(" ======== cs_reconstruct ========")
        assert set(BPMQ_i_monitors) <= set(self.i_bpms)
        self.BPMQ_llB2 = BPMQ_llB2
        self.PM_llB2 = PM_llB2

        if BPMQ_tolerances is None:
            BPMQ_tolerances = 0.5*torch.ones(len(BPMQ_i_monitors)) # combined loss is weighed assuming BPMQ error <~ 0.5 mm^2 
        else:
            if torch.any(BPMQ_tolerances <= 1e-6):
                raise ValueError("BPMQ_tolerances must not be negative or smaller than machine precision")
            BPMQ_tolerances = BPMQ_tolerances / BPMQ_tolerances.mean() *0.5  # combined loss is weighed assumingBPMQ error <~ 0.5 mm^2
        
        if BPMQ_tolerances.ndim == 1:
            BPMQ_tolerances = BPMQ_tolerances.view(1,-1,1)
        else:
            BPMQ_tolerances = BPMQ_tolerances[:,:,None]

        if sample_model_err and BPMQ_model_err is not None:
            BPMQ_targets_samples = BPMQ_targets.unsqueeze(-1).expand(-1, -1, batch_size * n_batch_padding_factor)
            BPMQ_targets_samples = BPMQ_targets_samples + BPMQ_model_err[:, :, None] * torch.randn_like(BPMQ_targets_samples)
        else:
            BPMQ_targets_samples = BPMQ_targets[:, :, None]

        if PM_xrms_targets is not None:
            assert set(PM_i_monitors) <= set(self.i_pms)
            if PM_xrms_tolerances is None:
                PM_xrms_tolerances = 0.02*torch.ones(len(PM_i_monitors))  # 2% error
            else:
                if torch.any(PM_xrms_tolerances <= 1e-6):
                    raise ValueError("PM_rms_tolerances must not be negative or smaller than machine precision")
            PM_xrms_tolerances = PM_xrms_tolerances / PM_xrms_tolerances.mean()
            if PM_yrms_tolerances is None:
                PM_yrms_tolerances = 0.02*torch.ones(len(PM_i_monitors))
            else:
                if torch.any(PM_yrms_tolerances <= 1e-6):
                    raise ValueError("PM_rms_tolerances must not be negative or smaller than machine precision")
            PM_yrms_tolerances = PM_yrms_tolerances / PM_yrms_tolerances.mean()
            if PM_xrms_tolerances.ndim == 1:
                PM_xrms_tolerances = PM_xrms_tolerances.view(1,-1,1)
            else:
                PM_xrms_tolerances = PM_xrms_tolerances[:,:,None]
            if PM_yrms_tolerances.ndim == 1:
                PM_yrms_tolerances = PM_yrms_tolerances.view(1,-1,1)
            else:
                PM_yrms_tolerances = PM_yrms_tolerances[:,:,None]

            PM_xrms_targets = PM_xrms_targets[:, :, None]
            PM_yrms_targets = PM_yrms_targets[:, :, None]

        if bootstrap:
            BPMQ_llB2_bootstrap, l_BPMQ_targets_like = self._bootstrap_data(BPMQ_llB2,[BPMQ_targets_samples,BPMQ_tolerances])
            BPMQ_targets_samples_bootstrap, BPMQ_tolerances_bootstrap = l_BPMQ_targets_like
            # print("BPMQ_targets_samples.shape, BPMQ_targets_bootstrap.shape",BPMQ_targets_samples.shape, BPMQ_targets_bootstrap.shape)
            args = (batch_size*n_batch_padding_factor, BPMQ_i_monitors, BPMQ_llB2_bootstrap, BPMQ_targets_samples_bootstrap, BPMQ_tolerances_bootstrap)
        else:
            args = (batch_size*n_batch_padding_factor, BPMQ_i_monitors, BPMQ_llB2, BPMQ_targets_samples, BPMQ_tolerances)

        kwargs = {
            'fit_err': fit_err,
            'iPM': PM_i_monitors,
            'PM_llB2': PM_llB2,
            'PM_xrms_targets': PM_xrms_targets,
            'PM_yrms_targets': PM_yrms_targets,
            'PM_xrms_tolerances': PM_xrms_tolerances,
            'PM_yrms_tolerances': PM_yrms_tolerances,
            'xnemit_target': xnemit_target,
            'ynemit_target': ynemit_target,
        }
        BPMQ_weight = BPMQ_weight or 1.0
        PM_weight = PM_weight or 1.0
        loss_weights = {'fitloss_bpmQ':BPMQ_weight, 
                        'fitloss_PM':PM_weight, 
                        'regloss_beamloss':1.0, 
                        'regloss_PM_err':1.0, 
                        'regloss_bpmQ_err':1.0, 
                        'regloss_emitprior':1.0
                        }
        loss_func = self._get_cs_reconst_loss_ftn(*args,**kwargs,
            compute_beam_loss = True,
            i_extra_aper = i_extra_aper)
        loss_func_wo_beamloss = self._get_cs_reconst_loss_ftn(*args,**kwargs,
            compute_beam_loss = False)
        
        n_param = 6
        n_bpmQ_err = 0
        n_pm_err = 0
        if fit_err:
            n_bpmQ_err = BPMQ_targets.shape[0] * BPMQ_targets.shape[1]
            n_param = n_param + n_bpmQ_err
            if PM_llB2 is not None:
                n_pm_err = PM_xrms_targets.shape[0] * PM_xrms_targets.shape[1]
                n_param = n_param + 2*n_pm_err
        x0 = torch.randn(batch_size*n_batch_padding_factor, n_param, dtype=self.dtype, requires_grad=True)

        if retrun_loss_ftn_4_debug: 
            return loss_func,loss_func_wo_beamloss,x0

        result = run_torch_optimizer(
                                    loss_func = loss_func,
                                    x0 = x0,
                                    max_iter = max_iter,
                                    loss_weights = loss_weights,
                                    low_fidelity_loss_func = loss_func_wo_beamloss,
                                    lr = lr,
                                    plot_history = plot_history
                                    )
        combined_losses = result.fun
        combined_noise_ensemble = result.x
        
        sorted_indices = combined_losses.argsort()[:batch_size]
        combined_losses = combined_losses[sorted_indices]
        combined_noise_ensemble = combined_noise_ensemble[sorted_indices]
        # print("BPMQ_targets.shape, BPMQ_tolerances.shape, BPMQ_model_err.shape",BPMQ_targets.shape, BPMQ_tolerances.shape, BPMQ_model_err.shape)

        irestart = 1
        for i in range(num_restarts-1):
            print("irestart",irestart)
            mask = combined_losses < 0.05*(1+np.log(irestart+1))
            if torch.sum(mask) > batch_size:
                break
            irestart += 1

            if sample_model_err and BPMQ_model_err is not None:
                BPMQ_targets_samples = BPMQ_targets.unsqueeze(-1).expand(-1, -1, batch_size * n_batch_padding_factor)
                BPMQ_targets_samples = BPMQ_targets_samples + BPMQ_model_err[:, :, None] * torch.randn_like(BPMQ_targets_samples)
            else:
                BPMQ_targets_samples = BPMQ_targets[:, :, None]

            if bootstrap:
                BPMQ_llB2_bootstrap, l_BPMQ_targets_like = self._bootstrap_data(BPMQ_llB2,[BPMQ_targets_samples,BPMQ_tolerances])
                BPMQ_targets_samples_bootstrap, BPMQ_tolerances_bootstrap = l_BPMQ_targets_like
                args = (batch_size*n_batch_padding_factor, BPMQ_i_monitors, BPMQ_llB2_bootstrap, BPMQ_targets_samples_bootstrap, BPMQ_tolerances_bootstrap)
            else:
                args = (batch_size*n_batch_padding_factor, BPMQ_i_monitors, BPMQ_llB2, BPMQ_targets_samples, BPMQ_tolerances)

            loss_func = self._get_cs_reconst_loss_ftn(*args,**kwargs,
                compute_beam_loss = True,
                i_extra_aper = i_extra_aper)
            loss_func_wo_beamloss = self._get_cs_reconst_loss_ftn(*args,**kwargs,
                compute_beam_loss = False)

            noise_ensemble = torch.randn(batch_size*n_batch_padding_factor, n_param, dtype=self.dtype, requires_grad=True)
            result = run_torch_optimizer(
                                        loss_func = loss_func,
                                        x0 = noise_ensemble,
                                        max_iter = max_iter,
                                        loss_weights = loss_weights,
                                        low_fidelity_loss_func = loss_func_wo_beamloss,
                                        lr = lr,
                                        plot_history = plot_history,
                                        )
            losses = result.fun
            noise_ensemble = result.x
            
            sorted_indices = losses.argsort()[:batch_size]
            losses = losses[sorted_indices]
            noise_ensemble = noise_ensemble[sorted_indices]

            combined_losses = torch.cat((combined_losses, losses))
            combined_noise_ensemble = torch.cat((combined_noise_ensemble, noise_ensemble), dim=0)

        # select some best solutions of ceil(1.2*batch_size/ irestart)  from each bootsrapped fitting
        if bootstrap:
            if irestart>1:
                solutions_per_restart = max(ceil(2*batch_size/ irestart),batch_size-1)
                indices = []
                for i in range(irestart):
                    start_idx = i * batch_size
                    indices += list(np.arange(start_idx, start_idx + solutions_per_restart))
            
            combined_losses = combined_losses[indices]
            combined_noise_ensemble = combined_noise_ensemble[indices]
        
        sorted_indices = combined_losses.argsort()[:batch_size]
        best_losses = combined_losses[sorted_indices]
        best_noise_ensemble = combined_noise_ensemble[sorted_indices]
        
        self.best_noise_ensemble = best_noise_ensemble
        self.best_noise_ensemble_cs_ref = self.cs_ref.clone()
        self.xcovs, self.ycovs = self.noise2covar(best_noise_ensemble[:,:6]) 
        self.selected_cov_index = select_n_most_distant_mmd4d_covs(2,self.xcovs.detach().numpy(),
                                                                     self.ycovs.detach().numpy(),
                                                                     xcov_ref = self.xcovs[0].detach().numpy(),
                                                                     ycov_ref = self.ycovs[0].detach().numpy())
        self.xcovs_mean = self.xcovs.mean(dim=0, keepdim=True)
        self.ycovs_mean = self.ycovs.mean(dim=0, keepdim=True)
        self.cs_mean = self.covar2cs(self.xcovs_mean,self.ycovs_mean).view(-1)
        self.cs = self.noise2cs(best_noise_ensemble[:1,:6]).view(-1)
        
        # fit_info = {'fit_loss': best_losses.detach().numpy(),
        #             'cs_ref': self.cs_ref.detach().numpy(),
        #             'cs_mean': self.cs_mean.detach().numpy(),
        #             'cs': self.cs.detach().numpy(),
        #             'best_noise_ensemble': best_noise_ensemble.detach().numpy(),
        #             'best_noise_ensemble_cs_ref': self.best_noise_ensemble_cs_ref.detach().numpy(),
        #             'best_noise_ensemble_cs': self.noise2cs(best_noise_ensemble[:,:6]).detach().numpy(),
        #             'best_noise_ensemble_xcovs': self.xcovs.detach().numpy(),
        #             'best_noise_ensemble_ycovs': self.ycovs.detach().numpy(),
        #             'best_noise_ensemble_xcovs_mean': self.xcovs_mean.detach().numpy(),
        #             'best_noise_ensemble_ycovs_mean': self.ycovs_mean.detach().numpy(),
        #             'best_noise_ensemble_cs_mean': self.cs_mean
        #             'bpmQ_err_fit'
        #}
        #self.history['loss_reconstCS'].append(best_losses.detach().numpy())
        # return fit_info
    
    def _get_loss_maximize_BPMQ_var(self, iBPMQ, llB2_penal=None, compute_beam_loss=True,i_extra_aper=None):
    
        if compute_beam_loss:
            if i_extra_aper is None:
                i_apers = self.i_quads
            else:
                i_apers = sorted(set(self.i_quads).union(i_extra_aper))
        else:
            i_apers = []
            
        iBPMQ = sorted(iBPMQ)
        i_apers_wBPMQ = sorted(set(iBPMQ).union(i_apers))
        apers_wBPMQ = torch.tensor([self.latmap.elements[idx].aper for idx in i_apers_wBPMQ], dtype=self.dtype)
        arg_iBPMQ = [i for i, imon in enumerate(i_apers_wBPMQ) if imon in iBPMQ]
        B2norm = 0.01*(self.B2max - self.B2min)  # 1% of range

        if self.selected_cov_index is not None:
            xcovs = self.xcovs[self.selected_cov_index]
            ycovs = self.ycovs[self.selected_cov_index]
        else:
            xcovs = self.xcovs
            ycovs = self.ycovs  
        
        def loss_fun(lB2):
            self.reconfigure_quadrupole_strengths(lB2)

            l_xcovs, l_ycovs = self.simulate_beam_covars(xcovs,ycovs,i_apers_wBPMQ)
            l_xvars, l_yvars = l_xcovs[:,:,0,0], l_ycovs[:,:,0,0]
            BPMQ_sim = l_xvars[arg_iBPMQ,:]*1e6 - l_yvars[arg_iBPMQ,:]*1e6  # (mm^2)
            
            # maximize variance of BPMQ for best resolving solution.   torch.tensor.max(axis=..) gives tuple of max and argmax
            # loss_BPMQ_var = -torch.mean(BPMQ_sim.max(axis=1).values-BPMQ_sim.min(axis=1).values)  
            loss = 1 -torch.mean(BPMQ_sim.std(axis=-1))*2 # use normalization factor of BPMQ error~0.5 mm^2
            regloss_beamloss = self._calculate_beam_loss(apers_wBPMQ,
                                                         l_xvars.unsqueeze(0),
                                                         l_yvars.unsqueeze(0)
                                                         ).max()
            regloss_B2_limit   = torch.mean(torch.relu((self.B2min-lB2)/B2norm)**2) \
                               + torch.mean(torch.relu((lB2-self.B2max)/B2norm)**2)
            regloss_BPMQ_limit = torch.mean(torch.relu(torch.abs(BPMQ_sim)-25))**2 
            
            if llB2_penal is None:
                regloss_llB2_penal = torch.zeros_like(loss)
            else:
                regloss_llB2_penal = torch.relu(1 - torch.abs(lB2.unsqueeze(0) - llB2_penal).mean())**2
            
            # make each regloss > 0 and < 1 for tolerable region to work with torch_helper.run_torch_optimizer
            return {"loss":loss, "regloss_beamloss"  :regloss_beamloss,  
                                 "regloss_B2_limit"  :regloss_B2_limit,
                                 "regloss_BPMQ_limit":regloss_BPMQ_limit,
                                 "regloss_llB2_penal":regloss_llB2_penal} 
        return loss_fun
    
    def query_candidate_quad_set_maximizing_BPMQ_var(self,BPMQ_i_monitors,
                                                     llB2_penal = None,
                                                     i_extra_aper = None,
                                                     max_iter=200,
                                                     num_restarts=3,
                                                     plot_history = True):
                                                     
        print(" ======== query_candidate_quad_set_maximizing_BPMQ_var ========")
        assert set(BPMQ_i_monitors) <= set(self.i_bpms)

        loss_func = self._get_loss_maximize_BPMQ_var(BPMQ_i_monitors,
                                                     llB2_penal = llB2_penal,
                                                     compute_beam_loss = True,
                                                     i_extra_aper = i_extra_aper)

        loss_func_wo_beamloss = self._get_loss_maximize_BPMQ_var(BPMQ_i_monitors,
                                                     llB2_penal = llB2_penal,
                                                     compute_beam_loss = False)
        for _ in range(5):
            candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
            result = run_torch_optimizer(
                                        loss_func = loss_func,
                                        x0 = candidate_quad_set,
                                        max_iter = max_iter,
                                        lr = 0.05,
                                        loss_weights = {"loss":1.0, "regloss_beamloss":2.0,  
                                                        "regloss_B2_limit":2.0, "regloss_BPMQ_limit":2.0,
                                                        "regloss_llB2_penal":1.0},
                                        low_fidelity_loss_func = loss_func_wo_beamloss,
                                        plot_history = plot_history,
                                        )
            if result.success:
                break
        
        best_loss = result.fun
        best_regloss_beamloss = result.losses["regloss_beamloss"].item()
        candidate_quad_set = result.x
        
        for i in range(num_restarts - 1):
            if best_loss < 0.5 and best_regloss_beamloss < 1e-3:
                break
            for _ in range(5):
                candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
                result = run_torch_optimizer(
                                            loss_func = loss_func,
                                            x0 = candidate_quad_set,
                                            max_iter = max_iter,
                                            lr = 0.05,
                                            loss_weights = {"loss":1.0, "regloss_beamloss":2.0,  
                                                            "regloss_B2_limit":2.0, "regloss_BPMQ_limit":2.0,
                                                            "regloss_llB2_penal":1.0},
                                            low_fidelity_loss_func = loss_func_wo_beamloss,
                                            plot_history = plot_history
                                            )
                if result.success:
                    break
            regloss_beamloss = result.losses["regloss_beamloss"].item()
            
            if regloss_beamloss < best_regloss_beamloss:
                if best_regloss_beamloss < 1e-3 or result.fun < best_loss:
                    best_loss = result.fun
                    best_regloss_beamloss = regloss_beamloss
                    candidate_quad_set = result.x.clone()
                    
                
        ensemble_var_of_BPMQ = best_loss
        # print("best_loss",best_loss)
        # print("best_regloss_beamloss",best_regloss_beamloss)
        return candidate_quad_set, ensemble_var_of_BPMQ
    

    def _get_loss_maximize_PM_var(self, iPM, llB2_penal=None, compute_beam_loss=True, i_extra_aper=None):

        if compute_beam_loss:
            if i_extra_aper is None:
                i_apers = self.i_quads
            else:
                i_apers = sorted(set(self.i_quads).union(i_extra_aper))
        else:
            i_apers = []

        iPM = sorted(iPM)
        i_apers_wPM = sorted(set(iPM).union(i_apers))
        apers_wPM = torch.tensor([self.latmap.elements[idx].aper for idx in i_apers_wPM], dtype=self.dtype)
        arg_iPM = [i for i, imon in enumerate(i_apers_wPM) if imon in iPM]
        B2norm = 0.01*(self.B2max - self.B2min)  # 1% of range

        if self.selected_cov_index is not None:
            xcovs = self.xcovs[self.selected_cov_index]
            ycovs = self.ycovs[self.selected_cov_index]
        else:
            xcovs = self.xcovs
            ycovs = self.ycovs  


        def loss_fun(lB2,regularize=True):
            self.reconfigure_quadrupole_strengths(lB2)

            l_xcovs, l_ycovs = self.simulate_beam_covars(xcovs,ycovs,i_apers_wPM)
            l_xvars, l_yvars = l_xcovs[:,:,0,0], l_ycovs[:,:,0,0]
            xvar_sim, yvar_sim = l_xvars[arg_iPM,:]*1e6, l_yvars[arg_iPM,:]*1e6  # (mm^2)

            # maximize variance of PM for best resolving solution.   torch.tensor.max(axis=..) gives tuple of max and argmax
            loss = 1 -torch.mean(  xvar_sim.std(axis=-1)/(xvar_sim.mean(axis=-1) + 0.2)  # 0.2mm systematic error assumed
                                 + yvar_sim.std(axis=-1)/(yvar_sim.mean(axis=-1) + 0.2)
                                 )*10  # 10% beam size variation from reference (mean value) is tolerance
            ## use normalization factor of rms error~0.1 mm -> 5= 1/0.1 mm *2 (for x-y plane)
            regloss_beamloss = self._calculate_beam_loss(apers_wPM,
                                                         l_xvars.unsqueeze(0),
                                                         l_yvars.unsqueeze(0)
                                                         ).max()
            regloss_B2_limit   = torch.mean(torch.relu((self.B2min-lB2)/B2norm)**2) \
                               + torch.mean(torch.relu((lB2-self.B2max)/B2norm)**2)
            regloss_PM_limit = torch.mean(torch.relu(torch.abs(xvar_sim)-25)+torch.relu(torch.abs(yvar_sim)-25))**2  # less than 5mm beam
            
            if llB2_penal is None:
                regloss_llB2_penal = torch.zeros_like(loss)
            else:
                regloss_llB2_penal = torch.relu(1 - torch.abs(lB2.unsqueeze(0) - llB2_penal).mean())**2

            # make each regloss > 0 and < 1 for tolerable region to work with torch_helper.run_torch_optimizer
            return {"loss":loss, "regloss_beamloss"  :regloss_beamloss,  
                                 "regloss_B2_limit"  :regloss_B2_limit,
                                 "regloss_PM_limit"  :regloss_PM_limit,
                                 "regloss_llB2_penal":regloss_llB2_penal} 
        return loss_fun

        
    def query_candidate_quad_set_maximizing_PM_var(self,PM_i_monitors,
                                                     llB2_penal = None,
                                                     i_extra_aper = None,
                                                     max_iter=200,
                                                     num_restarts=3,
                                                     plot_history = True):
                                                     
        print(" ======== query_candidate_quad_set_maximizing_PM_var ========")
        assert set(PM_i_monitors) <= set(self.i_pms)

        loss_func = self._get_loss_maximize_PM_var(PM_i_monitors,
                                                   llB2_penal = llB2_penal,
                                                   compute_beam_loss = True,
                                                   i_extra_aper = i_extra_aper)

        loss_func_wo_beamloss = self._get_loss_maximize_PM_var(PM_i_monitors,
                                                     llB2_penal = llB2_penal,
                                                     compute_beam_loss = False)
        
        candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
        result = run_torch_optimizer(
                                    loss_func = loss_func,
                                    x0 = candidate_quad_set,
                                    max_iter = max_iter,
                                    lr = 0.05,
                                    loss_weights = {"loss":1.0, "regloss_beamloss":2.0,  
                                                    "regloss_B2_limit":2.0, "regloss_PM_limit":2.0,
                                                    "regloss_llB2_penal":1.0},
                                    low_fidelity_loss_func = loss_func_wo_beamloss,
                                    plot_history = plot_history,
                                    )
        
        best_loss = result.fun
        best_regloss_beamloss = result.losses["regloss_beamloss"].item()
        candidate_quad_set = result.x
        
        for i in range(num_restarts - 1):
            if best_loss < 0.5 and best_regloss_beamloss < 1e-3:
                break
            candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
            result = run_torch_optimizer(
                                        loss_func = loss_func,
                                        x0 = candidate_quad_set,
                                        max_iter = max_iter,
                                        lr = 0.05,
                                        loss_weights = {"loss":1.0, "regloss_beamloss":2.0,  
                                                        "regloss_B2_limit":2.0, "regloss_PM_limit":2.0,
                                                        "regloss_llB2_penal":1.0},
                                        low_fidelity_loss_func = loss_func_wo_beamloss,
                                        plot_history = plot_history
                                        )
            regloss_beamloss = result.losses["regloss_beamloss"].item()
            
            if regloss_beamloss < best_regloss_beamloss:
                if best_regloss_beamloss < 1e-3 or result.fun < best_loss:
                    best_loss = result.fun
                    best_regloss_beamloss = regloss_beamloss
                    candidate_quad_set = result.x.clone()
                    
        ensemble_var_of_PM = best_loss
        # print("best_loss",best_loss)
        # print("best_regloss_beamloss",best_regloss_beamloss)
        return candidate_quad_set, ensemble_var_of_PM
        

class virtual_Evaluator_wBPMQ:
    def __init__(
        self,
        E_MeV_u, mass_number, charge_number,
        lattice_dicts = BDS_dicts_f5501_t5567,
        quads_to_scan = None,    # quads names for BPMQ scan. must be in order of lattice
        BPM_names = None,  # bpm names for BPMQ measure. must be in order of lattice
        B2min = None,     # min bounds in B2 (T/m)
        B2max = None,     # max bounds in B2 (T/m)
        xcovs = None,
        ycovs = None,
        cs_ref = None,
        dtype=_dtype,
        virtual_beamQerr = 0.0,
        virtual_beamQmodelerr = 0.0,
        seed = 0, # for reproducibility
        ):

        # print("in virtual_Evaluator_wBPMQ init, BPM_names",BPM_names)
        self.virtual_beamQerr = virtual_beamQerr
        self.virtual_beamQmodelerr = virtual_beamQmodelerr
        self.dtype = dtype
        self.seed = seed
        
        with torch.no_grad():
            if cs_ref is None:
                is_beamloss = True
                while is_beamloss:
                    cs_ref = noise2cs(torch.randn(1,6,dtype=dtype)).view(6)
                    env_model = EnvelopeEnsembleModel(
                        E_MeV_u, mass_number, charge_number,
                        lattice_dicts = lattice_dicts,                  
                        quads_to_scan = quads_to_scan,
                        B2min=B2min,
                        B2max=B2max,
                        xcovs=xcovs,
                        ycovs=ycovs,
                        cs_ref = cs_ref,
                        dtype  = dtype,
                        )
                    is_beamloss = env_model.simulate_beam_loss(env_model.xcovs,env_model.ycovs).max() > 0
            else:
                env_model = EnvelopeEnsembleModel(
                    E_MeV_u, mass_number, charge_number,
                    lattice_dicts = lattice_dicts,                  
                    quads_to_scan = quads_to_scan,
                    B2min=B2min,
                    B2max=B2max,
                    xcovs=xcovs,
                    ycovs=ycovs,
                    cs_ref = cs_ref,
                    dtype  = dtype,
                    )
                is_beamloss = env_model.simulate_beam_loss(env_model.xcovs,env_model.ycovs).max() > 0
                if is_beamloss:
                    raise ValueError(f"cs_ref {cs_ref} result in beam loss")


        self.cs_ref = cs_ref
        self.env_model = env_model
        
        if quads_to_scan is None:
            quads_to_scan = [q.name for q in self.env_model.quads_to_scan]
        self.mp_quads_to_scan = get_MPelem_from_PVnames(quads_to_scan)
        
        self.BPM_names = []
        self.i_bpms = []
        for i, elem in enumerate(lattice_dicts):
            if BPM_names is None or elem['name'] in BPM_names:
                if 'BPM' in elem['name']:
                    self.BPM_names.append(elem['name'])
                    self.i_bpms.append(i)
                    
        if BPM_names is not None:
            assert self.BPM_names == BPM_names
        self.BPM_names = sort_by_Dnum(self.BPM_names)

        BPM_TIS161_PVs = []      
        for i,name in enumerate(self.BPM_names):
            TIS161_PVs = [f"{name}:TISMAG161_{i + 1}_RD" for i in range(4)]
            BPM_TIS161_PVs += TIS161_PVs
        self.BPM_TIS161_PVs = np.array(BPM_TIS161_PVs)

    def read(self):
        with torch.no_grad():
            l_xcovs, l_ycovs = self.env_model.simulate_beam_covars(self.env_model.xcovs,
                                                                   self.env_model.ycovs,
                                                                   self.i_bpms)
            xvars, yvars = l_xcovs[:,0,0,0], l_ycovs[:,0,0,0]
            BPMQ_sim = (xvars -yvars).detach().numpy()*1e6
        np.random.seed(self.seed)
        self.seed = self.seed + 13
        BPMQ_sim += self.virtual_beamQerr*np.random.randn(*BPMQ_sim.shape)
        DiffSum = BPMQ_sim/241
        bpmU2 = 1 + DiffSum
        bpmU1 = 1 - DiffSum
        data = {}
        for bpm in self.BPM_names:
            data[bpm+':XPOS_RD'] = np.random.randn(5)*1e-6
            data[bpm+':YPOS_RD'] = np.random.randn(5)*1e-6
            data[bpm+':MAG_RD' ]  = 1+np.random.randn(5)*1e-6
            data[bpm+':TISMAG161_1_RD'] = np.random.randn(5)*1e-12
            data[bpm+':TISMAG161_2_RD'] = np.random.randn(5)*1e-12
            data[bpm+':TISMAG161_3_RD'] = np.random.randn(5)*1e-12
            data[bpm+':TISMAG161_4_RD'] = np.random.randn(5)*1e-12
        timestamps = []
        for i in range(5):
            timestamps.append(pd.Timestamp(datetime.now()))
            for j, bpm in enumerate(self.BPM_names):
                data[bpm + ':TISMAG161_1_RD'][i] += bpmU1[j]
                data[bpm + ':TISMAG161_2_RD'][i] += bpmU2[j]
            time.sleep(0.1)
        data = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps), columns=data.keys())
        return data

        
    def _set_and_read(self, x,                 
        ensure_set_kwargs = None,
        fetch_data_kwargs = None,
        ):
        lB2 = []
        for curr,mp_elem in zip(x,self.mp_quads_to_scan):
            lB2.append(mp_elem.convert(curr,from_field='I',to_field='B2'))
            
        with torch.no_grad():
            lB2 = torch.tensor(lB2,dtype=self.dtype)
            self.env_model.reconfigure_quadrupole_strengths(lB2)
        data =self.read()
        return data, data
        
    def submit(self, x, 
        ensure_set_kwargs = None,
        fetch_data_kwargs = None,
        ):
        """
        Submit a task to set and read data asynchronously.
        """
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._set_and_read, x, 
                                 ensure_set_kwargs=ensure_set_kwargs,
                                 fetch_data_kwargs=fetch_data_kwargs)
        executor.shutdown(wait=False)
        return future
        
        
    def _calculate_beamQ(self, data):
        """
        Calculates the beam charge (beamQ) for each BPM based on the TIS161 PVs and positional data.
        
        Args:
            data: Data object to update with beamQ values.
        """
        for i, name in enumerate(self.BPM_names):
            U = self.BPM_TIS161_PVs[4 * i:4 * (i + 1)]
            Q = (data[[U[1], U[2]]].sum(axis=1) - data[[U[0], U[3]]].sum(axis=1)) / data[U].sum(axis=1)
            data[f'{name}:Q'] = Q
            data[f'{name}:beamQ'] = (241 * Q) - (data[f'{name}:XPOS_RD'] ** 2 - data[f'{name}:YPOS_RD'] ** 2)
            if self.virtual_beamQmodelerr > 1e-3:
                data[f'{name}:beamQ_model_err'] = np.ones(len(data))*self.virtual_beamQmodelerr

    
    def get_result(self, future):
        """
        Retrieve the result from the future.
        """
        data, ramping_data = future.result()
        self._calculate_beamQ(data)
        self._calculate_beamQ(ramping_data)
        return data, ramping_data    
   
   
   
def plot_reconstructed_ellipse(model,selected_cov_index=None,cs_ref=None,bg=_bg,xlim=None,ylim=None):
    '''
    compare reconstructed ellipses for virtual machinie
    '''
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    MMD4 = None
    if selected_cov_index is None:
        selected_cov_index = np.arange(len(model.xcovs))
    for cov in model.xcovs[selected_cov_index]:
        plot_beam_ellipse_from_cov(cov.detach().numpy(),fig=fig,ax=ax[0])
    for cov in model.ycovs[selected_cov_index]:
        plot_beam_ellipse_from_cov(cov.detach().numpy(),fig=fig,ax=ax[1])
    if cs_ref is None:
        plot_beam_ellipse(*model.cs[:3],bg,'x',ls=':',color='k',fig=fig,ax=ax[0])
        plot_beam_ellipse(*model.cs[3:],bg,'y',ls=':',color='k',fig=fig,ax=ax[1])
    else:
        if isinstance(cs_ref,torch.Tensor):
            cs_ref  = cs_ref.detach().numpy()
        plot_beam_ellipse(*cs_ref[:3],bg,'x',color='k',fig=fig,ax=ax[0])
        plot_beam_ellipse(*cs_ref[3:],bg,'y',color='k',fig=fig,ax=ax[1])
        mis_x = calculate_mismatch_factor(cs_ref[:3],model.cs[:3])
        mis_y = calculate_mismatch_factor(cs_ref[3:],model.cs[3:])
        MMD4  = calculate_MMD4D(cs_ref,model.cs.detach().numpy())
        plot_beam_ellipse(*model.cs[:3],bg,'x',ls=':',color='k',fig=fig,ax=ax[0],label=f'{mis_x:.2f}')
        plot_beam_ellipse(*model.cs[3:],bg,'y',ls=':',color='k',fig=fig,ax=ax[1],label=f'{mis_y:.2f}')
    ax[0].legend()
    ax[1].legend()
    if xlim is not None:
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if ylim is not None:
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)     
    if MMD4:
        fig.text(0.47, 0.95, f'MMD10: {10*MMD4:.2f}', ha='left', va='bottom')
    fig.tight_layout()
   
   
   
_valid_quads = ["BDS_BTS:PSQ_D5501", "BDS_BTS:PSQ_D5509", "BDS_BTS:PSQ_D5552", "BDS_BTS:PSQ_D5559"]
_valid_bpms  = ["BDS_BTS:BPM_D5513", "BDS_BTS:BPM_D5565"]
_valid_pms   = ["BDS_BTS:PM_D5567"]
_valid_corrs = ["BDS_BTS:PSC2_D5496", "BDS_BTS:PSC1_D5496", "BDS_BTS:PSC2_D5563", "BDS_BTS:PSC1_D5563"]
@dataclass
class BPMQscan:
    E_MeV_u: float
    mass_number: int
    charge_number: int
    lattice_dicts: BDS_dicts_f5501_t5567
    quads_to_scan :  List[str  ] = field(default_factory=lambda: _valid_quads)
    quads_max_curr:  List[int  ] = field(default_factory=lambda: [150, 150, 150, 150])
    quads_min_curr:  List[int  ] = field(default_factory=lambda: [5, 5, 5, 5])
    quads_tol_curr:  List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 0.3])
    quads_init_rel_size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.05])
    corrs_to_scan :  List[str  ] = field(default_factory=lambda: _valid_corrs)
    corrs_max_curr:  List[int  ] = field(default_factory=lambda: [10, 10, 10, 10])
    corrs_min_curr:  List[int  ] = field(default_factory=lambda: [-10, -10, -10, -10])
    corrs_tol_curr:  List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.1])
    corrs_step_curr: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    BPM_names: List[str] = field(default_factory=lambda: _valid_bpms)  # bpm names for BPMQ measure. must be in order of lattice
    PM_names : List[str] = field(default_factory=lambda: _valid_pms)  # bpm names for BPMQ measure. must be in order of lattice
    BPMQ_model_type: str = 'TIS161',
    xnemit_target: Optional[float] = None
    ynemit_target: Optional[float] = None
    machineIO: Optional[Any] = None
    set_manually: bool = True
    correct_traj_each_iter: bool = False
    wait_before_measure: bool = False
    train_BPMQtol: Optional[List[float]] = None
    batch_size: int = 8
    n_batch_padding_factor: int = 16
    num_restarts: int = 5
    fit_err: bool = False
    sample_model_err: bool = False
    bootstrap: bool = False
    plot_history: bool = False
    plot_ellipse: bool = True
    plot_xlim : bool = None
    plot_ylim : bool = None
    cs_ref: Optional[List[float]] = None
    dtype: torch.dtype = _dtype
    virtual_beamQerr: float = 0.0
    virtual_beamQmodelerr: float = 0.0
    virtual_emitprior: bool = False
    ISAAC_preset_keywords: Optional[List[str]] = None
    n_init: Optional[int] = 4
    seed: Optional[int] = 42

    def __post_init__(self):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.seed = self.seed + 13
        self.now = datetime.now()
        self._validate_PVs()
        self._initialize_attributes()
        self._setup_quads_evaluator()
        if self.machineIO and self.correct_traj_each_iter:
            self._setup_corrs_evaluator()
            
        self.llB2_penal = None
        self.reconstructed_covs_history = []
        self.reconstructed_cs_history = []
        self.evaluated_dfs = []
        
        if self.cs_ref is not None:
            self.cs_ref = torch.tensor(self.cs_ref)
        self.model = EnvelopeEnsembleModel(
            self.E_MeV_u, self.mass_number, self.charge_number,
            lattice_dicts=self.lattice_dicts, 
            quads_to_scan=self.quads_to_scan,
            B2min=self.B2min,
            B2max=self.B2max, 
            dtype=self.dtype,
            # cs_ref = self.cs_ref,
        )
        self.mismatch_x = []
        self.mismatch_y = []
        self.mmd4d = []
        self.elapsed_time = []
    
        self.i_bpms = []
        for i, elem in enumerate(self.lattice_dicts):
            if elem['name'] in self.BPM_names:
                self.i_bpms.append(i)
        self.train_llB2 = None
        self.train_llBPMQ = None
        if self.train_BPMQtol is not None:
            self.train_BPMQtol = torch.tensor(self.train_BPMQtol, dtype=self.dtype)
        self.train_llBPMQtol = None
        self.train_llBPMQmodelerr = None

        self.i_pms = []
        for i, elem in enumerate(self.lattice_dicts):
            if elem['name'] in self.PM_names:
                self.i_pms.append(i)
        self.train_llB2rms = None
        self.train_llxrms = None
        self.train_llyrms = None
        self.train_llxrmstol = None
        self.train_llyrmstol = None
        
        self.trainllB2_rms = None
        self.train_llxrms = None
        self.train_llyrms = None
        self.train_llxrmstol = None
        self.train_llyrmstol = None

        
        

    def _validate_PVs(self):
        pass
        # invalid_quads = set(self.quads_to_scan) - set(_valid_quads)
        # invalid_bpms = set(self.BPM_names) - set(_valid_bpms)
        # invalid_corrs = set(self.corrs_to_scan) - set(_valid_corrs)
        
        # if invalid_quads:
        #     raise ImplementationError(f'Invalid quads_to_scan: {invalid_quads}')
        # if invalid_bpms:
        #     raise ImplementationError(f'Invalid BPM_names: {invalid_bpms}')
        # if invalid_corrs:
        #     raise ImplementationError(f'Invalid corrs_to_scan: {invalid_corrs}')

    def _initialize_attributes(self):
        """Initializes attributes based on provided parameters."""
        self.quads_to_scan = sort_by_Dnum(self.quads_to_scan)
        self.mp_quads_to_scan = get_MPelem_from_PVnames(self.quads_to_scan)
        self.BPM_names = sort_by_Dnum(self.BPM_names)
        self.BPM_MAG_PVs  = [name + ":MAG_RD"  for name in self.BPM_names]
        self.BPM_XPOS_PVs = [name + ":XPOS_RD" for name in self.BPM_names]
        self.BPM_YPOS_PVs = [name + ":YPOS_RD" for name in self.BPM_names]
        self.PM_names = sort_by_Dnum(self.PM_names)
        self.bg = calculate_betagamma(self.E_MeV_u, self.mass_number)

        # Compute B2min and B2max for quadrupoles
        self.B2min, self.B2max = self._calculate_b2_limits()

        # Generate CSETs and RDs lists
        self.quads_input_CSETs = [name + ':I_CSET' for name in self.quads_to_scan]
        self.quads_input_RDs   = [name + ':I_RD'   for name in self.quads_to_scan]
        self.corrs_input_CSETs = [name + ':I_CSET' for name in self.corrs_to_scan]
        self.corrs_input_RDs   = [name + ':I_RD'   for name in self.corrs_to_scan]
        self.corrs_output_RDs   = self.BPM_XPOS_PVs + self.BPM_YPOS_PVs

    def _calculate_b2_limits(self):
        """Calculates B2min and B2max for the quadrupoles."""
        B2min = []
        B2max = []
        for mp_quad, min_curr, max_curr in zip(self.mp_quads_to_scan, self.quads_min_curr, self.quads_max_curr):
            try:
                b2min = mp_quad.convert(min_curr, from_field='I', to_field='B2')
                b2max = mp_quad.convert(max_curr, from_field='I', to_field='B2')
            except ConversionError as e:
                raise RuntimeError(f"Error converting currents to B2 values for {mp_quad.name}: {e}")
            B2min.append(min(b2min, b2max))
            B2max.append(max(b2min, b2max))
        return B2min, B2max
        
    def _setup_quads_evaluator(self):
        """Sets up the quadrupole evaluator based on machineIO or initializes a virtual evaluator."""
        if self.machineIO is None:
            self.quads_evaluator = virtual_Evaluator_wBPMQ(
                self.E_MeV_u, self.mass_number, self.charge_number,
                lattice_dicts=self.lattice_dicts,
                quads_to_scan=self.quads_to_scan,
                BPM_names=self.BPM_names,
                B2min=self.B2min,
                B2max=self.B2max,
                xcovs=None,
                ycovs=None,
                cs_ref=None,
                dtype=self.dtype,
                virtual_beamQerr=self.virtual_beamQerr,
                virtual_beamQmodelerr=self.virtual_beamQmodelerr,
                seed=self.seed,
            )
            if self.cs_ref is None:
                self.cs_ref = self.quads_evaluator.cs_ref
        else:
            self.quads_evaluator = Evaluator_wBPMQ(
                self.machineIO,
                input_CSETs= self.quads_input_CSETs,
                input_RDs  = self.quads_input_RDs,
                input_tols = self.quads_tol_curr,
                output_RDs = self.corrs_input_CSETs + self.corrs_input_RDs,
                BPM_names  = self.BPM_names,
                model_type = self.BPMQ_model_type,
                ensure_set_kwargs = None,
                fetch_data_kwargs = None,
                set_manually = self.set_manually
            )

            if self.correct_traj_each_iter:
                self._setup_corrs_evaluator()

    def _setup_corrs_evaluator(self):
        """Sets up the trajectory machine evaluator if applicable."""
        self.corrs_evaluator = Evaluator_wBPMQ(
            self.machineIO,
            input_CSETs = self.corrs_input_CSETs,
            input_RDs   = self.corrs_input_RDs,
            input_tols  = self.corrs_tol_curr,
            output_RDs  = self.quads_input_CSETs + self.quads_input_RDs,
            BPM_names   = self.BPM_names,
            model_type = self.BPMQ_model_type,
            ensure_set_kwargs = None,
            fetch_data_kwargs = None,
            set_manually=self.set_manually
        )            

    def _setup_traj_controller(self):
        x0, _ = fetch_data(self.corrs_evaluator.input_CSETs,0.1)
        self.traj_controller = LinearControl(
                                x0  = x0,
                                dx  = self.corrs_step_curr,
                                xmin= self.corrs_min_curr,
                                xmax= self.corrs_max_curr,
                                goal= np.zeros(len(self.BPM_names)),
                                goal_tol=np.ones(len(self.BPM_names)),
                                evaluator = self.corrs_evaluator,
                                input_RDs = self.corrs_input_RDs,
                                output_RDs = self.corrs_output_RDs)
                                
                                
    def get_data(self):
        data = {
            "E_MeV_u": self.E_MeV_u,  # Use 'self' to reference instance attributes
            "mass_number": self.mass_number,
            "charge_number": self.charge_number,
            "quads_to_scan": self.quads_to_scan,
            "corrs_to_scan": self.corrs_to_scan,
            "BPM_names": self.BPM_names,
            "lattice_dicts": self.lattice_dicts,
            "bootstrap": self.bootstrap,
            "correct_traj_each_iter": self.correct_traj_each_iter,
            "xnemit_target": self.xnemit_target,
            "ynemit_target": self.ynemit_target,
            "reconstructed_cs_loc":self.lattice_dicts[0]["name"],
            "reconstructed_cs_history": self.reconstructed_cs_history,
            "reconstructed_covs_history": self.reconstructed_covs_history,
            "evaluated_dfs": self.evaluated_dfs,
            'train_llB2': self.train_llB2.detach().cpu().numpy() if self.train_llB2 is not None else None,
            'train_llBPMQ': self.train_llBPMQ.detach().cpu().numpy() if self.train_llBPMQ is not None else None,
            'train_llBPMQtol': self.train_llBPMQtol.detach().cpu().numpy() if self.train_llBPMQtol is not None else None,
            'train_llBPMQmodelerr': self.train_llBPMQmodelerr.detach().cpu().numpy() if self.train_llBPMQmodelerr is not None else None,
            'train_llB2rms': self.train_llB2rms.detach().cpu().numpy() if self.train_llB2rms is not None else None,
            'train_llxrms': self.train_llxrms.detach().cpu().numpy() if self.train_llxrms is not None else None,
            'train_llyrms': self.train_llyrms.detach().cpu().numpy() if self.train_llyrms is not None else None,
            'train_llxrmstol': self.train_llxrmstol.detach().cpu().numpy() if self.train_llxrmstol is not None else None,
            'train_llyrmstol': self.train_llyrmstol.detach().cpu().numpy() if self.train_llyrmstol is not None else None,
            'mismatch_x': self.mismatch_x if hasattr(self,'mismatch_x') else None,
            'mismatch_y': self.mismatch_y if hasattr(self,'mismatch_y') else None,
            'mmd4d': self.mmd4d if hasattr(self,'mmd4d') else None,
            'elapsed_time': self.elapsed_time if hasattr(self,'elapsed_time') else None,
            'B2min': self.B2min,
            'B2max': self.B2max,
            'BPM_MAG_PVs': self.BPM_MAG_PVs,
            'BPM_XPOS_PVs': self.BPM_XPOS_PVs,
            'BPM_YPOS_PVs': self.BPM_YPOS_PVs,
            'PM_names': self.PM_names,
            'quads_input_CSETs': self.quads_input_CSETs,
            'quads_input_RDs': self.quads_input_RDs,
            'quads_input_tols': self.quads_tol_curr,
            'quads_output_RDs': self.quads_input_CSETs,
            'corrs_input_CSETs': self.corrs_input_CSETs,
            'corrs_input_RDs': self.corrs_input_RDs,
            'corrs_input_tols': self.corrs_tol_curr,
            'corrs_output_RDs': self.corrs_output_RDs,
            'BPMQ_model_type': self.BPMQ_model_type,
        }
        return data

                
    def save_data(self, fname=None):
        # If fname is not provided, create a default filename
        if fname is None:
            # Format it as YYYYMMDD_HHMM
            timestamp = self.now.strftime("%Y%m%d_%H%M")
            # Create the default filename
            fname = f"{timestamp}_BPMQscan.pkl"
        
        data = self.get_data()
        with open(fname, 'wb') as file:
            pickle.dump(data, file)
        
    def initialize(self,lB2=None, init_llB2=None, n_init=None):
        '''
        Scan quadrupole magnets with preset and measure BPMQ.
        init_llB2: preset, list of list of B2s in unit of T/m
        lB2: base for automatic preset determination. list of B2s in unit of T/m
        '''
        n_init = n_init or self.n_init
        if self.machineIO is not None:
            fetch_time = self.machineIO._fetch_data_time_span
            self.machineIO._fetch_data_time_span = 2*fetch_time
            df = self.quads_evaluator.read()
            # self.evaluated_dfs.append(df)
            self.init_status = df
            self.init_BPM_MAGs = df[self.BPM_MAG_PVs].mean()
            quads_curr = df[self.quads_evaluator.input_RDs].mean()
            self.machineIO._fetch_data_time_span = fetch_time
            
        if init_llB2 is None:
            if lB2 is None:
                lB2 = []
                if self.machineIO is None:
                    lB2 = [q.properties['B2'] for q in self.quads_evaluator.env_model.quads_to_scan]
                else:
                    #quads_curr, _ = self.machineIO.fetch_data(self.machine.input_CSETs, 0.1)
                    lB2 = [mp_quad.convert(curr, from_field='I', to_field='B2') for mp_quad, curr in zip(self.mp_quads_to_scan, quads_curr)]

            #bounds = [(b2-0.2*abs(b2),b2+0.2*abs(b2)) for b2 in lB2]
            bounds = [(max(lB2[i] - self.quads_init_rel_size[i]*abs(lB2[i]), self.B2min[i]), 
                       min(lB2[i] + self.quads_init_rel_size[i]*abs(lB2[i]), self.B2max[i]) 
                      )
                      for i in range(len(lB2))]
            #bounds = [(self.B2min[i],self.B2max[i]) for i in range(len(self.B2min))]
            init_llB2_random_samples = proximal_ordered_init_sampler(
                2*n_init+1,
                bounds = bounds,
                x0 = lB2,
                ramping_rate=1,
                polarity_change_time=0,
                method='sobol',
                seed=self.seed,
            )
            n_preset = 0
            if self.ISAAC_preset_keywords is not None:
                preset_df = get_ISAAC_preset(n_init-1,
                                             keywords = self.ISAAC_preset_keywords,  
                                             E_MeV_u = self.E_MeV_u,
                                             mass_number = self.mass_number,
                                             charge_number = self.charge_number)
                if preset_df is not None:
                    if set(preset_df.columns) == set(self.quads_to_scan):
                        n_preset = len(preset_df)
                        preset_df = preset_df[self.quads_to_scan]
                        for i,col in enumerate(self.quads_to_scan):
                            b2col = [self.mp_quads_to_scan[i].convert(curr, from_field='I', to_field='B2') for curr in preset_df[col].values]
                            preset_df.iloc[:,i] = np.array(b2col, dtype=np.float64)

                        print("ISAAC preset:")
                        display(preset_df)
            print("evaluate_candidate")
            is_not_useful_data = self.evaluate_candidate(torch.tensor([lB2], dtype=self.dtype))
            print("evaluate_candidate done")
            for lB2 in init_llB2_random_samples:
                if self.train_llB2 is not None:
                    if len(self.train_llB2) >= n_init-n_preset:
                        break
                is_not_useful_data = self.evaluate_candidate(torch.tensor(np.array(lB2), dtype=self.dtype))

            if n_preset > 0:
                for lB2 in preset_df.values:
                    is_not_useful_data = self.evaluate_candidate(torch.tensor([lB2], dtype=self.dtype))


        else:
            init_llB2 = torch.tensor(init_llB2, dtype= self.dtype)
            for lB2 in init_llB2:
                self.evaluate_candidate(lB2)

           
        if self.train_llB2 is None:
            retry = input("choose (y/n): No good data without beam loss found. Shall we try with more random quad settings? If not, program need to abort and user need to find good quad settings w/o beam loss to begin with. Also may good to set ninit=1")
            if retry:
                init_llB2_random_samples = proximal_ordered_init_sampler(
                2*n_init+1,
                bounds = bounds,
                x0 = init_llB2[-1],
                ramping_rate=1,
                polarity_change_time=0,
                method='sobol',
                seed=self.seed,
                )
                for lB2 in init_llB2_random_samples:
                    if self.train_llB2 is not None:
                        if len(self.train_llB2) >= n_init:
                            break
                    is_not_useful_data = self.evaluate_candidate(torch.tensor([lB2], dtype=self.dtype))
            else:
                raise ValueError('No good data without beam loss found')
                    
        self.train_model()
        
    def lB2_to_lICSETs(self,lB2):
        if type(lB2) is torch.Tensor:
            lB2 = lB2.detach().numpy().flatten()

        if len(lB2) != len(self.mp_quads_to_scan):
                raise ValueError(f"Mismatch between number of quads ({len(self.mp_quads_to_scan)}) and B2 values ({len(lB2)}).")
        try:
            quad_Iset = [self.mp_quads_to_scan[i].convert(b2,from_field='B2',to_field='I') for i,b2 in enumerate(lB2)]
        except ConversionError as e:
            raise RuntimeError(f"Error converting currents to B2 values for {mp_quad.name}: {e}")
        
        return quad_Iset


    def evaluate_candidate(self,lB2):
        quad_Iset = self.lB2_to_lICSETs(lB2)
        with torch.no_grad():
            # if type(lB2) is torch.Tensor:
            #     lB2 = lB2.detach().numpy().flatten()

            # if len(lB2) != len(self.mp_quads_to_scan):
            #     raise ValueError(f"Mismatch between number of quads ({len(self.mp_quads_to_scan)}) and B2 values ({len(lB2)}).")

            # try:
            #     quad_Iset = [self.mp_quads_to_scan[i].convert(b2,from_field='B2',to_field='I') for i,b2 in enumerate(lB2)]
            # except ConversionError as e:
            #     raise RuntimeError(f"Error converting currents to B2 values for {mp_quad.name}: {e}")
            

            ldata = []
            future = self.quads_evaluator.submit(quad_Iset)
            if self.wait_before_measure:
                input("Press Enter to continue...")
            df,ramping_df = self.quads_evaluator.get_result(future)
            if self.machineIO is None:
                is_beamloss = self.quads_evaluator.env_model.simulate_beam_loss(self.quads_evaluator.env_model.xcovs,
                                                                        self.quads_evaluator.env_model.ycovs).max() > 0.1
            else:
                BPM_MAGs, BPM_MAGs_err = df[self.BPM_MAG_PVs].mean(), df[self.BPM_MAG_PVs].std()
                BPM_MAGs_Lo = 0.95*self.init_BPM_MAGs-2*BPM_MAGs_err
                BPM_MAGs_Hi = 1.05*self.init_BPM_MAGs+2*BPM_MAGs_err
                is_beamloss = np.any(BPM_MAGs < BPM_MAGs_Lo) or np.any(BPM_MAGs > BPM_MAGs_Hi)
                # print("BPM_MAGs, Lo, Hi:",BPM_MAGs, BPM_MAGs_Lo, BPM_MAGs_Hi)

            # lBPMQ = df[[col for col in df.columns if col.endswith(':beamQ')]].mean()    # lBPM is shape of (n_bpm,)
            lBPMQ = df[[name+':beamQ' for name in self.BPM_names]].mean()
            
            if 'GP' in self.BPMQ_model_type:
                lBPMQtol = df[[name+':beamQ_model_err' for name in self.BPM_names]].mean()
                lBPMQmodelerr = lBPMQtol
            else:
                lBPMQtol = None
                lBPMQmodelerr = None
            
            # display(pd.DataFrame(lBPMQ,columns=['']).T)
            is_BPMQ_too_large = np.any(np.abs(lBPMQ.values) > 40)
            if not is_beamloss and not is_BPMQ_too_large:
                self.evaluated_dfs.append(df)
                if self.machineIO is not None:
                    self.init_BPM_MAGs = 0.3*self.init_BPM_MAGs + 0.7*BPM_MAGs
#             if self.correct_traj_each_iter and self.machineIO is not None:
#                 self._setup_traj_controller()
#                 self.traj_controller.run()
#                 self.evaluated_dfs[-1]=ctr.eval_df
#                 df = ctr.eval_df[-1]
#                 BPM_MAGs = df[self.BPM_MAG_PVs]
#                 is_beamloss = np.any(BPM_MAGs < 0.95*self.init_BPM_MAGs)
            # use readback instead of set
            if self.machineIO is not None:
                lB2 = [self.mp_quads_to_scan[i].convert(df[qname+':I_RD'].mean(),from_field='I',to_field='B2') 
                        for i,qname in enumerate(self.quads_to_scan)]
            lB2 = lB2 if isinstance(lB2, torch.Tensor) else torch.tensor(lB2, dtype=self.dtype)
            
            if is_beamloss or is_BPMQ_too_large:
                if is_BPMQ_too_large:
                    print("[Warning] BPMQ too large!")
                if is_beamloss:
                    print("[Warning] Beam loss detected!")
                    if self.machineIO is not None:
                        print("BPM_MAG / initial_BPM_MAGs: ")
                        # display(BPM_MAGs/self.init_BPM_MAGs)
                if self.llB2_penal is None:
                    self.llB2_penal = lB2.unsqueeze(0)
                else:
                    self.llB2_penal = torch.cat((self.llB2_penal,lB2.unsqueeze(0)),dim=0)
            else:
                lBPMQ = lBPMQ if isinstance(lBPMQ, torch.Tensor) else torch.tensor(lBPMQ, dtype=self.dtype)
                lBPMQtol = torch.tensor(lBPMQtol, dtype=self.dtype) if lBPMQtol is not None and not isinstance(lBPMQtol, torch.Tensor) else lBPMQtol
                lBPMQmodelerr = torch.tensor(lBPMQmodelerr, dtype=self.dtype) if lBPMQmodelerr is not None and not isinstance(lBPMQmodelerr, torch.Tensor) else lBPMQmodelerr
                self._concat_BPMQ_train_data(lB2, lBPMQ, lBPMQtol, lBPMQmodelerr) # batch_size = 1
                
        return is_beamloss or is_BPMQ_too_large
        
    def _concat_BPMQ_train_data(self,lB2,lBPMQ,lBPMQtol=None,lBPMQmodelerr=None):
        if self.train_llB2 is None:
            self.train_llB2 = lB2.view(1,-1).clone()
        else:
            n_scan = self.train_llB2.shape[0]
            self.train_llB2 = torch.cat((self.train_llB2,lB2.view(1,-1)),dim=0)

        if self.train_llBPMQ is None:
            self.train_llBPMQ = lBPMQ[None, :]
        else:
            self.train_llBPMQ = torch.cat(( self.train_llBPMQ, lBPMQ[None, :] ), dim=0)

        if lBPMQtol is not None:
            if self.train_llBPMQtol is None:
                if self.train_BPMQtol is None:
                    self.train_llBPMQtol = lBPMQtol[None, :]
                else:
                    self.train_llBPMQtol = lBPMQtol[None, :]*self.train_BPMQtol[None,:]
            else:
                if self.train_BPMQtol is None:
                    self.train_llBPMQtol = torch.cat(( self.train_llBPMQtol, lBPMQtol[None, :] ), dim=0)
                else:
                    self.train_llBPMQtol = torch.cat(( self.train_llBPMQtol, lBPMQtol[None, :]*self.train_BPMQtol[None,:] ), dim=0)
   
        if lBPMQmodelerr is not None:
            if self.train_llBPMQmodelerr is None:
                self.train_llBPMQmodelerr = lBPMQmodelerr[None, :]
            else:
                self.train_llBPMQmodelerr = torch.cat(( self.train_llBPMQmodelerr, lBPMQmodelerr[None, :] ), dim=0)

    def concat_PM_train_data(self,lB2=None, quads_curr=None, lxrms=None, lyrms=None, lxrmstol=None, lyrmstol=None):
        if lB2 is None:
            if quads_curr is None:
                df = self.quads_evaluator.read()
                quads_curr = df[self.quads_evaluator.input_RDs].mean()
            lB2 = [mp_quad.convert(curr, from_field='I', to_field='B2') for mp_quad, curr in zip(self.mp_quads_to_scan, quads_curr)]
        lB2 = torch.tensor(lB2,dtype=_dtype).view(1,-1)
        if self.train_llB2rms is None:
            self.train_llB2rms = lB2
        else:  
            self.train_llB2rms = torch.cat((self.train_llB2rms,lB2.view(1,-1)),dim=0)

        if self.train_llxrms is None:
            self.train_llxrms = torch.tensor(lxrms,dtype=_dtype)[None, :]
        else:
            self.train_llxrms = torch.cat(( self.train_llxrms, torch.tensor(lxrms,dtype=_dtype)[None, :]), dim=0)

        if self.train_llyrms is None:
            self.train_llyrms = torch.tensor(lyrms,dtype=_dtype)[None, :]
        else:
            self.train_llyrms = torch.cat(( self.train_llyrms, torch.tensor(lyrms,dtype=_dtype)[None, :]), dim=0)
        if lxrmstol is None:
            self.train_llxrmstol = None
            self.train_llyrmstol = None
        else:
            if self.train_llxrmstol is None:
                self.train_llxrmstol = torch.tensor(lxrmstol,dtype=_dtype)[None, :]
            else:
                self.train_llxrmstol = torch.cat(( self.train_llxrmstol, torch.tensor(lxrmstol,dtype=_dtype)[None, :]), dim=0)

            if self.train_llyrmstol is None:
                self.train_llyrmstol = torch.tensor(lyrmstol,dtype=_dtype)[None, :]
            else:
                self.train_llyrmstol = torch.cat(( self.train_llyrmstol, torch.tensor(lyrmstol,dtype=_dtype)[None, :]), dim=0)

            

    def train_model(self, train_llB2=None,train_llBPMQ=None,train_llBPMQtol=None,train_llBPMQmodelerr=None,
                          xnemit_target=None,ynemit_target=None,
                          train_llB2rms=None,
                          train_llxrms=None,train_llyrms=None,
                          train_llxrmstol=None,train_llyrmstol=None,                          
                          fit_err=None,sample_model_err=None, bootstrap=None,
                          BPMQ_weight=None,PM_weight=None,
                          retrun_loss_ftn_4_debug=False):
        if train_llB2 is None:
            train_llB2 = self.train_llB2
            assert train_llBPMQ is None
            train_llBPMQ = self.train_llBPMQ
        
        if train_llBPMQtol is None:
            train_llBPMQtol = self.train_llBPMQtol 
        
        if train_llBPMQmodelerr is None:
            train_llBPMQmodelerr = self.train_llBPMQmodelerr

        if train_llBPMQmodelerr is None:
            train_llBPMQmodelerr = self.train_llBPMQmodelerr

        if train_llB2rms is None:
            train_llB2rms = self.train_llB2rms

        if train_llxrms is None:
            train_llxrms = self.train_llxrms

        if train_llyrms is None:
            train_llyrms = self.train_llyrms

        if train_llxrmstol is None:
            train_llxrmstol = self.train_llxrmstol

        if train_llyrmstol is None:
            train_llyrmstol = self.train_llyrmstol

        xnemit_target = xnemit_target or self.xnemit_target
        ynemit_target = ynemit_target or self.ynemit_target
        if self.virtual_emitprior and xnemit_target is None:  #debug
            xnemit_target = self.cs_ref[2]*(0.9+0.2*np.random.rand())
            ynemit_target = self.cs_ref[5]*(0.9+0.2*np.random.rand())

        if fit_err is None:
            fit_err = self.fit_err

        if sample_model_err is None:
            sample_model_err = self.sample_model_err

        if bootstrap is None:
            bootstrap = self.bootstrap

        #print("sample_model_err",sample_model_err)

        _ = self.model.cs_reconstruct(self.i_bpms, train_llB2, train_llBPMQ, train_llBPMQtol, train_llBPMQmodelerr,
                                  PM_i_monitors=self.i_pms, PM_llB2=train_llB2rms, 
                                  PM_xrms_targets=train_llxrms, PM_yrms_targets=train_llyrms, 
                                  PM_xrms_tolerances=train_llxrmstol, PM_yrms_tolerances=train_llyrmstol,
                                  BPMQ_weight=BPMQ_weight, PM_weight=PM_weight,
                                  xnemit_target=xnemit_target, ynemit_target=ynemit_target,
                                  batch_size=self.batch_size,
                                  n_batch_padding_factor=self.n_batch_padding_factor,
                                  num_restarts=self.num_restarts,
                                  fit_err=fit_err,
                                  sample_model_err=sample_model_err,
                                  bootstrap=bootstrap,
                                  plot_history=self.plot_history,
                                  retrun_loss_ftn_4_debug=retrun_loss_ftn_4_debug)
        if retrun_loss_ftn_4_debug:
            return _ 
        
        self.reconstructed_covs_history.append((self.model.xcovs.detach().cpu().numpy().copy(),
                                                self.model.ycovs.detach().cpu().numpy().copy()))
        self.reconstructed_cs_history.append(self.model.cs.detach().cpu().numpy())
        self.elapsed_time.append(datetime.now()-self.now)
        
        if self.cs_ref is not None:
            self.mismatch_x.append(calculate_mismatch_factor(self.cs_ref[:3],self.model.cs[:3]))
            self.mismatch_y.append(calculate_mismatch_factor(self.cs_ref[3:],self.model.cs[3:]))
            self.mmd4d.append(calculate_MMD4D(self.cs_ref.detach().cpu().numpy(),self.model.cs.detach().cpu().numpy()))
        
        if self.plot_ellipse:
            self.plot_reconstructed_ellipse(selected_cov_index=None,xlim=self.plot_xlim, ylim=self.plot_ylim)
            # self.plot_reconstructed_ellipse(selected_cov_index=self.model.selected_cov_index)
            
    def query_candidate(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.seed = self.seed + 13
        return self.model.query_candidate_quad_set_maximizing_BPMQ_var(self.i_bpms,
                                                                       llB2_penal=self.llB2_penal,
                                                                       plot_history=self.plot_history)
    
    def query_candidate_for_PMscan(self):
        return self.model.query_candidate_quad_set_maximizing_PM_var(self.i_pms,
                                                                     llB2_penal=self.llB2_penal,
                                                                     plot_history=self.plot_history)
        
    def run(self,budget):
        self.initialize()
        is_converged = False
        while(len(self.train_llB2) < budget):
            if len(self.train_llB2) == budget-1:
                is_converged = self.step(sample_model_err = False, bootstrap = False)
            else:
                is_converged = self.step()
                print("population of reconstructed ellipses are converged")
                break
            # if self.llB2_penal is not None:
            #     if len(self.llB2_penal) > 0.7*budget and len(self.train_llB2) > 0.7*budget:
            #         print(f"Queries keep seeing beam loss. Stopping after collection of {len(self.train_llB2)} train data.")
            #         break
        # if not is_converged:
        #     print(" [IMPORTANT] population of reconstructed ellipses are not yet converged")
            
    def step(self, xnemit_target=None,ynemit_target=None, fit_err=None, sample_model_err=None, bootstrap = None, dont_train_model=False):
        candidate_lB2, ensemble_std_of_BPMQ = self.query_candidate()
        is_not_useful_data = self.evaluate_candidate(candidate_lB2)
        if not is_not_useful_data and not dont_train_model:
            self.train_model(xnemit_target=xnemit_target,ynemit_target=ynemit_target, fit_err=fit_err, sample_model_err=sample_model_err, bootstrap = bootstrap)
        is_converged = ensemble_std_of_BPMQ < 0.5
        return is_converged

    def plot_reconstructed_ellipse(self,cs_ref=None,selected_cov_index=None,xlim=None,ylim=None):
        if cs_ref is None:
            cs_ref = self.cs_ref
        plot_reconstructed_ellipse(self.model,cs_ref=cs_ref,bg=self.bg,selected_cov_index=selected_cov_index,xlim=xlim,ylim=ylim)
