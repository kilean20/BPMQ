# Standard Libraries
import re
import datetime
import time
import warnings
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Callable
from copy import deepcopy as copy
import concurrent
from torch_helper import run_torch_optimizer, plot_loss_history

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
import torch
import matplotlib.pyplot as plt

# Local Libraries
from construct_machineIO import Evaluator, construct_machineIO
from machine_portal_helper import get_MPelem_from_PVnames
from utils import calculate_Brho, calculate_betagamma, get_Dnum_from_pv, sort_by_Dnum, calculate_mismatch_factor, plot_beam_ellipse_from_cov, plot_beam_ellipse


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
_xalpha,_xbeta,_xnemit = 0.0, 4.0, 0.15*1e-6
_yalpha,_ybeta,_ynemit = 0.0, 4.0, 0.15*1e-6
_cs_ref = [_xalpha,_xbeta,_xnemit,_yalpha,_ybeta,_ynemit]


PSQ_D5501 = {"name":"BDS_BTS:PSQ_D5501",
             "type":"quadrupole",
             "B2"  : -8.847796226101002,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aperture": 0.025}
DRIFT_D5502 = {"name":"BDS_BTS:DRIFT_D5502",
               "type":"drift",
               "L"   :0.489, 
               "aperture": 0.1 }
PSQ_D5509 = {"name":"BDS_BTS:PSQ_D5509",
             "type":"quadrupole",
             "B2"  : 9.423499969160641,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aperture": 0.025}
DRIFT_D5510 = {"name":"BDS_BTS:DRIFT_D5510",
               "type":"drift",
               "L"   :0.268080814, 
               "aperture": 0.1} 
BPM_D5513 = {"name":"BDS_BTS:BPM_D5513",
             "type":"drift",
             "L"   : 0.145282 +0.705851186 +0.25710829999999996*10 +0.356979, 
             "aperture": 0.1 }
PSQ_D5552 = {"name":"BDS_BTS:PSQ_D5552",
             "type":"quadrupole",
             "B2"  : -14.624756233974265,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aperture": 0.025}
DRIFT_D5553 = {"name":"BDS_BTS:DRIFT_D5553",
               "type":"drift",
               "L"   :0.489, 
               "aperture": 0.1}
PSQ_D5559 = {"name":"BDS_BTS:PSQ_D5559",
             "type":"quadrupole",
             "B2"  : 17.174836398354252,
             "Brho":_Brho,
             "L"   : 0.261, 
             "aperture": 0.025}
DRIFT_D5560 = {"name":"BDS_BTS:DRIFT_D5560",
               "type":"drift",
               "L"   :0.2195 + 0.234072, 
               "aperture": 0.1 }
BPM_D5565 = {"name":"BDS_BTS:BPM_D5565",
             "type":"drift",
             "L"   : 0.145282, 
             "aperture": 0.1}
PM_D5567 = {"name":"BDS_BTS:PM_D5567",
            "type":"drift",
            "L"   : 0.0, 
            "aperture": 0.1}
            
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
    x0, x1, x2, x3, x4, x5 = noise[:, 0], noise[:, 1], noise[:, 2], noise[:, 3], noise[:, 4], noise[:, 5]
    xalpha_term = xalpha + 0.7 * x0
    xbeta_term = xbeta * torch.exp(x1 * 0.5)
    xnemit_term = xnemit * torch.exp(x2 * 0.2)
    yalpha_term = yalpha + 0.7 * x3
    ybeta_term = ybeta * torch.exp(x4 * 0.5)
    ynemit_term = ynemit * torch.exp(x5 * 0.2)
    return torch.stack([xalpha_term, xbeta_term, xnemit_term, yalpha_term, ybeta_term, ynemit_term], dim=1)

            
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
    # Fill in ycov
    ycov[:, 0, 0] = cs[:, 4]
    ycov[:, 0, 1] = -cs[:, 3]
    ycov[:, 1, 0] = -cs[:, 3]
    ycov[:, 1, 1] = (cs[:, 3]**2 + 1) / cs[:, 4]
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
        B2 = torch.tensor(B2)
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
    def __init__(self, name: str, type: str, aperture: float,
                 map_generator: Callable = None, **properties):
        """
        Represents a lattice element (e.g., drift, quadrupole).

        Args:
            name (str): Name of the element.
            type (str): Type of the element (e.g., 'quadrupole', 'drift').
            aperture (float): Aperture of the element.
            map_generator (Callable, optional): Function to generate element's map.
            **properties: Additional properties for the element.
        """
        self.name = name
        for t,lt in _types.items():
            if type in lt:
                type = t
                break
        self.type = type
        self.aperture = aperture
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
        
BDS_lattice_map_f5501_t5567 = LatticeMap(BDS_dicts_f5501_t5567)

class EnvelopeEnsembleModel:
    def __init__(self,
                 E_MeV_u: float,
                 mass_number: int,
                 charge_number: int,
                 latmap = BDS_lattice_map_f5501_t5567, 
                 quads_to_scan: list = None,  # List of quadrupole names for scanning
                 B2min: list = None,     # Min bounds for quadrupole strength (T/m)
                 B2max: list = None,     # Max bounds for quadrupole strength (T/m)
                 xcovs: torch.Tensor = None,
                 ycovs: torch.Tensor = None,
                 beamloss_sig_level = 6
                 dtype  = _dtype,
                 cs_ref =_cs_ref): 
        """
        latmap : object
            Lattice map object, defaults to BDS_lattice_map_f5501_t5567.
        quads_to_scan : list
            Names of quadrupoles to scan. Must match the order in the lattice.
        B2min, B2max : list
            Min and max bounds for quadrupole strengths (T/m).
        xcovs, ycovs : torch.Tensor
            Beam covariance matrices in x and y directions.
        cs_ref : list
            Reference Courant-Snyder parameters.
        """
        self.latmap = latmap
        self.Brho = calculate_Brho(E_MeV_u,mass_number,charge_number)
        self.bg   = calculate_betagamma(E_MeV_u,mass_number)
        
        for elem in self.latmap.elements:
            if 'Brho' in elem.properties:
                elem.reconfigure(Brho=self.Brho)        

        self.dtype = dtype
        self.cs_ref = torch.tensor(cs_ref,dtype=self.dtype)
        self.beamloss_sig_level = beamloss_sig_level

        if xcovs is None:
            self.xcovs, self.ycovs = noise2covar(torch.zeros(1,6,dtype=dtype),*cs_ref,bg=self.bg)
        else:
            self.xcovs = torch.tensor(xcovs,dtype=dtype)
            self.ycovs = torch.tensor(ycovs,dtype=dtype)
            
        self._initialize_lattice_indices(quads_to_scan)        
        self._initialize_B2_bounds()
                        
        # self.history = {
            # 'loss_reconstCS':[],
            # 'loss_queryQuad_PM':[],
            # 'regloss_queryQuad_PM':[],
            # 'loss_queryQuad_BPMQ':[],
            # 'regloss_queryQuad_BPMQ':[],
        # } 
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
            self.quads_to_scan, self.i_quads_to_scan = [], []
            for i,elem in enumerate(self.latmap.elements):
                if elem.type == 'quadrupole':
                    if elem.name in quads_to_scan:
                        self.quads_to_scan.append(elem)
                        self.i_quads_to_scan.append(i)
                
    def _initialize_B2_bounds(self, B2min, B2max):
        if B2min is None:
            B2min, B2max = [], []
            for iq in self.i_quads_to_scan:
                B2 = self.latmap.elements[iq].properties['B2']
                B2min.append(2 if B2 >= 0 else -20)
                B2max.append(20 if B2 >= 0 else -2)
        self.B2min = torch.tensor(B2min, dtype=self.dtype)
        self.B2max = torch.tensor(B2max, dtype=self.dtype)
        
        
    def noise2covar(self,noise):
        return noise2covar(noise,*self.cs_ref,bg=self.bg)
    
    def noise2cs(self,noise):
        return noise2cs(noise,*self.cs_ref)
    
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

    def multi_reconfigure_simulate_beam_covars(self,llB2,xcovs,ycovs,i_monitors):
        ll_xcovs, ll_ycovs = [], []
        for lB2 in llB2:
            self.reconfigure_quadrupole_strengths(lB2)
            l_xcovs, l_ycovs = self.simulate_beam_covars(xcovs,ycovs,i_monitors)
            ll_xcovs.append(l_xcovs)
            ll_ycovs.append(l_ycovs)
        return torch.stack(ll_xcovs,dim=0),torch.stack(ll_ycovs,dim=0)   # shape of len(llB2), len(i_monitors), batch_size, 2, 2
        
     
    def _calculate_beam_loss(apertures, ll_xvars, ll_yvars):
        # apertures: tensor, shape : (n_monitors)
        # ll_xcovs : tensor, shape : (n_scan, n_monitors, batch_size, 2, 2)
        max_rms  = (torch.max(ll_xvars, ll_yvars).values)**0.5 # (meter)
        return torch.relu((self.beamloss_sig_level * max_rms / apertures[:,:,None] - 1))**2  # shape of (n_scan, n_monitors, batch_size)
        
        
    def simulate_beam_loss(self,i_extra_monitors=None):
        if i_extra_monitors is None:
            i_apertures = self.i_quads
        else:
            i_apertures = sorted(set(self.i_quads).union(i_extra_monitors))
        apertures = torch.tensor([self.latmap.elements[idx].aperture for idx in i_monitors], dtype=self.dtype)
        l_xcovs, l_ycovs = self.simulate_beam_covars(i_monitors=i_monitors)
        l_xvars, l_yvars = l_xcovs[:,:,0,0], l_ycovs[:,:,0,0]
        return self._calculate_beam_loss(apertures,l_xvars.unsqueeze(0),l_yvars.unsqueeze(0)).max(dim=[0,1]).values  # shape of batch_size
        
    
    def _get_cs_reconst_loss_ftn(self,
        iBPMQ, BPMQ_llB2, BPMQ_targets, BPMQ_tolerances, 
        iPM=None, PM_llB2=None, PM_xrms_targets=None, PM_yrms_targets=None, PM_rms_tolerances=None, 
        xnemit_target=None,
        ynemit_target=None,
        compute_beam_loss = True,
        i_extra_monitors = None):  

        if compute_beam_loss:
            if i_extra_monitors None:
                i_apertures = self.i_quads
            else:
                i_apertures = sorted(set(self.i_quads).union(i_extra_monitors))
        else:
            i_apertures = []
            
        iBPMQ = sorted(iBPMQ)
        i_apertures_wBPMQ = sorted(set(iBPMQ).union(i_apertures))
        apertures_wBPMQ = torch.tensor([self.latmap.elements[idx].aperture for idx in i_apertures_wBPMQ], dtype=self.dtype)
        arg_iBPMQ = [i for i, imon in enumerate(i_apertures_wBPMQ) if imon in iBPMQ]
        if iPM is not None:
            if len(iPM) > 0:
                iPM = sorted(iPM)
                i_apertures_wPM = sorted(set(iPM).union(i_apertures))
                arg_iPM = [i for i, imon in enumerate(i_apertures_wPM) if imon in iPM]
                apertures_wPM = torch.tensor([self.latmap.elements[idx].aperture for idx in i_apertures_wPM], dtype=self.dtype)
        
        def loss_fun(noise_ensemble):
            xcovs, ycovs = self.noise2covar(noise_ensemble)
            ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars(BPMQ_llB2,xcovs,ycovs,i_apertures_wBPMQ)
            # ll_xcovs: tensor, shape: (n_scan, n_monitors, batch_size, 2, 2)
            ll_xvars, ll_yvars = ll_xcovs[:,:,:,0,0], ll_ycovs[:,:,:,0,0]
            BPMQ_sim = (ll_xvars[:,arg_iBPMQ,:] - ll_yvars[:,arg_iBPMQ,:])*1e6            
            loss = torch.mean(torch.abs(BPMQ_sim - BPMQ_targets[:,:,None]) / BPMQ_tolerances[None,:,None], dim=[0,1])  # shape of batch_size
            
            regloss_beamloss = self._calculate_beam_loss(apertures_wBPMQ, ll_xvars, ll_yvars).max(dim=[0,1]).values  # shape of batch_size

            if PM_llB2 is not None:
                ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars(PM_llB2,xcovs,ycovs,i_monitors_wPM)
                ll_xvars, ll_yvars = ll_xcovs[:,:,:,0,0], ll_ycovs[:,:,:,0,0]
                xPM_sim = ll_xvars[:,arg_iPM,:]**0.5*1e3
                yPM_sim = ll_yvars[:,arg_iPM,:]**0.5*1e3
                loss_PM = 0.5*torch.mean(( (xPM_sim - PM_xrms_targets[:,:,None])**2 
                                          +(yPM_sim - PM_yrms_targets[:,:,None])**2 )
                                         /PM_rms_tolerances[None,:,None], dim=[0,1] )
                                         
                loss += loss_PM                     
                                         
                regloss_beamloss += self._calculate_beam_loss(apertures_wPM, ll_xvars, ll_yvars)

            if xnemit_target is not None:
                cs = self.noise2cs(noise_ensemble)
                xnemit_sim_ratio = cs[:,2]/xnemit_target
                ynemit_sim_ratio = cs[:,5]/ynemit_target
                regloss_emitprior = (torch.relu(torch.abs(xnemit_sim_ratio - 1) - 0.2)**2 +
                                     torch.relu(torch.abs(ynemit_sim_ratio - 1) - 0.2)**2)
            else:
                regloss_emitprior = torch.zeros_like(loss)
                                   
            return {"loss":loss, "regloss_beamloss":regloss_beamloss,  "regloss_emitprior":regloss_emitprior}
        
        return loss_fun

    
    def cs_reconstruct(self, 
                       BPMQ_i_monitors: List[int], 
                       BPMQ_llB2: torch.Tensor, 
                       BPMQ_targets: torch.Tensor, 
                       BPMQ_tolerances: torch.Tensor = None, 
                       PM_i_monitors: Optional[List[int]] = None, 
                       PM_llB2: torch.Tensor = None, 
                       PM_xrms_targets: Optional[torch.Tensor] = None, 
                       PM_yrms_targets: Optional[torch.Tensor] = None, 
                       PM_rms_tolerances: Optional[torch.Tensor] = None, 
                       xnemit_target: Optional[float] = None, 
                       ynemit_target: Optional[float] = None,
                       batch_size: int = 16,
                       max_iter: int = 100,
                       num_restarts: int = 20,
                       verbose = True,
                       plot_loss_history = False,
                       ):    
                       
        assert set(BPMQ_i_monitors) <= set(self.i_bpms)
        self.BPMQ_llB2 = BPMQ_llB2
        self.PM_llB2 = PM_llB2
        if BPMQ_tolerances is None:
            BPMQ_tolerances = torch.ones(len(BPMQ_i_monitors))
        else:
            if torch.any(BPMQ_tolerances <= 1e-6):
                raise ValueError("BPMQ_tolerances must not be negative or smaller than machine precision")
            BPMQ_tolerances = BPMQ_tolerances / BPMQ_tolerances.mean()          
        assert len(BPMQ_i_monitors) == len(BPMQ_tolerances)

        if PM_i_monitors is not None:
            assert set(PM_i_monitors) <= set(self.i_pms)
            if PM_rms_tolerances is None:
                PM_rms_tolerances = torch.ones(len(PM_i_monitors))
            else:
                if torch.any(PM_rms_tolerances <= 1e-6):
                    raise ValueError("PM_rms_tolerances must not be negative or smaller than machine precision")
                PM_rms_tolerances = PM_rms_tolerances / PM_rms_tolerances.mean()


        loss_func = self._get_cs_reconst_loss_ftn(
            iBPMQ, BPMQ_llB2, BPMQ_targets, BPMQ_tolerances, 
            iPM=iPM, PM_llB2=PM_llB2, PM_xrms_targets=PM_xrms_targets, PM_yrms_targets=PM_yrms_targets, PM_rms_tolerances=PM_rms_tolerances, 
            xnemit_target=xnemit_target,
            ynemit_target=ynemit_target,
            compute_beam_loss = True,
        )
        loss_func_wo_beamloss = self._get_cs_reconst_loss_ftn(
            iBPMQ, BPMQ_llB2, BPMQ_targets, BPMQ_tolerances, 
            iPM=iPM, PM_llB2=PM_llB2, PM_xrms_targets=PM_xrms_targets, PM_yrms_targets=PM_yrms_targets, PM_rms_tolerances=PM_rms_tolerances, 
            xnemit_target=xnemit_target,
            ynemit_target=ynemit_target,
            compute_beam_loss = False,
        )
        
        noise_ensemble = torch.randn(8*batch_size, 6, dtype=self.dtype, requires_grad=True)
        result = run_torch_optimizer(
                                    loss_func = loss_func,
                                    x0 = noise_ensemble,
                                    max_iter = max_iter,
                                    loss_weights = {"loss":1.0, "regloss_beamloss":10.0,  "regloss_emitprior":1.0}
                                    low_fidelity_loss_func = loss_func_wo_beamloss,
                                    return_history = plot_loss_history
                                    )

        combined_losses = result.fun
        combined_noise_ensemble = result.x
        history = result.history
        if plot_loss_history:
            self.plot_loss_history(history)
            
        for i in range(num_restarts - 1):
            mask = combined_losses < 0.1
            if torch.sum(mask) > batch_size:
                break
            noise_ensemble = torch.randn(8*batch_size, 6, dtype=self.dtype, requires_grad=True)
            result = run_torch_optimizer(
                                        loss_func = loss_func,
                                        x0 = noise_ensemble,
                                        max_iter = max_iter,
                                        loss_weights = {"loss":1.0, "regloss_beamloss":10.0,  "regloss_emitprior":1.0}
                                        low_fidelity_loss_func = loss_func_wo_beamloss,
                                        return_history = plot_loss_history
                                        )

            combined_losses = result.fun
            combined_noise_ensemble = result.x
            history = result.history
            if plot_loss_history:
                self.plot_loss_history(history)
            combined_losses = torch.cat((combined_losses, losses))
            combined_noise_ensemble = torch.cat((combined_noise_ensemble, noise_ensemble), dim=0)

        # Sort the combined losses and select the top `batch_size` values
        sorted_indices = combined_losses.argsort()[:batch_size]
        best_losses = combined_losses[sorted_indices]
        best_noise_ensemble = combined_noise_ensemble[sorted_indices]

        self.xcovs, self.ycovs = self.noise2covar(best_noise_ensemble) 
        self.xcovs_mean = self.xcovs.mean(dim=0, keepdim=True)
        self.ycovs_mean = self.ycovs.mean(dim=0, keepdim=True)
        self.cs_mean = self.covar2cs(self.xcovs_mean,self.ycovs_mean).view(-1)
        self.cs = self.noise2cs(best_noise_ensemble[:1,:]).view(-1)
        #self.history['loss_reconstCS'].append(best_losses.detach().numpy())
      
    
    def _get_loss_maximize_BPMQ_var(self, iBPMQ, compute_beam_loss=True):
        if compute_beam_loss:
            i_quads = self.i_quads
        else:
            i_quads = []
        iBPMQ = sorted(iBPMQ)
        i_apertures_wBPMQ = sorted(set(iBPMQ).union(i_quads))
        apertures_wBPMQ = torch.tensor([self.latmap.elements[idx].aperture for idx in i_apertures_wBPMQ], dtype=self.dtype)
        arg_iBPMQ = [i for i, imon in enumerate(i_apertures_wBPMQ) if imon in iBPMQ]
        
        B2norm = 0.01*(self.B2max - self.B2min)  # 1% of range
                    
        def loss_fun(lB2):
            self.reconfigure_quadrupole_strengths(lB2)
            l_xcovs, l_ycovs = self.simulate_beam_covars(self.xcovs,self.ycovs,i_apertures_wBPMQ)
            l_xvars, l_yvars = l_xcovs[:,:,0,0], l_ycovs[:,:,0,0]
            BPMQ_sim = l_xvars[arg_iBPMQ,:]*1e6 - l_yvars[arg_iBPMQ,:]*1e6  # (mm^2)
            
            # maximize variance of BPMQ for best resolving solution.   torch.tensor.max(axis=..) gives tuple of max and argmax
            # loss_BPMQ_var = -torch.mean(BPMQ_sim.max(axis=1).values-BPMQ_sim.min(axis=1).values)  
            loss = -torch.mean(BPMQ_sim.var(axis=-1))*4 # use normalization factor of BPMQ error~0.5 mm^2 
            regloss_beamloss = self._calculate_beam_loss(apertures,
                                                         l_xvars.unsqueeze(0),
                                                         l_yvars.unsqueeze(0)
                                                         ).max()
            regloss_B2_limit   = torch.mean(torch.relu((self.B2min-lB2)/B2norm)**2) \
                               + torch.mean(torch.relu((lB2-self.B2max)/B2norm)**2)
            regloss_BPMQ_limit = torch.mean(torch.relu(torch.abs(BPMQ_sim)-25))**2 
            
            return {"loss":loss, "regloss_beamloss"  :regloss_beamloss,  
                                 "regloss_B2_limit"  :regloss_B2_limit,
                                 "regloss_BPMQ_limit":regloss_BPMQ_limit} 
        return loss_fun
    
    
    def _query_candidate_quad_set_minimize_ADAM(self,loss_func,max_iter,regularize=True,num_restarts=100):        
        for i in range(num_restarts):
            candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
            candidate_quad_set.requires_grad_(True)
            optimizer = torch.optim.LBFGS([candidate_quad_set],lr=0.9,max_iter=max_iter)
            def closure():
                optimizer.zero_grad()
                loss = loss_func(candidate_quad_set,regularize)
                loss.backward(retain_graph=True)
                return loss
            try:
                optimizer.step(closure)
                loss = closure().item()
                return loss, candidate_quad_set.detach()
            except:
                continue
        raise ValueError(f'query failed after {num_restarts} attempts')
    
    def query_candidate_quad_set_maximum_BPMQ_var(self,BPMQ_i_monitors,
                                                  aperture_i_monitors=None,
                                                  train_quad_set=None,
                                                  regularize=True,
                                                  verbose=False,
                                                  max_iter=100,
                                                  num_restarts=100):
        if aperture_i_monitors is None:
            aperture_i_monitors = self.i_quads
        i_monitors = sorted(list(set(BPMQ_i_monitors + aperture_i_monitors)))
        iBPMQ = [i_monitors.index(imon) for imon in BPMQ_i_monitors]  
        loss_func, reset_eval_counter = self._get_loss_maximize_BPMQ_var(
                    i_monitors, iBPMQ,
                    train_quad_set = train_quad_set,
                    max_iter = max_iter)        
        with torch.no_grad():
            tol = max([loss_func(lB2,regularize=False).item() for lB2 in self.BPMQ_llB2])
        
        best_loss = np.inf
        best_reg = 1
        best_candidate_quad_set = None
        for i in range(num_restarts):
            reset_eval_counter()
            try:
                loss, candidate_quad_set = self._query_candidate_quad_set_minimize_loss_once(loss_func,max_iter,regularize=regularize)
            except:
                continue
            if loss < best_loss:
                best_loss = loss
                best_candidate_quad_set = candidate_quad_set
            if best_loss <= tol:
                break
        if best_candidate_quad_set is None:
            best_candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
            print('candidate_quad_set could not found, using random quadset')
#             raise ValueError(candidate_quad_set could not found)
                
        ensemble_std_of_BPMQ = -loss_func(best_candidate_quad_set,regularize=False).item()

        if verbose:
            print(f"candidate_quad_set: {best_candidate_quad_set}")
            print(f"ensemble_std_of_BPMQ: {ensemble_std_of_BPMQ}")
        #self.history['loss_queryQuad_BPMQ'].append(best_loss)
        #self.history['regloss_queryQuad_BPMQ'].append(best_loss-ensemble_std_of_BPMQ)
            
        return best_candidate_quad_set, ensemble_std_of_BPMQ
    
    def _get_loss_maximize_PM_var(self,i_monitors,iPM,train_quad_set=None,max_iter=20):
        apertures = torch.tensor(
                    [self.latmap.elements[idx].aperture for idx in i_monitors],
                    dtype=self.dtype
                    )
        eval_counter = [0]
        def reset_eval_counter():
            eval_counter[0] = 0
        def loss_fun(lB2,regularize=True):
            eval_counter[0] += 1
            self.reconfigure_quadrupole_strengths(lB2)
            l_xcovs, l_ycovs = self.simulate_beam_covars(self.xcovs,self.ycovs,i_monitors)
            l_xvars, l_yvars = l_xcovs[:,:,0,0], l_ycovs[:,:,0,0]
            xvar_sim, yvar_sim = l_xvars[iPM,:]*1e6, l_yvars[iPM,:]*1e6
            loss    = -torch.mean(xvar_sim.std(axis=1)) -torch.mean(yvar_sim.std(axis=1))  # variance of BPMQ maximized for best resolving solution
            max_rms  = (torch.max(l_xvars, l_yvars).max(dim=1).values)**0.5 # (meter)
            if regularize:
                reg_beam_loss  = torch.relu(5*(6*max_rms/apertures - 1)).max()**2 
                reg_B2_limit   = torch.mean(torch.relu(self.B2min-lB2)**2) + torch.mean(torch.relu(lB2-self.B2max))**2
                reg_PM_limit = torch.mean(torch.relu(xvar_sim-25) + torch.relu(yvar_sim-25)) 
                reg =  reg_beam_loss + reg_B2_limit + reg_PM_limit
    #             if train_quad_set is not None:
    #                 reg_penal = torch.relu(1 - torch.mean((lB2.view(1,-1) - train_quad_set)**2))  # as far as possbible from previous quads settings
    #                 reg = reg + reg_penal
                if reg > 9 or torch.isnan(reg):
#                     print(f"Evaluation {eval_counter[0]}: reg_beam_loss={reg_beam_loss.item()}, "
#                           f"reg_B2_limit={reg_B2_limit.item()}, reg_PM_limit={reg_PM_limit.item()}, loss={loss.item()}")
                    raise ValueError() #"Starting point of lB2 can already cause beam loss. Restart from a new local init."
                if reg > 1 and eval_counter[0]>= max_iter-2:
#                     print(f"Evaluation {eval_counter[0]}: reg_beam_loss={reg_beam_loss.item()}, "
#                           f"reg_B2_limit={reg_B2_limit.item()}, reg_PM_limit={reg_PM_limit.item()}, loss={loss.item()}")
                    raise ValueError() #solution is not good.
                loss += reg
            return loss
        return loss_fun, reset_eval_counter

    def query_candidate_quad_set_maximum_PM_var(self,PM_i_monitors=None,
                                                aperture_i_monitors=None,
                                                train_quad_set=None,
                                                regularize=True,
                                                verbose=False,
                                                max_iter=20,
                                                num_restarts=100):
        if PM_i_monitors is None:
            PM_i_monitors = self.i_pms
        if aperture_i_monitors is None:
            aperture_i_monitors = self.i_quads
        i_monitors = sorted(list(set(PM_i_monitors + aperture_i_monitors)))
        iPM = [i_monitors.index(imon) for imon in PM_i_monitors]
        loss_func, reset_eval_counter = self._get_loss_maximize_PM_var(
            i_monitors, iPM,
            train_quad_set = train_quad_set,
            max_iter = max_iter
        )
        with torch.no_grad():
            tol = max([loss_func(lB2,regularize=False).item() for lB2 in self.BPMQ_llB2])
            if self.PM_llB2 is not None:
                tol = max(tol,max([loss_func(lB2,regularize=False).item() for lB2 in self.PM_llB2]))
            
        best_loss = np.inf
        best_reg = 1
        best_candidate_quad_set = None
        for i in range(num_restarts):
            reset_eval_counter()
            try:
                loss, candidate_quad_set = self._query_candidate_quad_set_minimize_loss_once(loss_func,max_iter,regularize=regularize)
            except:
                continue
            if loss < best_loss:
                best_loss = loss
                best_candidate_quad_set = candidate_quad_set
            if best_loss <= tol:
                break
        if best_candidate_quad_set is None:
            best_candidate_quad_set = torch.rand(len(self.quads_to_scan), dtype=self.dtype) * (self.B2max - self.B2min) + self.B2min
            print('candidate_quad_set could not found, using random quadset')
#             raise ValueError('candidate_quad_set could not found')

        ensemble_std_of_PM = -loss_func(best_candidate_quad_set,regularize=False).item()

        if verbose:
            print(f"candidate_quad_set: {best_candidate_quad_set}")
            print(f"ensemble_std_of_PM: {ensemble_std_of_PM}")
        #self.history['loss_queryQuad_PM'].append(best_loss)
        #self.history['regloss_queryQuad_PM'].append(best_loss-ensemble_std_of_PM)
            
        return best_candidate_quad_set, ensemble_std_of_PM
        
        
    
class machine_BPMQ_Evaluator(Evaluator):
    def __init__(self,
        machineIO,
        input_CSETs: List[str],
        input_RDs  : List[str],
        output_RDs : List[str],
        BPM_names  : List[str] = None,
        BPMQ_models:  List[torch.nn.Module] = None,
        monitor_RDs: List[str] = None,
        set_manually = False,
        ):
        if monitor_RDs is None:
            monitor_RDs = []
        elif type(monitor_RDs) != List:
            monitor_RDs = list(monitor_RDs)
        BPM_TIS161_PVs = []   
        BPM_TIS161_coeffs = np.zeros(4*len(self.BPM_names))
        BPM_names = sort_by_Dnum(BPM_names)
        for i,name in enumerate(self.BPM_names):
            assert name in _BPM_TIS161_coeffs.keys(), f"{name} not found in _BPM_TIS161_coeffs"
            TIS161_PVs = [f"{name}:TISMAG161_{i + 1}_RD" for i in range(4)]
            BPM_TIS161_PVs += TIS161_PVs
            BPM_TIS161_coeffs[4*i:4*(i+1)] = _BPM_TIS161_coeffs[name]
            monitor_RDs += BPM_TIS161_PVs + [
                f"{name}:{tag}" for tag in ["XPOS_RD", "YPOS_RD", "PHASE_RD", "MAG_RD", "CURRENT_RD"
                ]]
        monitor_RDs = list(set(monitor_RDs))
        self.BPM_TIS161_PVs = np.array(BPM_TIS161_PVs)
        self.BPM_TIS161_coeffs = np.array(BPM_TIS161_coeffs)
        if BPMQ_models is None:
            self.BPMQ_models = [None]*len(BPM_names)
        else:
            self.BPMQ_models = BPMQ_models
        super().__init__(machineIO, input_CSETs=input_CSETs, input_RDs=input_RDs, monitor_RDs=monitor_RDs,set_manually=set_manually)
            
    def get_result(self, future):
        """
        Retrieve the result from the future.
        """
        data, ramping_data = future.result()
        data[self.BPM_TIS161_PVs]*= self.BPM_TIS161_coeffs[None,-1]
        if ramping_data is not None:
            ramping_data[self.BPM_TIS161_PVs]*= self.BPM_TIS161_coeffs[None,-1]
        
        for i,name in enumerate(self.BPM_names):
            U = self.BPM_TIS161_PVs[4*i:4*(i+1)]
            if self.BPMQ_models[i]:
                model = self.BPMQ_models[i]
                with torch.no_grad():
                    u_ = torch.tensor(data[[U[0],U[1],U[2],U[3]]].values,dtype=model.dtype)
                    x_ = torch.tensor(data[name+':XPOS_RD'].values,dtype=model.dtype)
                    y_ = torch.tensor(data[name+':YPOS_RD'].values,dtype=model.dtype)
                    data[name+':beamQ'] = model(u_,x_,y_).item()
            else:
                diffsum = (data[[U[1],U[2]]].sum(axis=1) -data[[U[0],U[3]]].sum(axis=1)) / data[U].sum(axis=1)
                data[name+':beamQ'] = (241*diffsum - (data[name+':XPOS_RD']**2 - data[name+':YPOS_RD']**2))
        if ramping_data is not None:
            for i,name in enumerate(self.BPM_names):
                U = self.BPM_TIS161_PVs[4*i:4*(i+1)]
                if self.BPMQ_models[i]:
                    model = self.BPMQ_models[i]
                    with torch.no_grad():
                        u_ = torch.tensor(ramping_data[[U[0],U[1],U[2],U[3]]].values,dtype=model.dtype)
                        x_ = torch.tensor(ramping_data[name+':XPOS_RD'].values,dtype=model.dtype)
                        y_ = torch.tensor(ramping_data[name+':YPOS_RD'].values,dtype=model.dtype)
                        ramping_data[name+':beamQ'] = model(u_,x_,y_).item()
                else:
                    diffsum = (ramping_data[[U[1],U[2]]].sum(axis=1) -ramping_data[[U[0],U[3]]].sum(axis=1)) / ramping_data[U].sum(axis=1)
                    ramping_data[name+':beamQ'] = (241*diffsum - (ramping_data[name+':XPOS_RD']**2 - ramping_data[name+':YPOS_RD']**2))
        return data, ramping_data


class virtual_machine_BPMQ_Evaluator:
    def __init__(
        self,
        E_MeV_u, mass_number, charge_number,
        latmap = BDS_lattice_map_f5501_t5567, 
        quads_to_scan = None,    # quads names for BPMQ scan. must be in order of lattice
        B2min = None,     # min bounds in B2 (T/m)
        B2max = None,     # max bounds in B2 (T/m)
        xcovs = None,
        ycovs = None,
        cs_ref = None,
        dtype=_dtype,
        virtual_beamQerr = 0.0,
        ):
        self.virtual_beamQerr = virtual_beamQerr
        self.dtype = dtype
        self.cs_ref = cs_ref if cs_ref is not None else noise2cs(torch.randn(1,6,dtype=dtype)).view(6)
        self.env_model = EnvelopeEnsembleModel(
            E_MeV_u, mass_number, charge_number,
            latmap=latmap,                  
            quads_to_scan = quads_to_scan,
            B2min=B2min,
            B2max=B2max,
            xcovs=xcovs,
            ycovs=ycovs,
            cs_ref = self.cs_ref,
            dtype  = dtype,
            )
            
        if quads_to_scan is None:
            quads_to_scan = [q.name for q in self.env_model.quads_to_scan]
        self.mp_quads_to_scan = get_MPelem_from_PVnames(quads_to_scan)

        self.BPM_names = self.env_model.bpm_names
        BPM_TIS161_PVs = []      
        for i,name in enumerate(self.BPM_names):
            TIS161_PVs = [f"{name}:TISMAG161_{i + 1}_RD" for i in range(4)]
            BPM_TIS161_PVs += TIS161_PVs
        self.BPM_TIS161_PVs = np.array(BPM_TIS161_PVs)
        
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
            l_xcovs, l_ycovs = self.env_model.simulate_beam_covars(self.env_model.xcovs,
                                                                   self.env_model.ycovs,
                                                                   self.env_model.i_bpms)
            xvars, yvars = l_xcovs[:,0,0,0], l_ycovs[:,0,0,0]
            BPMQ_sim = (xvars -yvars).detach().numpy()*1e6
        BPMQ_sim += self.virtual_beamQerr*np.random.randn(*BPMQ_sim.shape)
        DiffSum = BPMQ_sim/241
        bpmU2 = 1 + DiffSum
        bpmU1 = 1 - DiffSum
        timestamp = []
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
            timestamps.append(pd.Timestamp(datetime.datetime.now()))
            for j, bpm in enumerate(self.BPM_names):
                data[bpm + ':TISMAG161_1_RD'][i] += bpmU1[j]
                data[bpm + ':TISMAG161_2_RD'][i] += bpmU2[j]
            time.sleep(0.1)
        data = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps), columns=data.keys())
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
    
    def get_result(self, future):
        """
        Retrieve the result from the future.
        """
        data, ramping_data = future.result()
        self._calculate_beamQ(data)
        self._calculate_beamQ(ramping_data)
        return data, ramping_data    
   
   
   
def plot_reconstructed_ellipse(model,cs_ref=None,bg=_bg):
    '''
    compare reconstructed ellipses for virtual machinie
    '''
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    for cov in model.xcovs:
        plot_beam_ellipse_from_cov(cov.detach().numpy(),fig=fig,ax=ax[0])
    for cov in model.ycovs:
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
        plot_beam_ellipse(*model.cs[:3],bg,'x',ls=':',color='k',fig=fig,ax=ax[0],label=f'{mis_x:.2f}')
        plot_beam_ellipse(*model.cs[3:],bg,'y',ls=':',color='k',fig=fig,ax=ax[1],label=f'{mis_y:.2f}')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
   
   
   
class BPMQscan:
    def __init__(self,
        E_MeV_u,mass_number,charge_number,
        quads_to_scan  = ["BDS_BTS:PSQ_D5501","BDS_BTS:PSQ_D5509","BDS_BTS:PSQ_D5552","BDS_BTS:PSQ_D5559"],
        quads_max_curr = [150,150,150,150],
        quads_min_curr = [5,5,5,5],
        xnemit_target = None,
        ynemit_target = None,
        machineIO      = None,
        dtype          = _dtype,
        virtual_beamQerr = 0.0,
        virtual_cs_ref   = None,
        verbose          = True,
        set_manually     = False,
        wait_before_measure = False,
        ):
        self.quads_to_scan = quads_to_scan
        self.mp_quads_to_scan = get_MPelem_from_PVnames(quads_to_scan)
        self.machineIO = machineIO
        self.xnemit_target = xnemit_target
        self.ynemit_target = ynemit_target
        self.dtype = dtype
        self.verbose = verbose
        self.set_manually = set_manually
        self.wait_before_measure = wait_before_measure
        self.bg = calculate_betagamma(E_MeV_u,mass_number)
        
        B2min = []
        B2max = []
        for mp_quad, min_curr, max_curr in zip(self.mp_quads_to_scan,quads_min_curr,quads_max_curr):
            b2min = mp_quad.convert(min_curr,from_field='I',to_field='B2')
            b2max = mp_quad.convert(max_curr,from_field='I',to_field='B2')
            B2min.append(min(b2min,b2max))
            B2max.append(max(b2min,b2max))
        
        
        if self.machineIO is None:
            self.machine = virtual_machine_BPMQ_Evaluator(        
                E_MeV_u, mass_number, charge_number,
                latmap = BDS_lattice_map_f5501_t5567, 
                quads_to_scan = quads_to_scan,
                B2min = None,
                B2max = None,
                xcovs = None,
                ycovs = None,
                cs_ref = virtual_cs_ref,
                dtype  = dtype,
                virtual_beamQerr = virtual_beamQerr)
        else:
            self.machine = machine_BPMQ_Evaluator(
                machineIO,
                input_CSETs = [name+':I_CSET' for name in quads_to_scan],
                input_RDs = [name+':I_RD' for name in quads_to_scan],
                BPM_names = ["BDS_BTS:BPM_D5513","BDS_BTS:BPM_D5565"],
                )

        self.model = EnvelopeEnsembleModel(
            E_MeV_u, mass_number, charge_number,
            latmap = BDS_lattice_map_f5501_t5567, 
            quads_to_scan = quads_to_scan,
            B2min = B2min,
            B2max = B2max, 
            dtype = dtype)   
            
    
    def initialize(self,lB2=None, init_llB2=None, set_manually=False, wait_before_measure=False):
        '''
        Scan quadrupole magnets with preset and measure BPMQ.
        init_llB2: preset, list of list of B2s in unit of T/m
        lB2: base for automatic preset determination. list of B2s in unit of T/m
        '''
        if init_llB2 is None:
            if lB2 is None:
                lB2 = []
                if self.machineIO is None:
                    lB2 = [q.properties['B2'] for q in self.machine.env_model.quads_to_scan]
                else:
                    quads_curr, _ = self.machineIO.fetch_data(self.self.machine.input_CSETs, 0.1)
                    lB2 = [mp_quad.convert(curr, from_field='I', to_field='B2') for mp_quad, curr in zip(self.mp_quads_to_scan, quads_curr)]
            if len(lB2) >= 2:
                init_llB2 = torch.tensor([lB2] * 4, dtype= self.dtype)  # preset q-scan
                init_llB2[0][0] *= 1.2
                init_llB2[1][0] *= 0.8
                init_llB2[2][1] *= 1.2
                init_llB2[3][1] *= 0.8
            else:
                init_llB2 = torch.tensor([lB2] * 3, dtype= self.dtype)  # preset q-scan
                init_llB2[0][0] *= 1.2
                init_llB2[2][0] *= 0.8
        else:
            init_llB2 = torch.tensor(init_llB2, dtype= self.dtype)
                    
        llB2_RDs = []
        llBPMQ  = []
        for lB2 in init_llB2:
            self.evaluate_candidate(lB2)
        self.train_model()
        
        
    def evaluate_candidate(self,lB2):
#         if type(lB2) is torch.Tensor:
#             lB2 = lB2.detach().numpy().flatten()
        quad_Iset = [self.mp_quads_to_scan[i].convert(b2,from_field='B2',to_field='I') for i,b2 in enumerate(lB2)]
        
        if self.set_manually and self.machineIO:
            input('set the followings quads')
            display(pd.DataFrame(quad_Iset,columns=self.quads_to_scan))
            ave, data = self.machine.machineIO.fetch_data(self.machine.monitor_RDs)
            ramping_data = None
        else:
            future = self.machine.submit(quad_Iset)
            if self.wait_before_measure:
                input("Press Enter to continue...")
            data,ramping_data = self.machine.get_result(future)
            
        # use readback instead of set
        if self.machineIO is not None:
            lB2 = [self.mp_quads_to_scan[i].convert(data[qname+':I_RD'].mean(),from_field='I',to_field='B2') 
                    for i,qname in enumerate(quads_to_scan)]
                    
        bpm_cols = [col for col in data.columns if col.endswith(':beamQ')]
        lBPMQ = data[bpm_cols].mean()    # lBPM is shape of (n_bpm,)
        
        if self.verbose:
            display(pd.DataFrame(lBPMQ,columns=['']).T)

        self._concat_train_data(torch.tensor(lB2, dtype=self.dtype), torch.tensor(lBPMQ, dtype=self.dtype)) # batch_size = 1
    
    def _concat_train_data(self,lB2,lBPMQ):
        if hasattr(self,'train_llB2'):
            self.train_llB2 = torch.concat((self.train_llB2,lB2.view(1,-1)),dim=0)
        else:
            self.train_llB2 = lB2.view(1,-1).clone()

        if hasattr(self,'train_llBPMQ'):
            self.train_llBPMQ = torch.concat(( self.train_llBPMQ, lBPMQ[None, :] ), dim=0)
        else:
            self.train_llBPMQ = lBPMQ[None, :]    

    def train_model(self, train_llB2=None,train_llBPMQ=None,xnemit_target=None,ynemit_target=None):
        if train_llB2 is None:
            train_llB2 = self.train_llB2
            train_llBPMQ = self.train_llBPMQ
        xnemit_target = xnemit_target or self.xnemit_target
        ynemit_target = ynemit_target or self.ynemit_target
        self.model.cs_reconstruct(self.model.i_bpms, train_llB2, train_llBPMQ,
                                  xnemit_target=xnemit_target, ynemit_target=ynemit_target)
            
    def query_candidate(self):
        return self.model.query_candidate_quad_set_maximum_BPMQ_var(self.model.i_bpms,verbose=self.verbose)
        
        
    def run(self,budget,plot=True,cs_ref=None):
        self.initialize()
        while(len(self.train_llB2) < budget):
            self.step(plot=plot,cs_ref=cs_ref)
            
            
    def step(self,plot=True,cs_ref=None):
        candidate_llB2, ensemble_std_of_BPMQ = self.query_candidate()
        self.evaluate_candidate(candidate_llB2)
        self.train_model()
        if plot:
            self.plot_reconstructed_ellipse(cs_ref=cs_ref)
            plt.show()
             
    def plot_reconstructed_ellipse(self,cs_ref=None):
        if cs_ref is None and hasattr(self.machine,'cs_ref'):
            cs_ref = self.machine.cs_ref
        plot_reconstructed_ellipse(self.model,cs_ref=cs_ref,bg=self.bg)