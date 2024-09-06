# Standard Libraries
import re
import datetime
import time
import warnings
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
from copy import deepcopy as copy

# Third-Party Libraries
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display

# Local Libraries
from construct_machineIO import Evaluator, construct_machineIO
from machine_portal_helper import get_MPelem_from_PVnames
from utils import calculate_Brho, calculate_betagamma, get_Dnum_from_pv, sort_by_Dnum, calculate_mismatch_factor


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


_BPM_TIS161_coeffs = OrderedDict([
    ("FE_MEBT:BPM_D1056", np.array([32287, 32731, 27173, 27715])),
    ("FE_MEBT:BPM_D1072", np.array([28030, 27221, 32767, 31131])),
    ("FE_MEBT:BPM_D1094", np.array([31833, 32757, 26390, 27947])),
    ("FE_MEBT:BPM_D1111", np.array([27269, 27939, 32227, 32760])),
    ("LS1_CA01:BPM_D1129", np.array([32761, 31394, 28153, 28781])),
    ("LS1_CA01:BPM_D1144", np.array([27727, 28614, 32766, 31874])),
    ("LS1_WA01:BPM_D1155", np.array([32762, 32240, 26955, 29352])),
    ("LS1_CA02:BPM_D1163", np.array([27564, 27854, 32566, 32761])),
    ("LS1_CA02:BPM_D1177", np.array([32722, 30943, 27022, 26889])),
    ("LS1_WA02:BPM_D1188", np.array([28227, 27740, 32752, 32404])),
    ("LS1_CA03:BPM_D1196", np.array([32760, 32111, 28850, 28202])),
    ("LS1_CA03:BPM_D1211", np.array([27622, 27772, 32751, 31382])),
    ("LS1_WA03:BPM_D1222", np.array([32485, 32767, 26412, 26301])),
    ("LS1_CB01:BPM_D1231", np.array([27488, 28443, 30934, 32746])),
    ("LS1_CB01:BPM_D1251", np.array([32757, 31820, 30114, 30358])),
    ("LS1_CB01:BPM_D1271", np.array([26349, 27227, 30934, 32762])),
    ("LS1_WB01:BPM_D1286", np.array([32227, 32766, 27066, 28581])),
    ("LS1_CB02:BPM_D1295", np.array([27323, 28137, 32497, 32762])),
    ("LS1_CB02:BPM_D1315", np.array([32764, 32205, 26524, 27304])),
    ("LS1_CB02:BPM_D1335", np.array([27841, 27972, 32275, 32749])),
    ("LS1_WB02:BPM_D1350", np.array([31773, 32767, 26605, 26186])),
    ("LS1_CB03:BPM_D1359", np.array([26771, 27352, 32762, 32452])),
    ("LS1_CB03:BPM_D1379", np.array([32763, 32178, 28888, 28548])),
    ("LS1_CB03:BPM_D1399", np.array([27792, 28589, 32767, 32015])),
    ("LS1_WB03:BPM_D1413", np.array([32674, 32740, 27702, 29077])),
    ("LS1_CB04:BPM_D1423", np.array([27084, 28184, 31037, 32755])),
    ("LS1_CB04:BPM_D1442", np.array([32743, 31782, 26311, 26977])),
    ("LS1_CB04:BPM_D1462", np.array([27387, 28631, 32639, 32765])),
    ("LS1_WB04:BPM_D1477", np.array([32277, 32767, 27516, 28706])),
    ("LS1_CB05:BPM_D1486", np.array([28280, 27538, 31488, 32746])),
    ("LS1_CB05:BPM_D1506", np.array([32755, 32475, 26147, 28303])),
    ("LS1_CB05:BPM_D1526", np.array([27094, 28077, 32518, 32753])),
    ("LS1_WB05:BPM_D1541", np.array([32750, 31993, 29001, 28028])),
    ("LS1_CB06:BPM_D1550", np.array([32766, 31956, 26858, 27938])),
    ("LS1_CB06:BPM_D1570", np.array([26975, 27074, 32764, 32718])),
    ("LS1_CB06:BPM_D1590", np.array([32655, 32759, 27428, 27689])),
    ("LS1_WB06:BPM_D1604", np.array([27702, 27872, 32767, 32684])),
    ("LS1_CB07:BPM_D1614", np.array([32500, 32756, 28433, 28144])),
    ("LS1_CB07:BPM_D1634", np.array([27453, 28106, 32763, 31629])),
    ("LS1_CB07:BPM_D1654", np.array([32673, 32759, 26435, 26782])),
    ("LS1_WB07:BPM_D1668", np.array([32762, 32410, 27616, 27670])),
    ("LS1_CB08:BPM_D1677", np.array([29512, 28207, 32764, 31941])),
    ("LS1_CB08:BPM_D1697", np.array([32060, 32760, 27914, 27520])),
    ("LS1_CB08:BPM_D1717", np.array([26616, 27323, 30786, 32751])),
    ("LS1_WB08:BPM_D1732", np.array([31676, 32767, 28261, 27470])),
    ("LS1_CB09:BPM_D1741", np.array([27056, 27996, 32761, 32464])),
    ("LS1_CB09:BPM_D1761", np.array([32580, 32755, 28495, 27466])),
    ("LS1_CB09:BPM_D1781", np.array([27081, 27400, 32765, 31943])),
    ("LS1_WB09:BPM_D1796", np.array([32738, 32523, 27305, 28514])),
    ("LS1_CB10:BPM_D1805", np.array([32752, 32651, 28317, 27619])),
    ("LS1_CB10:BPM_D1825", np.array([27841, 26725, 31684, 32763])),
    ("LS1_CB10:BPM_D1845", np.array([32761, 32571, 27227, 26692])),
    ("LS1_WB10:BPM_D1859", np.array([26790, 27824, 32766, 31553])),
    ("LS1_CB11:BPM_D1869", np.array([31793, 32765, 27328, 28204])),
    ("LS1_CB11:BPM_D1889", np.array([29556, 28492, 32110, 32739])),
    ("LS1_CB11:BPM_D1909", np.array([32666, 32767, 27219, 27940])),
    ("LS1_WB11:BPM_D1923", np.array([27786, 28350, 32765, 32735])),
    ("LS1_BTS:BPM_D1967", np.array([32403, 32743, 28313, 27464])),
    ("LS1_BTS:BPM_D2027", np.array([31336, 32749, 27048, 27244])),
    ("LS1_BTS:BPM_D2054", np.array([28209, 27945, 32757, 32424])),
    ("LS1_BTS:BPM_D2116", np.array([32749, 32169, 28443, 28303])),
    ("LS1_BTS:BPM_D2130", np.array([26988, 26401, 30754, 32764])),
    ("FS1_CSS:BPM_D2212", np.array([32504, 32753, 26907, 27222])),
    ("FS1_CSS:BPM_D2223", np.array([27008, 27707, 32757, 32146])),
    ("FS1_CSS:BPM_D2248", np.array([32767, 30874, 27504, 27588])),
    ("FS1_CSS:BPM_D2278", np.array([26976, 27852, 31420, 32766])),
    ("FS1_CSS:BPM_D2313", np.array([32742, 32371, 27486, 28596])),
    ("FS1_CSS:BPM_D2369", np.array([28504, 28147, 31881, 32755])),
    ("FS1_CSS:BPM_D2383", np.array([32757, 31686, 27892, 26735])),
    ("FS1_BBS:BPM_D2421", np.array([9159, 9268, 10918, 10303])),
    ("FS1_BBS:BPM_D2466", np.array([10918, 10183, 9241, 8850])),
    ("FS1_BMS:BPM_D2502", np.array([32751, 32671, 27507, 28983])),
    ("FS1_BMS:BPM_D2537", np.array([28319, 28030, 32452, 32763])),
    ("FS1_BMS:BPM_D2587", np.array([32767, 31061, 26621, 28059])),
    ("FS1_BMS:BPM_D2600", np.array([27259, 28217, 32588, 32767])),
    ("FS1_BMS:BPM_D2665", np.array([31323, 32756, 26910, 26613])),
    ("FS1_BMS:BPM_D2690", np.array([28799, 29947, 32163, 32767])),
    ("FS1_BMS:BPM_D2702", np.array([32716, 31529, 27273, 28315])),
    ("LS2_WC01:BPM_D2742", np.array([28000, 27046, 32765, 32351])),
    ("LS2_WC02:BPM_D2782", np.array([31987, 32726, 26097, 27093])),
    ("LS2_WC03:BPM_D2821", np.array([27683, 27736, 32462, 32744])),
    ("LS2_WC04:BPM_D2861", np.array([32260, 32755, 27775, 26737])),
    ("LS2_WC05:BPM_D2901", np.array([28876, 28397, 32755, 32347])),
    ("LS2_WC06:BPM_D2941", np.array([32706, 32585, 26922, 28398])),
    ("LS2_WC07:BPM_D2981", np.array([28193, 27484, 32628, 32714])),
    ("LS2_WC08:BPM_D3020", np.array([32736, 32734, 27119, 28366])),
    ("LS2_WC09:BPM_D3060", np.array([27325, 28001, 31760, 32765])),
    ("LS2_WC10:BPM_D3100", np.array([32762, 31868, 27192, 27197])),
    ("LS2_WC11:BPM_D3140", np.array([28508, 28213, 32762, 31950])),
    ("LS2_WC12:BPM_D3180", np.array([31275, 32766, 27045, 26362])),
    ("LS2_WD01:BPM_D3242", np.array([26266, 26802, 32767, 30716])),
    ("LS2_WD02:BPM_D3304", np.array([32576, 32743, 27589, 27440])),
    ("LS2_WD03:BPM_D3366", np.array([27464, 27749, 32745, 31346])),
    ("LS2_WD04:BPM_D3428", np.array([32725, 32487, 27931, 28026])),
    ("LS2_WD05:BPM_D3490", np.array([28442, 27800, 32744, 31802])),
    ("LS2_WD06:BPM_D3552", np.array([32250, 32752, 26890, 27612])),
    ("LS2_WD07:BPM_D3614", np.array([28010, 27436, 32763, 32740])),
    ("LS2_WD08:BPM_D3676", np.array([32416, 32748, 28640, 27388])),
    ("LS2_WD09:BPM_D3738", np.array([27865, 27307, 32748, 30772])),
    ("LS2_WD10:BPM_D3800", np.array([32753, 31738, 26514, 26555])),
    ("LS2_WD11:BPM_D3862", np.array([27851, 28014, 32709, 31513])),
    ("LS2_WD12:BPM_D3924", np.array([32747, 31185, 25967, 26142])),
    ("FS2_BTS:BPM_D3943", np.array([27406, 27134, 32394, 32764])),
    ("FS2_BTS:BPM_D3958", np.array([32742, 32747, 27196, 28687])),
    ("FS2_BBS:BPM_D4019", np.array([32763, 32462, 27499, 27832])),
    ("FS2_BBS:BPM_D4054", np.array([27464, 27578, 31677, 32747])),
    ("FS2_BBS:BPM_D4087", np.array([32762, 31327, 27183, 27516])),
    ("FS2_BMS:BPM_D4142", np.array([27371, 26615, 32743, 30524])),
    ("FS2_BMS:BPM_D4164", np.array([31771, 32767, 27977, 29179])),
    ("FS2_BMS:BPM_D4177", np.array([26043, 27381, 32739, 31500])),
    ("FS2_BMS:BPM_D4216", np.array([32740, 32260, 26892, 27304])),
    ("FS2_BMS:BPM_D4283", np.array([28375, 27356, 31309, 32767])),
    ("FS2_BMS:BPM_D4326", np.array([32638, 32684, 28433, 26931])),
    ("LS3_WD01:BPM_D4389", np.array([28205, 26969, 32767, 32505])),
    ("LS3_WD02:BPM_D4451", np.array([32742, 31517, 26887, 26986])),
    ("LS3_WD03:BPM_D4513", np.array([27718, 26385, 32764, 31143])),
    ("LS3_WD04:BPM_D4575", np.array([32711, 32609, 28080, 26950])),
    ("LS3_WD05:BPM_D4637", np.array([28282, 27973, 32491, 32760])),
    ("LS3_WD06:BPM_D4699", np.array([32676, 30797, 26850, 26891])),
    ("LS3_BTS:BPM_D4753", np.array([28033, 28013, 32358, 32765])),
    ("LS3_BTS:BPM_D4769", np.array([32764, 32025, 26094, 27198])),
    ("LS3_BTS:BPM_D4843", np.array([32766, 32421, 27854, 27019])),
    ("LS3_BTS:BPM_D4886", np.array([28370, 27839, 32730, 31856])),
    ("LS3_BTS:BPM_D4968", np.array([32743, 32078, 27092, 28561])),
    ("LS3_BTS:BPM_D5010", np.array([27906, 26757, 32758, 32617])),
    ("LS3_BTS:BPM_D5092", np.array([32611, 32727, 26691, 27708])),
    ("LS3_BTS:BPM_D5134", np.array([28708, 28562, 31937, 32711])),
    ("LS3_BTS:BPM_D5216", np.array([31056, 32767, 27866, 26341])),
    ("LS3_BTS:BPM_D5259", np.array([27038, 27485, 32767, 32254])),
    ("LS3_BTS:BPM_D5340", np.array([31847, 32706, 26916, 26818])),
    ("LS3_BTS:BPM_D5381", np.array([27342, 28318, 32766, 32423])),
    ("LS3_BTS:BPM_D5430", np.array([32734, 32240, 28146, 26966])),
    ("LS3_BTS:BPM_D5445", np.array([27052, 26354, 30865, 32756])),
    ("BDS_BTS:BPM_D5499", np.array([32751, 32087, 26576, 26592])),
    ("BDS_BTS:BPM_D5513", np.array([28344, 28530, 32626, 32765])),
    ("BDS_BTS:BPM_D5565", np.array([32256, 32737, 28547, 27498])),
    ("BDS_BBS:BPM_D5625", np.array([27742, 27831, 32667, 32435])),
    ("BDS_BBS:BPM_D5653", np.array([32735, 31587, 28817, 28221])),
    ("BDS_BBS:BPM_D5680", np.array([30691, 32729, 27155, 27157])),
    ("BDS_FFS:BPM_D5742", np.array([26544, 26681, 31966, 32767])),
    ("BDS_FFS:BPM_D5772", np.array([32740, 32436, 25151, 26329])),
    ("BDS_FFS:BPM_D5790", np.array([28058, 27615, 32697, 32764])),
    ("BDS_FFS:BPM_D5803", np.array([30801, 32767, 26359, 26019])),
    ("BDS_FFS:BPM_D5818", np.array([27247, 26734, 32767, 31213])),
])

_BPM_names = list(_BPM_TIS161_coeffs.keys())


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
    
    if k>0:
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
                 dtype  = _dtype,
                 cs_ref =_cs_ref):                     
        """
        Initialize the EnvelopeEnsembleModel.

        Parameters:
        -----------
        E_MeV_u : float
            Energy per nucleon in MeV/u.
        mass_number : float
            Atomic mass number.
        charge_number : float
            Ion charge number.
        latmap : object
            Lattice map object, defaults to BDS_lattice_map_f5501_t5567.
        quads_to_scan : list
            Names of quadrupoles to scan. Must match the order in the lattice.
        B2min, B2max : list
            Min and max bounds for quadrupole strengths (T/m).
        xcovs, ycovs : torch.Tensor
            Beam covariance matrices in x and y directions.
        dtype : torch.dtype
            Data type for tensors.
        cs_ref : list
            Reference Courant-Snyder parameters.
        """
        self.latmap = latmap
        self.Brho = calculate_Brho(E_MeV_u,mass_number,charge_number)
        self.bg   = calculate_betagamma(E_MeV_u,mass_number)
        
        for elem in self.env_model.latmap.elements:
            if 'Brho' in elem.properties:
                elem.reconfigure(Brho=self.Brho)        

        self.dtype = dtype
        self.cs_ref = torch.tensor(cs_ref,dtype=self.dtype)

        if xcovs is None:
            self.xcovs, self.ycovs = noise2covar(torch.zeros(1,6,dtype=dtype),*cs_ref,bg=bg)
        else:
            self.xcovs = torch.tensor(xcovs,dtype=dtype)
            self.ycovs = torch.tensor(ycovs,dtype=dtype)
            
        self.quads, self.i_quads = [], []
        self.bpms, self.i_bpms, self.bpm_names = [], [], []
        self.pms, self.i_pms, self.pm_names = [], [], []
        for i,elem in enumerate(self.latmap.elements):
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
                        
        if B2min is None:
            B2min, B2max = [], []
            for i,iq in enumerate(self.i_quads_to_scan):
                if self.latmap.elements[iq].properties['B2'] >= 0:
                    B2min.append(2)
                    B2max.append(20)
                else:
                    B2min.append(-20)
                    B2max.append(-2)
        self.B2min = torch.tensor(B2min,dtype=self.dtype)
        self.B2max = torch.tensor(B2max,dtype=self.dtype)
        
        self.history = {
            'loss_reconstCS':[],
            'loss_queryQuad_PM':[],
            'regloss_queryQuad_PM':[],
            'loss_queryQuad_BPMQ':[],
            'regloss_queryQuad_BPMQ':[],
        } 
        
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
        xcovs.shape = (batch_size,2,2) in ver.0.
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
        """
        Simulate beam covariances for multiple quadrupole configurations.
        """
        ll_xcovs, ll_ycovs = [], []
        for lB2 in llB2:
            self.reconfigure_quadrupole_strengths(lB2)
            l_xcovs, l_ycovs = self.simulate_beam_covars(xcovs,ycovs,i_monitors)
            ll_xcovs.append(l_xcovs)
            ll_ycovs.append(l_ycovs)
        return torch.stack(ll_xcovs,dim=0),torch.stack(ll_ycovs,dim=0)   # shape of len(llB2), len(i_monitors), batch_size, 2, 2
   
    def _get_cs_reconst_loss(self,
        i_monitors, 
        iBPMQ, BPMQ_llB2, BPMQ_targets, BPMQ_tolerances, 
        iPM=None, PM_llB2=None, PM_xrms_targets=None, PM_yrms_targets=None, PM_rms_tolerances=None, 
        xnemit_target=None,
        ynemit_target=None,
        debug=False):  
        
        apertures = torch.tensor([self.latmap.elements[idx].aperture for idx in i_monitors], dtype=self.dtype)
        eval_counter = [0]
        
        def reset_eval_counter():
            eval_counter[0] = 0
            
        def loss_fun(noise_ensemble):
            eval_counter[0] += 1
            xcovs, ycovs = self.noise2covar(noise_ensemble)
            ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars(BPMQ_llB2,xcovs,ycovs,i_monitors)
            # ll_xcovs: tensor, shape: (n_scan, n_monitors, batch_size, 2, 2)
            ll_xvars, ll_yvars = ll_xcovs[:,:,:,0,0], ll_ycovs[:,:,:,0,0]

            BPMQ_sim = (ll_xvars[:,iBPMQ,:] - ll_yvars[:,iBPMQ,:])*1e6
            loss = torch.mean(torch.abs(BPMQ_sim - BPMQ_targets[:,:,None]) / BPMQ_tolerances[None,:,None], dim=[0,1])
#             print("BPMQloss",loss)
#             max_rms  = (torch.max(ll_xvars, ll_yvars).max(dim=0).values)**0.5 # (meter)
#             reg_beam_loss  = torch.relu(5*(4*max_rms/apertures[:,None] - 1)).max(dim=0).values**2 # 4 sigma, because it is hard to detect beam loss less than 1% with BPM-MAGs
#             print("reg_beam_loss",reg_beam_loss)
            if PM_llB2 is not None:
                ll_xcovs, ll_ycovs = self.multi_reconfigure_simulate_beam_covars(PM_llB2,xcovs,ycovs,i_monitors)
                ll_xvars, ll_yvars = ll_xcovs[:,:,:,0,0], ll_ycovs[:,:,:,0,0]
                xPM_sim = ll_xvars[:,iPM,:]**0.5*1e3
                yPM_sim = ll_yvars[:,iPM,:]**0.5*1e3
                PM_loss = 0.5*torch.mean(( (xPM_sim - PM_xrms_targets[:,:,None])**2 
                                          +(yPM_sim - PM_yrms_targets[:,:,None])**2 )
                                         /PM_rms_tolerances[None,:,None], dim=[0,1] )
#                 max_rms  = (torch.max(ll_xvars, ll_yvars).max(dim=0).values)**0.5 # (meter)
#                 reg_beam_loss = torch.max(reg_beam_loss, torch.relu(5*(4*max_rms/apertures[:,None] - 1)).max(dim=0).values**2 )
                if debug:
                    print(f'cs_reconst, PM_loss: {PM_loss}, reg_beam_loss: {reg_beam_loss}')
                loss = 0.7*loss + 0.3*PM_loss

            if xnemit_target is not None:
                cs = self.noise2cs(noise_ensemble)
                xnemit_sim_ratio = cs[:,2]/xnemit_target
                ynemit_sim_ratio = cs[:,5]/ynemit_target
                emit_prior_loss = (torch.relu(torch.abs(xnemit_sim_ratio - 1) - 0.2)**2 +
                                   torch.relu(torch.abs(ynemit_sim_ratio - 1) - 0.2)**2)
                loss += emit_prior_loss
                if debug:
                    print(f'cs_reconst, emit_prior_loss: {emit_prior_loss}')
            
#             loss += reg_beam_loss    
            if torch.sum(loss > 100) > 0.5*len(loss):
                raise ValueError("Loss exceeds threshold.")
#             if loss.max() > 1 and eval_counter[0]>= max_iter-2:
#                 raise ValueError() 
            return loss
        
        return loss_fun, reset_eval_counter
    
    def _cs_reconstruct_one_iter(self,batch_loss_func,batch_size,max_iter,num_restarts=100,debug=False):
        """
        Executes one iteration of the cross-section reconstruction process using L-BFGS optimization.
        """        
        if debug:
            num_restarts = 1
        for _ in range(num_restarts):
            noise_ensemble = torch.randn(batch_size, 6, dtype=self.dtype, requires_grad=True)
            optimizer = torch.optim.LBFGS([noise_ensemble],lr=0.9,max_iter=max_iter)
            # Your training loop
            def closure():
                optimizer.zero_grad()
                loss = batch_loss_func(noise_ensemble).mean()
                loss.backward()
                return loss
            if debug:
                optimizer.step(closure)
                loss = closure().item()
                noise_ensemble = noise_ensemble.detach()
                return loss, noise_ensemble
            else:
                try:
                    optimizer.step(closure)
                    loss = closure().item()
                    noise_ensemble = noise_ensemble.detach()
                    return loss, noise_ensemble
                except:
                    continue
        raise ValueError(f'cs_reconstruct failed after {num_restarts} attempts')
    
    
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
                       aperture_i_monitors: Optional[List[int]] = None, 
                       xnemit_target: Optional[float] = None, 
                       ynemit_target: Optional[float] = None,
                       batch_size: int = 16,
                       max_iter: int = 100,
                       num_restarts: int = 20,
                       debug = False,
                       ):    
        '''
        Parameters:
            llB2: tensor, shape: (n_scan, n_monitors)
            BPMQ_targets: tensor, shape: (n_scan, n_monitors)
            BPMQ_tolerances: tensor, shape: (n_monitors)
            BPMQ_i_monitors: list
            PM_xrms_targets: tensor, shape: (n_scan, n_monitors)
            PM_yrms_targets: tensor, shape: (n_scan, n_monitors)
            PM_rms_tolerances: tensor, shape: (n_monitors)
            PM_i_monitors: list
            aperture_i_monitors: list
            xemittance_target: float
            yemittance_target: float
        '''
        self.BPMQ_llB2 = BPMQ_llB2
        self.PM_llB2 = PM_llB2
        if BPMQ_tolerances is None:
            BPMQ_tolerances = torch.ones(len(BPMQ_i_monitors))
        else:
            if torch.any(BPMQ_tolerances <= 1e-6):
                raise ValueError("BPMQ_tolerances must not be negative or smaller than machine precision")
            BPMQ_tolerances = BPMQ_tolerances / BPMQ_tolerances.mean()          
        assert len(BPMQ_i_monitors) == len(BPMQ_tolerances)

        if aperture_i_monitors is None:
            aperture_i_monitors = []
            
        if PM_i_monitors is None:
            PM_i_monitors = []
        else:
            if PM_rms_tolerances is None:
                PM_rms_tolerances = torch.ones(len(PM_i_monitors))
            else:
                if torch.any(PM_rms_tolerances <= 1e-6):
                    raise ValueError("PM_rms_tolerances must not be negative or smaller than machine precision")
                PM_rms_tolerances = PM_rms_tolerances / PM_rms_tolerances.mean()

        i_monitors = sorted(list(set(BPMQ_i_monitors + PM_i_monitors + aperture_i_monitors)))
        iBPMQ = [i_monitors.index(imon) for imon in BPMQ_i_monitors]
        iPM = [i_monitors.index(imon) for imon in PM_i_monitors]

        loss_func, reset_eval_counter = self._get_cs_reconst_loss(
            i_monitors,
            iBPMQ, BPMQ_llB2, BPMQ_targets, BPMQ_tolerances, 
            iPM=iPM, PM_llB2=PM_llB2, PM_xrms_targets=PM_xrms_targets, PM_yrms_targets=PM_yrms_targets, PM_rms_tolerances=PM_rms_tolerances, 
            xnemit_target=xnemit_target,
            ynemit_target=ynemit_target,
            debug = debug,
        )

        loss, best_noise_ensemble = self._cs_reconstruct_one_iter(loss_func, batch_size, max_iter, debug=debug)

        with torch.no_grad():
            best_losses = loss_func(best_noise_ensemble)

        for i in range(num_restarts - 1):
            mask = best_losses < 0.1
            if torch.sum(mask) > min(8,0.5 * batch_size):
                break
            reset_eval_counter()
            loss, noise_ensemble = self._cs_reconstruct_one_iter(loss_func, batch_size, max_iter,debug=debug)
            with torch.no_grad():
                losses = loss_func(noise_ensemble)

            combined_losses = torch.cat((best_losses, losses))
            combined_noise_ensemble = torch.cat((best_noise_ensemble, noise_ensemble), dim=0)

            # Sort the combined losses and select the top `batch_size` values
            sorted_indices = combined_losses.argsort()[:batch_size]
            best_losses = combined_losses[sorted_indices]
#             print("best_losses",best_losses)
            best_noise_ensemble = combined_noise_ensemble[sorted_indices]

        # Final selection of noise ensembles based on the mask
        n_ens = min(8,round(0.5 * batch_size))
#         mask = best_losses < 0.1
#         if torch.sum(mask) > n_ens:
#             best_losses = best_losses[mask]
#             best_noise_ensemble = best_noise_ensemble[mask]
#         else:
        best_losses = best_losses[:n_ens]
        best_noise_ensemble = best_noise_ensemble[:n_ens]
        self.xcovs, self.ycovs = self.noise2covar(best_noise_ensemble) 
        self.xcovs_mean = self.xcovs.mean(dim=0, keepdim=True)
        self.ycovs_mean = self.ycovs.mean(dim=0, keepdim=True)
        self.cs_mean = self.covar2cs(self.xcovs_mean,self.ycovs_mean).view(-1)
        self.cs = self.noise2cs(best_noise_ensemble[:1,:]).view(-1)
        self.history['loss_reconstCS'].append(best_losses.detach().numpy())
#         print("self.noise2cs(best_noise_ensemble[:1,:])",self.noise2cs(best_noise_ensemble[:1,:]))
#         print("self.covar2cs(*self.noise2covar(best_noise_ensemble[:1,:])))",self.covar2cs(*self.noise2covar(best_noise_ensemble[:1,:])))
#         print("self.noise2cs(best_noise_ensemble[:2,:])",self.noise2cs(best_noise_ensemble[:2,:]))
#         print("self.covar2cs(*self.noise2covar(best_noise_ensemble[:2,:])))",self.covar2cs(*self.noise2covar(best_noise_ensemble[:2,:])))
        
      
    
    def _get_loss_maximize_BPMQ_var(self,i_monitors,iBPMQ,train_quad_set=None,max_iter=100):
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
            BPMQ_sim = l_xvars[iBPMQ,:]*1e6 - l_yvars[iBPMQ,:]*1e6  # (mm^2)
            loss    = -torch.mean(BPMQ_sim.std(axis=1))  # variance of BPMQ maximized for best resolving solution
            if regularize:
#                 reg_beam_loss = 1+ F.elu(10*(6*max_rms/apertures) - 10).max()
#                 reg_B2_limit = torch.mean(1+F.elu(5*(self.B2min-lB2))) + torch.mean(1+F.elu(5*(lB2-self.B2max)))
#                 reg_BPMQ_limit = torch.mean(1+F.elu(0.5*(torch.abs(BPMQ_sim)-25)))
                max_rms  = (torch.max(l_xvars, l_yvars).max(dim=1).values)**0.5 # (meter)
                reg_beam_loss  = torch.relu(5*(6*max_rms/apertures - 1)).max()**2 
                reg_B2_limit   = torch.mean(torch.relu(self.B2min-lB2)**2) + torch.mean(torch.relu(lB2-self.B2max))**2
                reg_BPMQ_limit = torch.mean(torch.relu(torch.abs(BPMQ_sim)-25))**2 
                reg =  reg_beam_loss + reg_B2_limit + reg_BPMQ_limit
    #             if train_quad_set is not None:
    #                 reg_penal = torch.relu(1 - torch.mean((lB2.view(1,-1) - train_quad_set)**2))  # as far as possbible from previous quads settings
    #                 reg = reg + reg_penal
                if reg > 9 or torch.isnan(reg):
#                     print(f"Evaluation {eval_counter[0]}: reg_beam_loss={reg_beam_loss.item()}, "
#                           f"reg_B2_limit={reg_B2_limit.item()}, reg_BPMQ_limit={reg_BPMQ_limit.item()}, loss={loss.item()}")
                    raise ValueError #"Starting point of lB2 can already cause beam loss. Restart from a new local init."
                if reg > 1 and eval_counter[0]>= max_iter-2:
                    raise ValueError #solution is not good.
                loss += reg
            return loss
        return loss_fun, reset_eval_counter
    
    def _query_candidate_quad_set_minimize_loss_once(self,loss_func,max_iter,regularize=True,num_restarts=100):        
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
        self.history['loss_queryQuad_BPMQ'].append(best_loss)
        self.history['regloss_queryQuad_BPMQ'].append(best_loss-ensemble_std_of_BPMQ)
            
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
        self.history['loss_queryQuad_PM'].append(best_loss)
        self.history['regloss_queryQuad_PM'].append(best_loss-ensemble_std_of_PM)
            
        return best_candidate_quad_set, ensemble_std_of_PM
        
        
    
class machine_BPMQ_Evaluator(Evaluator):
    def __init__(self,
        machineIO,
        input_CSETs: List[str],
        input_RDs: List[str],
        BPM_names: List[str],
        BPMQ_models:  List[torch.nn.Module] = None,
        set_manually = False
        ):
        monitor_RDs = []    
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
        dtype=_dtype
        virtual_beamQerr = 0.0,
        ):
        self.virtual_beamQerr = virtual_beamQerr
        self.cs_ref = cs_ref if cs_ref is not None else torch.concat(noise2cs(torch.randn(1,6,dtype=dtype)))
        self.env_model = EnvelopeEnsembleModel(
            E_MeV_u, mass_number, charge_number,
            latmap=latmap,                  
            quads_to_scan = quads_to_scan,
            B2min=B2min,
            B2max=B2max,
            xcovs=xcovs,
            ycovs=ycovs,
            cs_ref = self.cs_ref,
            dtype  = self.dtype,
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
        lB2 = torch.tensor(lB2,dtype=_dtype)
        self.env_model.reconfigure_quadrupole_strengths(lB2)
        l_xcovs, l_ycovs = self.simulate_beam_covars(self.env_model.xcovs,
                                                     self.env_model.ycovs,
                                                     self.env_model.i_bpms`)
        xvars, yvars = l_xcovs[:,0,0,0], l_ycovs[:,0,0,0]
        BPMQ_sim = (xvars -yvars).detach.numpy()*1e6
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
   
   
   
def plot_reconstructed_ellipse(model,cs_ref=None):
    '''
    compare reconstructed ellipses for virtual machinie
    '''
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    for cov in model.xcovs:
        plot_beam_ellipse_from_cov(cov[:,:],fig=fig,ax=ax[0])
    for cov in model.ycovs:
        plot_beam_ellipse_from_cov(cov[:,:],fig=fig,ax=ax[1])
    if cs_ref is None:
        plot_beam_ellipse(*model.cs[:3],'x',ls=':',color='k',fig=fig,ax=ax[0])
        plot_beam_ellipse(*model.cs[3:],'y',ls=':',color='k',fig=fig,ax=ax[1])
    else:
        plot_beam_ellipse(*cs_ref[:3],'x',color='k',fig=fig,ax=ax[0])
        plot_beam_ellipse(*cs_ref[3:],'y',color='k',fig=fig,ax=ax[1])
        mis_x = calculate_mismatch_factor(cs_ref[:3],model.cs[:3])
        mis_y = calculate_mismatch_factor(cs_ref[3:],model.cs[3:])
        plot_beam_ellipse(*model.cs[:3],'x',ls=':',color='k',fig=fig,ax=ax[0],label=f'{mis_x:.2f}')
        plot_beam_ellipse(*model.cs[3:],'y',ls=':',color='k',fig=fig,ax=ax[1],label=f'{mis_y:.2f}')
    
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
                    lB2 = [q.properties['B2'] for q in self.machine.quads_to_scan]
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
        
        
    def evaluate_candidate(self,lB2):
#         if type(lB2) is torch.Tensor:
#             lB2 = lB2.detach().numpy().flatten()
        quad_Iset = [self.mp_quads_to_scan[i].convert(b2,from_field='B2',to_field='I') for i,b2 in enumerate(lB2)]
                     
        set_manually = set_manually or self.set_manually
        wait_before_measure = wait_before_measure or self.wait_before_measure
        
        if self.set_manually and self.machineIO:
            input('set the followings quads')
            display(pd.DataFrame(quad_Iset,columns=self.quads_to_scan))
            ave, data = self.machine.machineIO.fetch_data(self.machine.monitor_RDs)
            ramping_data = None
        else:
            future = self.machine.submit(quad_Iset)
            if wait_before_measure:
                input("Press Enter to continue...")
            data,ramping_data = self.machine.get_result(future)
            
        # use readback instead of set
        if self.machineIO is not None:
            lB2 = [self.mp_quads_to_scan[i].convert(data[qname+':I_RD'].mean(),from_field='I',to_field='B2') 
                    for i,qname in enumerate(quads_to_scan)]
                    
        bpm_cols = [col for col in data.columns if col.endswith(':beamQ')]
        lBPMQ = data[bpm_cols].mean()
        
        if self.verbose:
            display(pd.DataFrame(lBPMQ,columns=['']).T)

        self._concat_train_data(torch.tensor(lB2, dtype=self.dtype), torch.tensor(lBPMQ, dtype=self.dtype).view(1, -1))
    
    def concat_train_data(self,lB2,lBPMQ):
        if hasattr(self,'train_lB2'):
            self.train_llB2 = torch.concat((self.train_lB2,lB2.view(1,-1)),dim=0)
        else:
            self.train_llB2 = lB2.view(1,-1).clone()
        if hasattr(self,'train_llBPMQ'):
            self.train_llBPMQ = torch.concat((self.train_llBPMQ,lBPMQ.view(1,1,-1)),dim=0)
        else:
            self.train_llBPMQ = lBPMQ.view(1,1,-1).clone()

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
        
        
    def run(self,budget,cs_ref=None):
        self.initialize()
        while(len(self.train_llB2) < budget):
            candidate_llB2, ensemble_std_of_BPMQ = self.query_candidate()
            self.evaluate_candidate(candidate_llB2)
            self.train_model()
            if self.machineIO is None and cs_ref is None:
                cs_ref = self.machine.cs_ref
            plot_reconstructed_ellipse(self.model,cs_ref=cs_ref)
            plt.show()