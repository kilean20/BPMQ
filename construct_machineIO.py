import time
import datetime
import random
import warnings
import click
from queue import Queue, Empty
from functools import partial
from typing import Optional, List, Union, Dict
from copy import deepcopy as copy
import concurrent

import numpy as np
import pandas as pd

from gui import popup_handler
from utils import warn, cyclic_mean_var, suppress_outputs
from abc import ABC, abstractmethod


import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'models'))
from BPMQ_model import BPMQ_model, raw2Q_processor

try:
    from IPython.display import display as _display
except ImportError:
    _display = print

def display(obj):
    try:
        _display(obj)
    except:
        print(obj)
    
popup_ramping_issue = popup_handler(
    "Action required", "Ramping not OK. Manually adjust PV CSETs to jitter the power supply before continuing."
)
n_popup_ramping_issue = 0

# Default configuration values
DEFAULT_CHECK_CHOPPER_BLOCKING = True

try:
    from phantasy import fetch_data as _phantasy_fetch_data
    from phantasy import ensure_set as _phantasy_ensure_set
    _phantasy_imported = True
except ImportError:
    warn("Failed to import 'phantasy'.")
    _phantasy_imported = False
    DEFAULT_CHECK_CHOPPER_BLOCKING = False
    
try:
    #from epics_tool import _epics_caget, _epics_caput, _epics_fetch_data, _epics_ensure_set 
    from epics import caget as _epics_caget
    from epics import caput as _epics_caput
    with suppress_outputs():
        if _epics_caget("REA_EXP:ELMT") is not None:
            DEFAULT_CHECK_CHOPPER_BLOCKING = False  # Skip check if machine is REA
    _epics_imported = True
except ImportError:
    warn("Failed to import 'epics_tool'.")
    _epics_imported = False
    DEFAULT_CHECK_CHOPPER_BLOCKING = False


class AbstractMachineIO(ABC):
    def __init__(self,
                 ensure_set_timeout: int = 20, 
                 fetch_data_time_span: float = 2.0,
                 resample_interval: float = 0.2,
                 check_chopper_blocking: bool = DEFAULT_CHECK_CHOPPER_BLOCKING,
                 n_popup_ramping_issue: int = n_popup_ramping_issue,
                 keep_history: bool = True,
                 verbose: bool = False
                ):
        self._ensure_set_timeout = ensure_set_timeout
        self._ensure_set_timewait_after_ramp = 0.25
        self._fetch_data_time_span = fetch_data_time_span
        self._resample_interval = resample_interval
        self._check_chopper_blocking = check_chopper_blocking
        self._n_popup_ramping_issue = n_popup_ramping_issue
        self.verbose = verbose
        self._test = False
        self.keep_history = keep_history
        self.history = {}
      
    def _record_history(self, pvname: str, timestamps, values):
        if self.keep_history:
            if pvname not in self.history:
                self.history[pvname] = {'t': [], 'v': []}
            self.history[pvname]['t'] += timestamps
            self.history[pvname]['v'] += values
      
        
    @abstractmethod
    def _caget(self, pvname: str):
        raise NotImplementedError
        
    def caget(self, pvname: str):
        now = datetime.datetime.now()
        value = self._caget(pvname)
        self._record_history(pvname,now,value)
        return value
        
    @abstractmethod
    def _caput(self, pvname: str, value: Union[float, int]):
        raise NotImplementedError

    def caput(self, pvname: str, value: Union[float, int]):
        now = datetime.datetime.now()
        self._caput(pvname, value)
        self._record_history(pvname,now,value)
 
    
    @abstractmethod
    def _ensure_set(self,
                    setpoint_pv: List[str], 
                    readback_pv: List[str], 
                    goal: List[float], 
                    tol: List[float],
                    timeout: Union[int, None] = None,
                    verbose: Union[bool, None] = None,
                    keep_data: bool = False,
                    extra_monitors: Optional[List[str]] = None,
                    fillna_method: str = 'linear',
                    **kws) -> Union[str, None]:
        return 'PutFinish', None
        

    def ensure_set(self,
                   setpoint_pv: List[str], 
                   readback_pv: List[str], 
                   goal: List[float], 
                   tol: List[float],
                   timeout: Union[int, None] = None,
                   verbose: Union[bool, None] = None,
                   resample_interval: float = None,
                   keep_data: bool = False,
                   extra_monitors: Optional[List[str]] = None,
                   fillna_method: str = 'linear',
                   **kws):
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print('Ramping in progress...')
            display(pd.DataFrame(np.array(goal).reshape(1, -1), columns=setpoint_pv))

        timeout = timeout or self._ensure_set_timeout
        ret, data = self._ensure_set(setpoint_pv,readback_pv,goal,tol,
                                     timeout=timeout,
                                     verbose=verbose,
                                     keep_data = keep_data,
                                     extra_monitors = extra_monitors,
                                     fillna_method = fillna_method,
                                     **kws,
                                     )
        self._ramping_data_wo_resample = data
        self._ramping_data = data
        time.sleep(self._ensure_set_timewait_after_ramp)
        
        if data is not None:
            resample_interval = resample_interval or self._resample_interval
            resample_interval = f'{int(1000 * resample_interval)}ms'
            data = data.resample(resample_interval).ffill().dropna()
            self._ramping_data = data
            if self.keep_history:
                for pvname in data.columns:
                    if pvname not in self.history:
                        self.history[pvname] = {'t': [], 'v': []}
                    self.history[pvname]['t'] += data.index.tolist()
                    self.history[pvname]['v'] += data[pvname].tolist()
        
        return ret, data
                

    @abstractmethod
    def _fetch_data(self, pvlist: List[str], time_span: float = None, abs_z: Union[float, None] = None, with_data: bool = False,
                    data_opt: Dict = {'with_timestamp': True}, verbose: bool = False, **kws):
        pass

    def fetch_data(self,
                   pvlist: List[str],
                   time_span: float = None, 
                   abs_z: Union[float, None] = None, 
                   with_data: bool = False,
                   resample_interval : float = None,
                   verbose: Union[bool, None] = None,
                   check_chopper_blocking: Union[bool, None] = None,
                   **kws):
        
        now = datetime.datetime.now()
        time_span = time_span or self._fetch_data_time_span
        verbose = self.verbose if verbose is None else verbose
        # Sanitize kws to avoid conflicts with explicit 'with_timestamp' in data_opt
        if kws is not None:
            if 'data_opt' in kws and isinstance(kws['data_opt'], dict):
                kws['data_opt'].pop('with_timestamp', None)

        ave, data = self._fetch_data(pvlist,
                                     time_span = time_span, 
                                     abs_z = abs_z, 
                                     with_data = True,
                                     data_opt = {'with_timestamp':True},
                                     verbose = verbose,
                                     **kws,
                                     )
        self._data_wo_resample = data
        resample_interval = resample_interval or self._resample_interval                             
        resample_interval  = str(int(1000*resample_interval ))+'ms'
        data = data.resample(resample_interval).ffill().dropna()
        self._data = data
        
        if self.keep_history:
            for pvname, val in zip(pvlist, ave):
                if pvname not in self.history:
                    self.history[pvname] = {'t': [], 'v': []}
                self.history[pvname]['t'] += data.index.tolist()  # Use data timestamps
                self.history[pvname]['v'] += data[pvname].tolist()  # Use corresponding values
        return ave, data
    
    
class construct_machineIO(AbstractMachineIO):
    def __init__(self):
        super().__init__()
        
    def _caget(self,pvname):
        if _epics_imported:
            f = _epics_caget(pvname)
        else:
            if self._test:
                warn("EPICS is not imported. caget will return fake zero")
                f = 0
            else:
                raise ValueError("EPICS is not imported. Cannot caget.")
        return f
            
    def _caput(self, pvname: str, value: Union[float, int]):
        if self._test:
            pass
        elif _epics_imported:
            _epics_caput(pvname, value)
        else:
            raise ValueError("EPICS is not imported. Cannot caput.")
      
    def _ensure_set(self,
                    setpoint_pv: List[str], 
                    readback_pv: List[str], 
                    goal: List[float], 
                    tol: List[float], 
                    timeout: Optional[int] = None,
                    verbose: Optional[bool] = None,
                    keep_data: bool = True,
                    extra_monitors: Optional[List[str]] = None,
                    fillna_method: str = 'linear',
                    **kws):
        
        t0 = time.monotonic()
        if self._test:
            ret = None
            extra_data = None
        elif _phantasy_imported:
            ret, extra_data = _phantasy_ensure_set(
                setpoint_pv, readback_pv, goal, tol, timeout,
                verbose=False,
                keep_data=keep_data,
                extra_monitors=extra_monitors,
                fillna_method=fillna_method,
                **kws
            )
        else:
            raise ValueError("Cannot change SET: PHANTASY is not imported.")
        
        if ret == "Timeout":
            if self._n_popup_ramping_issue < 2:
                popup_ramping_issue()
                self._n_popup_ramping_issue += 1
            else:
                warn("'ramping_not_OK' issued 2 times already. Ignoring 'ramping_not_OK' issue from now on...")
        
        return ret, extra_data
             
        
    def _fetch_data(self,
                    pvlist: List[str],
                    time_span: Optional[float] = None, 
                    abs_z: Optional[float] = None, 
                    verbose: Optional[bool] = None,
                    check_chopper_blocking: Optional[bool] = None,
                    **kws):
        
        check_chopper_blocking = check_chopper_blocking or self._check_chopper_blocking
        
        if check_chopper_blocking and not self._test:
            try:
                i_chopper = pvlist.index("ACS_DIAG:CHP:STATE_RD")
            except ValueError:
                pvlist = list(pvlist) + ["ACS_DIAG:CHP:STATE_RD"]
                i_chopper = -1  
            
        while True:
            if _phantasy_imported:
                ave, raw = _phantasy_fetch_data(
                    pvlist, time_span, abs_z=abs_z, verbose=False, with_data=True, **kws
                )
            else:
                raise ValueError("PHANTASY is not imported and the machineIO is not in test mode.")
                
            if check_chopper_blocking and not self._test:
                if ave[i_chopper] != 3:
                    warn("Chopper blocked during fetch_data. Re-try in 5 sec... ")
                    time.sleep(5)
                    continue
                else:
                    if i_chopper == -1:
                        pvlist = pvlist[:-1]
                        ave = ave[:-1]
                        raw.drop("ACS_DIAG:CHP:STATE_RD", inplace=True)
                        break
            else:
                break
                
        return ave, raw
        
        
class Evaluator:
    def __init__(self,
                 machineIO,
                 input_CSETs: List[str],
                 input_RDs  : List[str],
                 input_tols : Union[List[float], np.ndarray],
                 output_RDs : Optional[List[str]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False,
                 ):
        """
        Initialize the evaluator with machine I/O and relevant data sets.
        
        Parameters:
        - machineIO: An instance of the machine I/O class.
        - input_CSETs: List of control PVs.
        - input_RDs  : List of readback PVs.
        - input_tols : List or array of tolerances for each control PVs.
        - monitors   : Optional list of additional monitors.
        - ensure_set_kwargs: Optional dictionary of keyword arguments for ensure_set method.
        - fetch_data_kwargs: Optional dictionary of keyword arguments for fetch_data method.
        """
        self.machineIO = machineIO
        self.ensure_set_kwargs = ensure_set_kwargs or {}
        self.fetch_data_kwargs = fetch_data_kwargs or {}
        assert isinstance(input_CSETs, list), f"Expected input_CSETs to be of type list, but got {type(input_CSETs).__name__}"
        assert isinstance(input_RDs  , list), f"Expected input_RDs to be of type list, but got {type(input_RDs).__name__}"
        assert isinstance(input_tols , (list, np.ndarray)), f"Expected input_tols to be of type list or np.ndarray, but got {type(input_tols).__name__}"
        if output_RDs is None:
            output_RDs = []
        assert isinstance(output_RDs , list), f"Expected output_RDs to be of type list, but got {type(output_RDs).__name__}"
        self.input_CSETs = input_CSETs
        self.input_RDs   = input_RDs
        self.input_tols  = input_tols
        self.output_RDs  = output_RDs
        self.set_manually = set_manually

        self.fetch_data_monitors = list(set(input_CSETs + input_RDs + extra_monitors))
        self.ensure_set_monitors = [m for m in self.fetch_data_monitors if m not in input_RDs and m not in input_CSETs]
        
    def read(self, fetch_data_kwargs: Optional[Dict] = None):
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        ave, data = self.machineIO.fetch_data(self.fetch_data_monitors,**fetch_data_kwargs)
        return data
        
    def _set_and_read(self, x,                 
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        ):
        """
        Internal method to set the values and read the data.
        """
        ensure_set_kwargs = ensure_set_kwargs or self.ensure_set_kwargs
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        
        if self.set_manually:
            ret, ramping_data = None, None
        else:
            ret, ramping_data = self.machineIO.ensure_set(self.input_CSETs, 
                                                          self.input_RDs, 
                                                          x,
                                                          self.input_tols,
                                                          extra_monitors=self.ensure_set_monitors,
                                                          **ensure_set_kwargs)
                                                          
        ave, data = self.machineIO.fetch_data(self.fetch_data_monitors,
                                              **fetch_data_kwargs)
        return data, ramping_data

    def submit(self, x, 
        ensure_set_kwargs = None,
        fetch_data_kwargs = None,
        ):
        """
        Submit a task to set and read data asynchronously.
        """
        if self.set_manually:
            display(pd.DataFrame(x,columns=self.input_CSETs))
            input("Set the above PVs and press any key to continue...")
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._set_and_read, x, 
                                     ensure_set_kwargs = ensure_set_kwargs,
                                     fetch_data_kwargs = fetch_data_kwargs)
        return future

    def is_job_done(self, future: concurrent.futures.Future) -> bool:
        """
        Check if the submitted job is done.
        """
        return future.done()

    def get_result(self, future: concurrent.futures.Future):
        """
        Retrieve the result from the future.
        """
        data, ramping_data = future.result()
        self._data = data
        self._rampling_data = ramping_data
        return data, ramping_data


            
class Evaluator_wBPMQ(Evaluator):
    def __init__(self,
                 machineIO,
                 input_CSETs: List[str],
                 input_RDs  : List[str],
                 input_tols : Union[List[float], np.ndarray],
                 output_RDs : Optional[List[str]] = None,
                 BPM_names  : List[str] = None,
                 BPMQ_models: Dict[str,BPMQ_model] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False,
                 ):
                 
                 
        self.verbose = verbose
        if output_RDs is None:
            output_RDs = []
        else:
            assert type(output_RDs) is List

        if BPM_names is not None:
            self.raw2Q = raw2Q_processor(BPM_names=BPM_names,BPMQ_models=BPMQ_models,verbose=verbose)
            output_RDs = list(set(output_RDs + self.BPMQ_wrapper.PVs2read))

        super().__init__(machineIO, 
                         input_CSETs= input_CSETs, 
                         input_RDs  = input_RDs,
                         input_tols = input_tols,
                         output_RDs = output_RDs,
                         ensure_set_kwargs = ensure_set_kwargs,
                         fetch_data_kwargs = fetch_data_kwargs,
                         set_manually   = set_manually, 
                         )
    def read(self, fetch_data_kwargs: Optional[Dict] = None):
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        ave, data = self.machineIO.fetch_data(self.fetch_data_monitors,**fetch_data_kwargs)
        return self.raw2Q(data)
        
   
    def get_result(self, future: concurrent.futures.Future):
        """
        Retrieve the result from the future.
        """
        data, ramping_data = future.result()
        data = self.raw2Q(data)
        if ramping_data is not None:
            ramping_data = self.raw2Q(ramping_data)
        self._data = data
        self._rampling_data = ramping_data
        return data, ramping_data