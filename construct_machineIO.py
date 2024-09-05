import time
import datetime
import random
import warnings
import click
from queue import Queue, Empty
from functools import partial
from typing import List, Union, Dict
import concurrent

import numpy as np
import pandas as pd

from gui import popup_handler
from utils import warn, cyclic_mean_var, suppress_outputs, get_middle_row_of_group
from abc import ABC, abstractmethod
    
popup_ramping_not_OK = popup_handler("Action required", "Ramping not OK. Manually adjust PV CSETs to jitter the power suppply before continue.")
_n_popup_ramping_not_OK = 0

_ensure_set_timeout = 30
_fetch_data_time_span = 2.05
_fetch_data_resample_rate = 0.2
_check_chopper_blocking = True

try:
    from phantasy import fetch_data as _phantasy_fetch_data
    from phantasy import ensure_set as _phantasy_ensure_set
    _phantasy_imported = True
except ImportError:
    warn("import phantasy failed")
    _phantasy_imported = False
    _check_chopper_blocking = False
    
try:
    from epics_tool import _epics_caget, _epics_caput, _epics_fetch_data, _epics_ensure_set 
    with suppress_outputs():
        if _epics_caget("REA_EXP:ELMT") is not None:
            _check_chopper_blocking = False  # don't check FRIB chopper if machine is REA
    _epics_imported = True
except ImportError:
    warn("import epics failed")
    _epics_imported = False
    _check_chopper_blocking = False



class AbstractMachineIO(ABC):
    def __init__(self,
                 _ensure_set_timeout = _ensure_set_timeout, 
                 _fetch_data_time_span = _fetch_data_time_span,
                 _fetch_data_resample_rate = _fetch_data_resample_rate,
                 _check_chopper_blocking = _check_chopper_blocking,
                 _n_popup_ramping_not_OK = _n_popup_ramping_not_OK,
                ):
        self._ensure_set_timeout = _ensure_set_timeout
        self._ensure_set_timewait_after_ramp = 0.25
        self._fetch_data_time_span = _fetch_data_time_span
        self._fetch_data_resample_rate = _fetch_data_resample_rate
        self._return_obj_var = False
        self._check_chopper_blocking = _check_chopper_blocking
        self._n_popup_ramping_not_OK = _n_popup_ramping_not_OK
        self._verbose = False
        self._test = False
        self.history = {}
        
#     def view(self):
#         for k,v in vars(self).items():
#             if k not in ['caget','caput','ensure_set','fetch_data']:
#                 print("  ",k,":",v)
        
    @abstractmethod
    def _caget(self,pvname):
        pass
        
    def caget(self,pvname):
        now = datetime.datetime.now()
        value = self._caget(pvname)
        if not pvname in self.history:
            self.history[pvname] = {'t':[],'v':[]}
#         self.history[pvname].append(f)
        self.history[pvname]['t'].append(now)
        self.history[pvname]['v'].append(value)
        
        return value
        
    @abstractmethod
    def _caput(self,pvname,value):
        pass
    
    def caput(self,pvname,value):
        now = datetime.datetime.now()
        self._caput(pvname,value)
        if not pvname in self.history:
            self.history[pvname] = {'t':[],'v':[]}
        self.history[pvname]['t'].append(now)
        self.history[pvname]['v'].append(value)        
    
    @abstractmethod
    def _ensure_set(self,
                   setpoint_pv,readback_pv,goal,
                   tol=0.01,
                   timeout = None,
                   verbose =None,
                   keep_data: bool = False,
                   extra_monitors: List[str] = None,
                   fillna_method: str = 'linear',
                   **kws):
         return 'PutFinish', None

    def ensure_set(self,
                   setpoint_pv,readback_pv,goal,
                   tol=0.01,
                   timeout=None,
                   verbose=None,
                   keep_data: bool = False,
                   extra_monitors: List[str] = None,
                   fillna_method: str = 'linear',
                   **kws):
        
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print('ramping...')
            display(pd.DataFrame(np.array(goal).reshape(1,-1), 
                                 columns=setpoint_pv))

        timeout = timeout or self._ensure_set_timeout
        ret, extra_data = self._ensure_set(setpoint_pv,readback_pv,goal,
                                           tol=tol,
                                           timeout=timeout,
                                           verbose=verbose,
                                           keep_data = keep_data,
                                           extra_monitors = extra_monitors,
                                           fillna_method = 'linear',
                                           **kws,                                   
                                           )
        time.sleep(self._ensure_set_timewait_after_ramp)
        
        # now = datetime.datetime.now()
        # for pvname,val in zip(setpoint_pv, goal):
            # if not pvname in self.history:
                # self.history[pvname] = {'t':[],'v':[]}
            # self.history[pvname]['t'].append(now)
            # self.history[pvname]['v'].append(value)
            # 
            
        for pvname in extra_data.columns:
            if not pvname in self.history:
                self.history[pvname] = {'t':[],'v':[]}
            self.history[pvname]['t'] += extra_data.index.tolist()
            self.history[pvname]['v'] += extra_data[pvname].tolist()
        
        return ret, extra_data
                

    @abstractmethod
    def _fetch_data(self,pvlist,
                    time_span = None, 
                    abs_z = None, 
                    with_data = False,
                    data_opt = {'with_timestamp':True},
                    verbose = None,
                    **kws):
        pass

    def fetch_data(self,pvlist,
                   time_span = None, 
                   abs_z = None, 
                   with_data = False,
                   resample_rate = None,
                   verbose = None,
                   check_chopper_blocking = None,
                   debug = False,
                   **kws,
                   ):
        
        now = datetime.datetime.now()
        time_span = time_span or self._fetch_data_time_span
        resample_rate = resample_rate or self._fetch_data_resample_rate
        verbose = verbose or self._verbose
                
        ave, raw = self._fetch_data(pvlist,
                                    time_span = time_span, 
                                    abs_z = abs_z, 
                                    with_data = True,
                                    timeout = timeout,
                                    data_opt = {'with_timestamp':True},
                                    **kws,
                                    )
                                    
        raw.index = raw.index.round(str(int(1000*resample_rate))+'ms')
        raw = raw.groupby(raw.index).apply(get_middle_row_of_group)
        
        for pvname, val in zip(pvlist, ave, err):
            if not pvname in self.history:
                self.history[pvname] = {'t':[],'v':[]}
            self.history[pvname]['t'].append(now)
            self.history[pvname]['v'].append(val)
        
        if verbose:
            print(f'fetched data:')
            display(pd.DataFrame(np.array(ave).reshape(1,-1), columns=pvlist))
            
        return ave, raw
    
    
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
                raise ValueError("EPICS is not imported. cannot caget")
        return f
            
    def _caput(self,pvname,value):
        if self._test:
            pass
        elif _epics_imported:
            _epics_caput(pvname,value)
        else:
            raise ValueError("EPICS is not imported. cannot caput")
      
    def _ensure_set(self,
                   setpoint_pv,readback_pv,goal,
                   tol=0.01,
                   timeout=None,
                   verbose=None,
                   keep_data: bool = True,
                   extra_monitors: List[str] = None,
                   fillna_method: str = 'linear',
                   **kws):
        
        t0 = time.monotonic()
        if self._test:
            ret = None,
            extra_data = None  #ToDo?
        elif _phantasy_imported:
            ret, extra_data = _phantasy_ensure_set(
                setpoint_pv,readback_pv,goal,tol,timeout,
                verbose=False,
                keep_data=keep_data,
                extra_monitors = extra_monitors,
                fillna_method = fillna_method,
                **kws,
                )
        elif _epics_imported:
            ret, extra_data = _epics_ensure_set(
                setpoint_pv,readback_pv,goal,tol,timeout,
                verbose=False,
                keep_data=keep_data,
                extra_monitors = extra_monitors,
                fillna_method = fillna_method,
                **kws,
                )
        else:
            raise ValueError("Cannot change SET: PHANTASY or EPICS is not imported.")
        
        if time.monotonic() - t0 > timeout: 
            if self._n_popup_ramping_not_OK<2:
                popup_ramping_not_OK()
                self._n_popup_ramping_not_OK +=1
            else:
                warn("'ramping_not_OK' issued 2 times already. Ignoring 'ramping_not_OK' issue from now on...")
        
        return ret, extra_data
                  

  
    def _fetch_data(self,pvlist,
                   time_span = None, 
                   abs_z = None, 
                   resample_rate = None,
                   verbose = None,
                   check_chopper_blocking = None,
                   debug = False,
                   **kws,
                   ):
        
        check_chopper_blocking = check_chopper_blocking or self._check_chopper_blocking
        if check_chopper_blocking and not self._test :
            try:
                i_chopper = pvlist.index("ACS_DIAG:CHP:STATE_RD")
            except ValueError:
                pvlist = list(pvlist) + ["ACS_DIAG:CHP:STATE_RD"]
                i_chopper = -1  
            
        while(True):
            if debug:
                print(  '_phantasy_imported, _epics_imported',_phantasy_imported, _epics_imported)
                print(  'pvlist', pvlist)
                
            if _phantasy_imported:
                ave,raw = _phantasy_fetch_data(pvlist,time_span,abs_z,resample_rate=resample_rate,verbose=False)
            elif _epics_imported:
                ave,raw =    _epics_fetch_data(pvlist,time_span,abs_z,resample_rate=resample_rate,verbose=False)
            else:
                raise ValueError("PHANTASY or EPICS is not imported and the machineIO is not in test mode.")
                
            if check_chopper_blocking and not self._test :
                if ave[i_chopper] != 3:
                    warn("Chopper blocked during fetch_data. Re-try in 5 sec... ")
                    time.sleep(5)
                    continue
                else:
                    if i_chopper == -1:
                        pvlist = pvlist[:-1]
                        ave  = ave[:-1]
                        raw.drop("ACS_DIAG:CHP:STATE_RD",inplace=True)
                        break
            else:
                break
                        
        # ToDo
        # for i,pv in enumerate(pvlist):
            # if 'PHASE' in pv:
                # if 'BPM' in pv:
                    # Lo = -90
                    # Hi =  90
                # else:
                    # Lo = -180
                    # Hi =  180
                # nsample = raw.iloc[i,-3]    
                # mean,var = cyclic_mean(raw.iloc[i,:nsample].dropna().values,Lo,Hi)
                # ave[i] = mean
                
        return ave, raw
            
# ToDo
# class construct_manual_fetch_data:
    # def __init__(self,pv_for_manual_fetch):
        # self.pv_for_manual_fetch = pv_for_manual_fetch
        # self._fetch_data_time_span = _fetch_data_time_span
        
    # def __call__(self,pvlist,
                 # time_span=None, 
                 # abs_z=None, 
                 # with_data=False,
                 # verbose=False):
        # time_span = time_span or self._fetch_data_time_span
        
        # if _phantasy_imported or _epics_imported:
            # print("=== Manual Input. Leave blank for automatic data read. ===")
        # else:
            # print("=== Manual Input: ===")
        # values = []
        # pvlist_blank = []
        # ipv_blank = []
        # for i,pv in enumerate(pvlist):
            # val = None
            # if pv in self.pv_for_manual_fetch:
                # try:
                    # val = float(input(pv + ': '))
                # except:
                    # print("Input not accepatble format")
                    # if _epics_imported:
                        # print(f"trying caget {pv}...")
                        # test = _epics_caget(pv)
                        # if test is None:
                            # while(val is None):
                                # try:
                                    # val = float(input(pv + ': '))
                                # except:
                                    # print("Input not accepatble format")
                                    # pass
            # if val is None:
                # pvlist_blank.append(pv)
                # ipv_blank.append(i)
            # values.append(val)
            
        # n_data = 2  # dummy numer of data samples
        # if len(pvlist_blank) > 0:
            # if _phantasy_imported:
                # ave,raw = _phantasy_fetch_data(pvlist_blank,time_span,abs_z,with_data=True,verbose=False)
                # n_data = raw.shape[1]-3
            # elif _epics_imported:
                # ave,raw =    _epics_fetch_data(pvlist_blank,time_span,abs_z,with_data=True,verbose=False)
                # n_data = raw.shape[1]-3
            # else:
                # print("Automatic data read failed. please input manually:")
                # for i,pv in zip(ipv_blank,pvlist_blank):
                    # val = float(input(pv))
                    # values[i] = val
                # ipv_blank = []
                # pvlist_blank = []
        
        # data = {pv:[val]*n_data for pv,val in zip(pvlist,values)}
        # for i,pv in enumerate(pvlist):
            # if i in ipv_blank:
                # data[pv].append(raw.loc[pv]['#'])
                # data[pv].append(raw.loc[pv]['mean'])
                # data[pv].append(raw.loc[pv]['std'] )
            # else:
                # mean = np.mean(data[pv])
                # std  = np.std (data[pv])
                # data[pv].append(n_data)
                # data[pv].append(mean  )
                # data[pv].append(std   )
        # index = list(np.arange(len(data[pv])))
        # index[-1] = 'std'
        # index[-2] = 'mean'
        # index[-3] = '#'
        # data = pd.DataFrame(data,index=index).T

        # return data['mean'].to_numpy(), data
        
        
class Evaluator:
    def __init__(self,
                 machineIO,
                 input_CSETs: List[str],
                 input_RDs: List[str],
                 monitor_RDs: List[str],
                 calculator = None,
                 ):
        """
        Initialize the evaluator with machine I/O and relevant data sets.
        """
        self.machineIO = machineIO
        self.input_CSETs = input_CSETs
        self.input_RDs = input_RDs
        if monitor_RDs is None:
            self.monitor_RDs = input_CSETs + input_RDs
        else:
            self.monitor_RDs = input_CSETs + input_RDs + [m for m in monitor_RDs if m not in input_RDs and m not in input_CSETs]
        self.extra_monitors = [m for m in monitor_RDs if m not in input_RDs and m not in input_CSETs]
        self.calculator = calculator
    
    def _set_and_read(self, x,                 
        ensure_set_kwargs = None,
        fetch_data_kwargs = None,
        ):
        """
        Internal method to set the values and read the data.
        """
        ensure_set_kwargs = ensure_set_kwargs or {}
        fetch_data_kwargs = fetch_data_kwargs or {}
        ret, ramping_data = self.machineIO.ensure_set(self.input_CSETs, self.input_RDs, x,
                                                      extra_monitors=self.extra_monitors,
                                                      **ensure_set_kwargs)
        ave, data = self.machineIO.fetch_data(self.monitor_RDs,
                                              **fetch_data_kwargs)
        return data, ramping_data

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

    def is_job_done(self, future):
        """
        Check if the submitted job is done.
        """
        return future.done()

    def get_result(self, future):
        """
        Retrieve the result from the future.
        """
        data, ramping_data = future.result()
        if self.calculator:
            cal_data = self.calculator(data)
            cal_ramping_data = self.calculator(ramping_data)
            data = pd.concat((data,cal_data), axis=1)
            ramping_data = pd.concat((ramping_data,cal_ramping_data), axis=1)
        return data, ramping_data