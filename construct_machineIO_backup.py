import time
import datetime
import random
import warnings
import click
from queue import Queue, Empty
from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd

from gui import popup_handler
from util import warn, cyclic_mean_var, suppress_outputs
from BPMQ import _BPM_TIS161_coeffs
from abc import ABC, abstractmethod
    
      
popup_ramping_not_OK = popup_handler("Action required", "Ramping not OK. Manually adjust PV CSETs to jitter the power suppply before continue.")
_n_popup_ramping_not_OK = 0

_ensure_set_timeout = 30
_fetch_data_time_span = 2.05
_check_chopper_blocking = True


try:
    from phantasy import fetch_data as _phantasy_fetch_data
    from phantasy import ensure_set as _phantasy_ensure_set
    _phantasy_imported = True
except:
    warn("import phantasy failed")
    _phantasy_imported = False
    _check_chopper_blocking = False
    
# try:
    # from epics import caget as _epics_caget
    # from epics import caput as _epics_caput
    # from epics import get_pv as _epics_get_pv    
    # _epics_imported = True
    
    # COLORS = ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan',
              # 'bright_red', 'bright_green', 'bright_yellow', 'bright_blue', 'bright_magenta', 'bright_cyan')    
              
    # def _epoch2human(epoch_time):
        # return time.strftime('%Y-%m-%d %H:%M:%S.%f', time.localtime(epoch_time))
        
    # class PVsAreReady(Exception):
        # pass
        
    # class PutFinishedException(Exception):
      # def __init__(self, *args, **kws):
          # super().__init__(*args, **kws)

    # with suppress_outputs():
        # if _epics_caget("REA_EXP:ELMT") is not None:
            # _check_chopper_blocking = False    # don't check FRIB chopper if machine is REA
    
    # def _epics_fetch_data(pvlist,time_span = _fetch_data_time_span, 
                          # abs_z = None, 
                          # with_data=False,
                          # verbose=False):
        # data = {pv:[] for pv in pvlist}
        # t0 = time.monotonic()
        # while (time.monotonic()-t0 < time_span):
            # for pv in pvlist:   
                # data[pv].append(caget(pv))
            # time.sleep(0.2)
        # for pv in pvlist:
            # mean = np.mean(data[pv])
            # std  = np.std (data[pv])
            # if abs_z is not None and std > 0:
                # mask = np.logical_and(mean -abs_z*std < data[pv], data[pv] < mean +abs_z*std )
                # mean = np.mean(np.array(data[pv])[mask])
                # std  = np.std (np.array(data[pv])[mask])
            # data[pv].append(len(data[pv]))
            # data[pv].append(mean)
            # data[pv].append(std )
        # index = list(np.arange(len(data[pv])))
        # index[-1] = 'std'
        # index[-2] = 'mean'
        # index[-3] = '#'
        # data = pd.DataFrame(data,index=index).T

        # return data['mean'].to_numpy(), data
    
 
 
    # def _epics_ensure_set(setpoint_pvs: List[str], 
                          # readback_pvs: List[str],
                          # goals: Union[float, List[float]],
                          # tols: Union[float, List[float]] = 0.01, 
                          # timeout: float = 10.0,
                          # verbose: bool = False, 
                          # keep_data: bool = False,
                          # extra_monitors: List[str] = None,
                          # fillna_method: str = 'linear',
                          # **kws):
        # """Set a list of PVs (setpoint_pvs), such that the readback values (readback_pvs) all reach the
        # goals within the value discrepancy of tolerance (tols), in the max time period in
        # seconds (timeout); 'keep_data', 'extra_monitors' and 'fillna_method' keyword arguments could be
        # used for extra data retrieval during the whole ensure settings procedure.
        # """
        # # if keep the data during ensure_set?
        # keep_data = kws.get('keep_data', False)
        # _extra_pvobjs = []
        # extra_data = None  # store the retrieved dataset
        # if keep_data:
            # # initial pvs for data retrieval
            # extra_pvs = setpoint_pvs + readback_pvs
            # # retrieve extra monitors
            # extra_monitors = kws.get('extra_monitors', None)
            # if extra_monitors is not None:
                # for i in extra_monitors:
                    # if i in extra_pvs:
                        # continue
                    # extra_pvs.append(i)
            # #
            # # initial the data container for extra_pvs values
            # extra_data = [[] for _ in range(len(extra_pvs))]
            # #
            # _start_daq = False
            # conn_sts = [False] * len(extra_pvs)
            # conn_q = Queue()
            # def conn_cb(idx: int, pvname: str, conn: bool, **kws):
                # if conn:
                    # conn_sts[idx] = True
                    # if all(conn_sts):
                        # conn_q.put(True)
            # def cb0(idx: int, **kws):
                # if not _start_daq:
                    # return
                # ts = kws.get('timestamp')
                # val = kws.get('value')
                # extra_data[idx].append((ts, val))
            # for _i, _pv in enumerate(extra_pvs):
                # o = _epics_get_pv(_pv, connection_callback=partial(conn_cb, _i),
                           # auto_monitor=True)
                # o.add_callback(partial(cb0, _i))
                # _extra_pvobjs.append(o)

            # t0 = time.perf_counter()
            # while True:
                # try:
                    # v = conn_q.get(timeout=timeout)
                    # if v: raise PVsAreReady
                # except Empty:
                    # print(f"Failed connecting to all PVs in {timeout:.1f}s.")
                    # if verbose:
                        # not_conn_pvs = [
                            # o.pvname for o in _extra_pvobjs if not o.connected
                        # ]
                        # click.secho(
                            # f"{len(not_conn_pvs)} PVs are not established in {timeout:.1f}s.",
                            # fg="red")
                        # click.secho("{}".format('\n'.join(not_conn_pvs)), fg="red")
                    # return
                # except PVsAreReady:
                    # if verbose:
                        # click.secho(
                            # f"Established {len(extra_pvs)} PVs in {(time.perf_counter() - t0) * 1e3:.1f}ms.",
                            # fg="green")
                    # break

        # def _fill_values():
            # # add time aligned data for all monitors
            # ts0 = time.time()
            # for i, o in enumerate(_extra_pvobjs):
                # extra_data[i].append((ts0, o.value))

        # # initial the first value, and start accumulating events
        # _fill_values()
        # _start_daq = True
        # #
        # nsize = len(setpoint_pvs)
        # fgcolors = random.choices(COLORS, k=nsize)
        # if isinstance(goals, (float, int)):
            # goals = [goals] * nsize
        # if isinstance(tols, (float, int)):
            # tols = [tols] * nsize
        # _dval = np.array([False] * nsize)

        # def is_equal(v, goal, tol):
            # return abs(v - goal) < tol

        # def cb(q, idx, **kws):
            # val = kws.get('value')
            # ts = kws.get('timestamp')
            # _dval[idx] = is_equal(val, goals[idx], tols[idx])
            # if verbose:
                # if _dval[idx]:
                    # is_reached = "[OK]"
                # else:
                    # is_reached = ""
                # click.secho(
                    # f"[{_epoch2human(ts)[:-3]}]{kws.get('pvname')} now is {val:<6g} (goal: {goals[idx]}){is_reached}",
                    # fg=fgcolors[idx])
            # q.put((_dval.all(), ts))

        # _read_pvobjs = [] # subset of _extra_pvobjs
        # q = Queue()
        # for idx, pv in enumerate(readback_pvs):
            # o = _epics_get_pv(pv)
            # o.add_callback(partial(cb, q, idx))
            # _read_pvobjs.append(o)

        # def _clear():
            # objs = _extra_pvobjs if _extra_pvobjs else _read_pvobjs
            # for i in objs:
                # i.clear_callbacks()
                # del i

        # t0 = time.time()
        # [_epics_caput(ipv, iv) for ipv, iv in zip(setpoint_pvs, goals)]
        # _dval = np.array([
            # is_equal(ipv.value, igoal, itol)
            # for ipv, igoal, itol in zip(_read_pvobjs, goals, tols)
        # ])
        # all_done = _dval.all()
        # while True:
            # try:
                # if all_done: raise PutFinishedException
                # all_done, ts = q.get(timeout=timeout)
                # if ts - t0 > timeout: raise TimeoutError
            # except Empty:
                # ret = "Empty"
                # _clear()
                # if verbose:
                    # click.secho(f"[{_epoch2human(time.time())[:-3]}]Return '{ret}'", fg='red')
                # break
            # except TimeoutError:
                # ret = "Timeout"
                # _clear()
                # if verbose:
                    # click.secho(f"[{_epoch2human(time.time())[:-3]}]Return '{ret}'", fg='yellow')
                # break
            # except PutFinishedException:
                # ret = 'PutFinished'
                # _clear()
                # if verbose:
                    # click.secho(f"[{_epoch2human(time.time())[:-3]}]Return '{ret}'", fg='green')
                # break
        # if extra_data is not None:
            # # pack_data
            # dfs = []
            # for i, row in enumerate(extra_data):
                # df = pd.DataFrame(row, columns=['timestamp', extra_pvs[i]])
                # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                # df.set_index('timestamp', inplace=True)
                # dfs.append(df)
            # dataset = pd.concat(dfs, axis=1)
            # fillna_method = kws.get('fillna_method', 'linear')
            # if fillna_method == "linear":
                # dataset = dataset.interpolate('linear')
            # elif fillna_method == "nearest":
                # dataset = dataset.nearest()
            # elif fillna_method == "ffill":
                # dataset = dataset.ffill()
            # elif fillna_method == "bfill":
                # dataset = dataset.bfill()
            # return ret, dataset
        # else:
            # return ret, extra_data


# except:
    # warn("import epics failed")
    # _epics_imported = False
    # _check_chopper_blocking = False
    

def _dummy_fetch_data(pvlist,time_span = _fetch_data_time_span, 
                      abs_z = None, 
                      with_data=False,
                      verbose=False):

    time.sleep(time_span)
    data = {pv:[0]*10 for pv in pvlist} # 10 dummy meausre
    for pv in pvlist:
        mean = np.mean(data[pv])
        std  = np.std (data[pv])
        if abs_z is not None and std > 0:
            mask = np.logical_and(mean -abs_z*std < data[pv], data[pv] < mean +abs_z*std )
            mean = np.mean(np.array(data[pv])[mask])
            std  = np.std (np.array(data[pv])[mask])
        data[pv].append(len(data[pv]))
        data[pv].append(mean)
        data[pv].append(std )
    index = list(np.arange(len(data[pv])))
    index[-1] = 'std'
    index[-2] = 'mean'
    index[-3] = '#'
    data = pd.DataFrame(data,index=index).T

    return data['mean'].to_numpy(), data


class Abstract_machineIO(ABC):
    def __init__(self,
                 _ensure_set_timeout = _ensure_set_timeout, 
                 _fetch_data_time_span = _fetch_data_time_span,
                 _check_chopper_blocking = _check_chopper_blocking,
                 _n_popup_ramping_not_OK = _n_popup_ramping_not_OK,
                ):
        self._ensure_set_timeout = _ensure_set_timeout
        self._ensure_set_timewait_after_ramp = 0.25
        self._fetch_data_time_span = _fetch_data_time_span
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
        
        now = datetime.datetime.now()
        for pvname,val in zip(setpoint_pv, goal):
            if not pvname in self.history:
                self.history[pvname] = {'t':[],'v':[]}
            self.history[pvname]['t'].append(now)
            self.history[pvname]['v'].append(val)
            
        return ret, extra_data
                

    @abstractmethod
    def _fetch_data(self,pvlist,
                    time_span = None, 
                    abs_z = None, 
                    with_data = False,
                    verbose = None):
        pass

    def fetch_data(self,pvlist,
                   time_span = None, 
                   abs_z = None, 
                   with_data = False,
                   verbose = None,
                   check_chopper_blocking = None,
                   debug = False,
                   timeout = 5):
        
        now = datetime.datetime.now()
        time_span = time_span or self._fetch_data_time_span
        verbose = verbose or self._verbose
        
        pvlist_wrap = []
        for i,pv in enumerate(pvlist):
            if 'BPM' in pv and ':Q' in pv:
                name = pv[:-2]
                pvlist_wrap += [name+":TISMAG161_"+str(i)+"_RD" for i in range(1,5)]
            else:
                pvlist_wrap.append(pv)
        
        ave, raw = self._fetch_data(pvlist_wrap,
                                    time_span = time_span, 
                                    abs_z = abs_z, 
                                    with_data = True,
                                    timeout = timeout)
        
        if len(pvlist_wrap) > len(pvlist):
            index_to_remove = []
            count = 0
            ave_ = np.zeros(len(pvlist))
            for i,pv in enumerate(pvlist):
                if 'BPM' in pv and ':Q' in pv:
                    Coeffs = _BPM_TIS161_coeffs[pv[:-2]]
                    if raw is None:
                        ave[count:count+4] *= Coeffs
                        ave_[i] = (ave[count+1] + ave[count+2] - (ave[count] + ave[count+3])) / ave[count:count+4].sum()
                    else:
#                         print("raw.shape",raw.shape)
#                         display(raw)
                        U4_names = pvlist_wrap[count:count+4]
                        index_to_remove += U4_names
                        raw.iloc[count:count+4,-2:]*= Coeffs[:,None]
                        raw.iloc[count:count+4,:-3]*= Coeffs[:,None] 
                        U0 = raw.iloc[count,:]
                        U1 = raw.iloc[count+1,:]
                        U2 = raw.iloc[count+2,:]
                        U3 = raw.iloc[count+3,:]
                        Q = ((U1+U2) - (U0+U3)) / raw.iloc[count:count+4,:].values.sum(axis=0)
#                         print("Q",Q)
#                         print("Q.shape",Q.shape)
#                         display(Q)
                        raw.loc[pv] = Q
                        ave_[i] = Q['mean']
                    count += 3
                else:
                    ave_[i] = ave[count]
                    count += 1
            ave = ave_
            if raw is not None:
                raw.drop(index_to_remove,inplace=True)          
        
        for pvname, val in zip(pvlist, ave):
            if not pvname in self.history:
                self.history[pvname] = {'t':[],'v':[]}
            self.history[pvname]['t'].append(now)
            self.history[pvname]['v'].append(val)
        
        if verbose:
            print(f'fetched data:')
            display(pd.DataFrame(np.array(ave).reshape(1,-1), columns=pvlist))
            
        return ave, raw
    
    
class construct_machineIO(Abstract_machineIO):
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
                   verbose=None):
        
        t0 = time.monotonic()
        if self._test:
            pass
        elif _phantasy_imported:
            _phantasy_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False)
        elif _epics_imported:
            _epics_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False)
        else:
            raise ValueError("Cannot change SET: PHANTASY or EPICS is not imported.")
        
        # if ramping fail, try to re-set twice 
        for i in range(2):
            if time.monotonic() - t0 > timeout: 
                warn("ramping_not_OK. trying again...")
                t0 = time.monotonic()
                if _phantasy_imported:
                    _phantasy_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False)
                elif _epics_imported:
                    _epics_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False) 
                    
        if time.monotonic() - t0 > timeout: 
            if self._n_popup_ramping_not_OK<2:
                popup_ramping_not_OK()
                self._n_popup_ramping_not_OK +=1
            else:
                warn("'ramping_not_OK' issued 2 times already. Ignoring 'ramping_not_OK' issue from now on...")
                
                
#     def _ensure_set(self,
#                    setpoint_pv,readback_pv,goal,
#                    tol=0.01,
#                    timeout=None,
#                    verbose=None):
        
#         t0 = time.monotonic()
#         if self._test:
#             pass
#         elif _phantasy_imported:
#             _phantasy_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False)
#         elif _epics_imported:
#             _epics_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False)
#         else:
#             raise ValueError("Cannot change SET: PHANTASY or EPICS is not imported.")
        
#         if time.monotonic() - t0 > timeout: 
#             warn("ramping_not_OK. trying again...")
#             t0 = time.monotonic()
#             if _phantasy_imported:
#                 _phantasy_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False)
#             elif _epics_imported:
#                 _epics_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose=False) 
#             if time.monotonic() - t0 > timeout: 
#                 if self._n_popup_ramping_not_OK<2:
#                     popup_ramping_not_OK()
#                     self._n_popup_ramping_not_OK +=1
#             else:
#                 warn("'ramping_not_OK' issued 2 times already. Ignoring 'ramping_not_OK' issue from now on...")
                
            
    def _fetch_data(self,pvlist,
                   time_span = None, 
                   abs_z = None, 
                   with_data=False,
                   verbose=None,
                   check_chopper_blocking = None,
                   debug = False,
                   timeout  = 5,
                   ):
        
        check_chopper_blocking = check_chopper_blocking or self._check_chopper_blocking
        if check_chopper_blocking and not self._test :
            pvlist = list(pvlist) + ["ACS_DIAG:CHP:STATE_RD"]
            
        while(True):
            if debug:
                print('[debug][objFuncs][machineIO][construct_machineIO]fetch_data')
                print(  '_phantasy_imported, _epics_imported',_phantasy_imported, _epics_imported)
                print(  'pvlist', pvlist)
                
            if _phantasy_imported:
                ave,raw = _phantasy_fetch_data(pvlist,time_span,abs_z,with_data=True,verbose=False, timeout=timeout)
            elif _epics_imported:
                ave,raw =    _epics_fetch_data(pvlist,time_span,abs_z,with_data=True,verbose=False)
            elif self._test:
                warn("PHANTASY or EPICS is not imported. fetch_data will return zeros")
                ave,raw =    _dummy_fetch_data(pvlist,time_span,abs_z,with_data=True,verbose=False)
            else:
                raise ValueError("PHANTASY or EPICS is not imported and the machineIO is not in test mode.")
                
            if check_chopper_blocking and not self._test :
                if ave[-1] != 3:
                    warn("Chopper blocked during fetch_data. Re-try in 5 sec... ")
                    time.sleep(5)
                    continue
                else:
                    pvlist = pvlist[:-1]
                    ave  = ave[:-1]
                    raw.drop("ACS_DIAG:CHP:STATE_RD",inplace=True)
                    break
            else:
                break
                
        if np.any(pd.isna(raw[0])):
            raise ValueError("fetch_data 0th column have NaN. re-fetch")
        
        std = raw['std'].to_numpy()
        for i,pv in enumerate(pvlist):
            if 'PHASE' in pv:
                if 'BPM' in pv:
                    Lo = -90
                    Hi =  90
                else:
                    Lo = -180
                    Hi =  180
                nsample = raw.iloc[i,-3]    
                mean,var = cyclic_mean_var(raw.iloc[i,:nsample].dropna().values,Lo,Hi)
                ave[i] = mean
                std[i] = var**0.5
                
        if with_data:
            raw['mean'] = ave
            raw['std']  = std
            return ave,raw
        else:
            return ave,None
            

class construct_manual_fetch_data:
    def __init__(self,pv_for_manual_fetch):
        self.pv_for_manual_fetch = pv_for_manual_fetch
        self._fetch_data_time_span = _fetch_data_time_span
        
    def __call__(self,pvlist,
                 time_span=None, 
                 abs_z=None, 
                 with_data=False,
                 verbose=False):
        time_span = time_span or self._fetch_data_time_span
        
        if _phantasy_imported or _epics_imported:
            print("=== Manual Input. Leave blank for automatic data read. ===")
        else:
            print("=== Manual Input: ===")
        values = []
        pvlist_blank = []
        ipv_blank = []
        for i,pv in enumerate(pvlist):
            val = None
            if pv in self.pv_for_manual_fetch:
                try:
                    val = float(input(pv + ': '))
                except:
                    print("Input not accepatble format")
                    if _epics_imported:
                        print(f"trying caget {pv}...")
                        test = _epics_caget(pv)
                        if test is None:
                            while(val is None):
                                try:
                                    val = float(input(pv + ': '))
                                except:
                                    print("Input not accepatble format")
                                    pass
            if val is None:
                pvlist_blank.append(pv)
                ipv_blank.append(i)
            values.append(val)
            
        n_data = 2  # dummy numer of data samples
        if len(pvlist_blank) > 0:
            if _phantasy_imported:
                ave,raw = _phantasy_fetch_data(pvlist_blank,time_span,abs_z,with_data=True,verbose=False)
                n_data = raw.shape[1]-3
            elif _epics_imported:
                ave,raw =    _epics_fetch_data(pvlist_blank,time_span,abs_z,with_data=True,verbose=False)
                n_data = raw.shape[1]-3
            else:
                print("Automatic data read failed. please input manually:")
                for i,pv in zip(ipv_blank,pvlist_blank):
                    val = float(input(pv))
                    values[i] = val
                ipv_blank = []
                pvlist_blank = []
        
        data = {pv:[val]*n_data for pv,val in zip(pvlist,values)}
        for i,pv in enumerate(pvlist):
            if i in ipv_blank:
                data[pv].append(raw.loc[pv]['#'])
                data[pv].append(raw.loc[pv]['mean'])
                data[pv].append(raw.loc[pv]['std'] )
            else:
                mean = np.mean(data[pv])
                std  = np.std (data[pv])
                data[pv].append(n_data)
                data[pv].append(mean  )
                data[pv].append(std   )
        index = list(np.arange(len(data[pv])))
        index[-1] = 'std'
        index[-2] = 'mean'
        index[-3] = '#'
        data = pd.DataFrame(data,index=index).T

        return data['mean'].to_numpy(), data
        
        
def is_device_connected(machineIO,pvs):
    t0 = time.monotonic()
    ave,raw = machineIO.fetch_data(self.input_CSETs,0.1,timeout=1)
    t1 = time.monotonic()
    if t1-t0 >= 1 and np.any(np.isnan(ave)):
        for pv,flag in zip(pvs,np.isnan(ave)):
            if flag:
                print(f'{pv} is not connected')
        raise ValueError()
        

class evaluator:
    """
    make funtion of machine with input CSETs and output RDs

    Parameters:
    - input_CSETs (Optional[List[str]]):.
    - input_RDs   (Optional[List[str]]): List of input PVs.
    - output_RDs  (Optional[List[str]]): List of output PVs.
    """
    def __init__(self,
                 machineIO  : Abstract_machineIO,
                 input_CSETs: [List[str]],
                 output_RDs : [List[str]],
                 input_RDs  : [List[str]] = None,
                 ):
        # Initialize decision variables and objectives
        self.input_CSETs = input_CSETs
        self.output_RDs = output_RDs
        is_device_connected(machineIO,self.input_CSETs)
        is_device_connected(machineIO,self.output_RDs)
        if input_RDs is None:
            self.input_RDs = [pv.replace('_CSET','_RD') for pv in self.input_CSETs]
            try:
                is_device_connected(machineIO,self.input_RDs)
            except:
                raise ValueError('input_RDs could not determined automatically')  

    def __call__(self, x):
        ret, extra_data = machineIO.ensure_set(self.input_CSETs,x)
        ave,raw =  = machineIO.fetch_data(self.output_RDs)
        return ave, raw['std']
        
      