import time
import random
from typing import List, Tuple, Union
from functools import partial
from queue import Queue, Empty
from threading import Thread, Event
import weakref
import pandas as pd
import numpy as np
import click
from epics import PV, caget, caput, get_pv
from epics.ca import CAThread
from epics.exceptions import ChannelAccessException

_epics_imported = True

COLORS = ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan',
          'bright_red', 'bright_green', 'bright_yellow', 'bright_blue', 'bright_magenta', 'bright_cyan')    

def epoch2human(epoch: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S.%f', time.localtime(epoch))

    
class PVsAreReady(Exception):
    pass
    
class PutFinishedException(Exception):
  def __init__(self, *args, **kws):
      super().__init__(*args, **kws)
      
def _epics_ensure_set(setpoint_pv: List[str],
               readback_pv: List[str],
               goal: List[float],
               tol: List[float] = 0.01,
               timeout: float = 10.0,
               verbose: bool = False,
               keep_data: bool = False,
               extra_monitors: List[str] = None,
               fillna_method: str = 'linear'):
    assert len(set(setpoint_pv)) == len(setpoint_pv) == len(readback_pv)
    return _ensure_set_array(setpoint_pv, readback_pv, goal, tol, timeout, verbose,
                             keep_data=keep_data,
                             extra_monitors=extra_monitors,
                             fillna_method=fillna_method)

def _ensure_set_array(setpoint_pvs: List[str], readback_pvs: List[str],
                      goals: Union[float, List[float]],
                      tols: Union[float, List[float]] = 0.01, timeout: float = 10.0,
                      verbose: bool = False, **kws):
    """Set a list of PVs (setpoint_pvs), such that the readback values (readback_pvs) all reach the
    goals within the value discrepancy of tolerance (tols), in the max time period in
    seconds (timeout); 'keep_data', 'extra_monitors' and 'fillna_method' keyword arguments could be
    used for extra data retrieval during the whole ensure settings procedure.
    """
    # if keep the data during ensure_set?
    keep_data = kws.get('keep_data', False)
    _extra_pvobjs = []
    extra_data = None  # store the retrieved dataset
    if keep_data:
        # initial pvs for data retrieval
        extra_pvs = setpoint_pvs + readback_pvs
        # retrieve extra monitors
        extra_monitors = kws.get('extra_monitors', None)
        if extra_monitors is not None:
            for i in extra_monitors:
                if i in extra_pvs:
                    continue
                extra_pvs.append(i)
        # initial the data container for extra_pvs values
        extra_data = [[] for _ in range(len(extra_pvs))]
        _start_daq = False
        conn_sts = [False] * len(extra_pvs)
        conn_q = Queue()
        def conn_cb(idx: int, pvname: str, conn: bool, **kws):
            if conn:
                conn_sts[idx] = True
                if all(conn_sts):
                    conn_q.put(True)
        def cb0(idx: int, **kws):
            if not _start_daq:
                return
            ts = kws.get('timestamp')
            val = kws.get('value')
            extra_data[idx].append((ts, val))
        for _i, _pv in enumerate(extra_pvs):
            o = get_pv(_pv, connection_callback=partial(conn_cb, _i),
                       auto_monitor=True)
            o.add_callback(partial(cb0, _i))
            _extra_pvobjs.append(o)

        t0 = time.perf_counter()
        while True:
            try:
                v = conn_q.get(timeout=timeout)
                if v: raise PVsAreReady
            except Empty:
                print(f"Failed connecting to all PVs in {timeout:.1f}s.")
                if verbose:
                    not_conn_pvs = [
                        o.pvname for o in _extra_pvobjs if not o.connected
                    ]
                    click.secho(
                        f"{len(not_conn_pvs)} PVs are not established in {timeout:.1f}s.",
                        fg="red")
                    click.secho("{}".format('\n'.join(not_conn_pvs)), fg="red")
                return
            except PVsAreReady:
                if verbose:
                    click.secho(
                        f"Established {len(extra_pvs)} PVs in {(time.perf_counter() - t0) * 1e3:.1f}ms.",
                        fg="green")
                break

    def _fill_values():
        # add time aligned data for all monitors
        ts0 = time.time()
        for i, o in enumerate(_extra_pvobjs):
            extra_data[i].append((ts0, o.value))

    # initial the first value, and start accumulating events
    _fill_values()
    _start_daq = True
    #
    nsize = len(setpoint_pvs)
    fgcolors = random.choices(COLORS, k=nsize)
    if isinstance(goals, (float, int)):
        goals = [goals] * nsize
    if isinstance(tols, (float, int)):
        tols = [tols] * nsize
    _dval = np.array([False] * nsize)

    def is_equal(v, goal, tol):
        return abs(v - goal) < tol

    def cb(q, idx, **kws):
        val = kws.get('value')
        ts = kws.get('timestamp')
        _dval[idx] = is_equal(val, goals[idx], tols[idx])
        if verbose:
            if _dval[idx]:
                is_reached = "[OK]"
            else:
                is_reached = ""
            click.secho(
                f"[{epoch2human(ts)[:-3]}]{kws.get('pvname')} now is {val:<6g} (goal: {goals[idx]}){is_reached}",
                fg=fgcolors[idx])
        q.put((_dval.all(), ts))

    _read_pvobjs = [] # subset of _extra_pvobjs
    q = Queue()
    for idx, pv in enumerate(readback_pvs):
        o = get_pv(pv)
        o.add_callback(partial(cb, q, idx))
        _read_pvobjs.append(o)

    def _clear():
        objs = _extra_pvobjs if _extra_pvobjs else _read_pvobjs
        for i in objs:
            i.clear_callbacks()
            del i

    t0 = time.time()
    [epics.caput(ipv, iv) for ipv, iv in zip(setpoint_pvs, goals)]
    _dval = np.array([
        is_equal(ipv.value, igoal, itol)
        for ipv, igoal, itol in zip(_read_pvobjs, goals, tols)
    ])
    all_done = _dval.all()
    while True:
        try:
            if all_done: raise PutFinishedException
            all_done, ts = q.get(timeout=timeout)
            if ts - t0 > timeout: raise TimeoutError
        except Empty:
            ret = "Empty"
            _clear()
            if verbose:
                click.secho(f"[{epoch2human(time.time())[:-3]}]Return '{ret}'", fg='red')
            break
        except TimeoutError:
            ret = "Timeout"
            _clear()
            if verbose:
                click.secho(f"[{epoch2human(time.time())[:-3]}]Return '{ret}'", fg='yellow')
            break
        except PutFinishedException:
            ret = 'PutFinished'
            _clear()
            if verbose:
                click.secho(f"[{epoch2human(time.time())[:-3]}]Return '{ret}'", fg='green')
            break
    if extra_data is not None:
        # pack_data
        dfs = []
        for i, row in enumerate(extra_data):
            df = pd.DataFrame(row, columns=['timestamp', extra_pvs[i]])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            dfs.append(df)
        dataset = pd.concat(dfs, axis=1)
        fillna_method = kws.get('fillna_method', 'linear')
        if fillna_method == "linear":
            dataset = dataset.interpolate('linear')
        elif fillna_method == "nearest":
            dataset = dataset.nearest()
        elif fillna_method == "ffill":
            dataset = dataset.ffill()
        elif fillna_method == "bfill":
            dataset = dataset.bfill()
        return ret, dataset
    else:
        return ret, extra_data


class FetchDataFinishedException(Exception):
    pass


class DataFetcher:
    """ DataFetcher provides a more robust, flexible and efficient way for fetching data through CA.
    It's wrapping the `fetch_data` function but offers less overhead in terms of managing the
    working objects.

    Parameters
    ----------
    pvlist : List[str]
        A list of PVs with unique names.

    Keyword Arguments
    -----------------
    timeout : float
        The overall connection timeout for all PVs, defaults 5 seconds, meaning if in 5 seconds
        not all the PVs can be reached, raise an error; increase the timeout by set timeout value
        via <DataFetcher instance>.timeout = <new timeout>.
    verbose : bool
        If set, show more print out messages, defaults False.

    See Also
    --------
    fetch_data

    Examples
    --------
    >>> from phantasy import DataFetcher
    >>> pvs = [
    >>>  'VA:LS1_CA01:CAV1_D1127:PHA_RD',
    >>>  'VA:LS1_CA01:CAV2_D1136:PHA_RD',
    >>>  'VA:LS1_CA01:CAV3_D1142:PHA_RD',
    >>>  'VA:SVR:NOISE'
    >>> ]
    >>> # instantiation
    >>> data_fetcher = DataFetcher(pvs, timeout=10)
    >>> # fetch the data, see fetch_data() for the parameters definition
    >>> avg, df = data_fetcher(time_span=2.0, with_data=True, verbose=True)
    >>> # another fetch for just mean values.
    >>> avg, _ = data_fetcher(1.0)
    >>> # return raw fetch data, save post-processing
    >>> avg, df_raw = data_fetcher(1.0, with_data=True, expanded=False)
    >>> # clean up (optional)
    >>> data_fetcher.clean_up()
    >>> # Re-instantiation is required after clean_up if working with the DataFetcher with
    >>> # the same variable name, e.g. data_fetcher = DataFetcher(pvs), ...
    >>> # If working with a large list of PVs for multiple DataFetcher instances,
    >>> # cleaning up the not-needed DataFetcher instances is useful to save computing resources.
    """

    def __init__(self, pvlist: List[str], **kws):
        self.__check_unique_list(pvlist)
        self._pvlist = pvlist
        self._npv = len(pvlist)
        self._pvs = [None] * self._npv  # weakrefs
        self._cb_idx = [None] * self._npv  # cb indices
        #
        self.timeout = kws.get('timeout', 5)
        self.verbose = kws.get('verbose', False)
        # start data accumulating if set.
        self._run = False
        #
        self.pre_setup()

    @staticmethod
    def pack_data(df: pd.DataFrame, abs_z: float = None, with_data: bool = False,
                  expanded: bool = True):
        """Pack the original retrieved dataframe with three more columns of data, row-wised, if
        *with_data* if True.

        - '#': The total count of valid fetched data points.
        - 'mean': The averaged value from all valid data points.
        - 'std': The standard deviation value from all valid data points.

        Only use the data of interest if *abs_z* is set.

        Parameters
        ----------
        df : pd.DataFrame
            The original retrieved dataset with varied length of valid data points for each row.
            The PV name list as the row index.
        abs_z : float
            The absolute value of z-score, drop the data beyond, if not set, keep all the data.
        with_data : bool
            If set, return data array as the second element of the returned tuple.
        expanded : bool
        If set along with *with_data*, return an expanded dataset, defaults to True.

        Returns
        -------
        r : Tuple
            A tuple of average array and processed dataframe, with more columns and/or
            data-of-interested defined with *abs_z*.
        """
        def _pack_df(_df: pd.DataFrame):
            if with_data:
                if expanded:
                    n_col = _df.shape[1]
                    col_mean = _df.mean(axis=1)
                    col_std = _df.std(ddof=0, axis=1)
                    _df['#'] = _df.apply(lambda i: n_col - i.isna().sum(), axis=1)
                    _df['mean'] = col_mean
                    _df['std'] = col_std
                return _df
            else:
                return None
        # mean, std
        _avg, _std = df.mean(axis=1), df.std(ddof=0, axis=1)
        if abs_z is None:
            return _avg.to_numpy(), _pack_df(df)
        else:
            # - mean
            df_sub = df.sub(_avg, axis=0)
            # idx1: df_sub == 0
            idx1 = df_sub == 0.0
            # ((- mean) / std) ** 2
            df1 = df_sub.div(_std, axis=0)**2
            idx2 = df1 <= abs_z**2
            # data of interest
            df_final = df[idx1 | idx2]
            # mean array
            avg_arr = df_final.mean(axis=1).to_numpy()
            return avg_arr, _pack_df(df_final)

    def __check_unique_list(self, pvlist: List[str]):
        if len(set(pvlist)) != len(pvlist):
            raise RuntimeError("Duplicated PV names!")

    def __check_all_pvs(self):
        # return a boolean if all PVs are ready to work or not.
        return all(self._cb_idx) and \
                all((o() is not None and o().connected for o in self._pvs))

    @property
    def timeout(self):
        """float: Maximum allowed waiting time in seconds before all PVs are ready to work.
        """
        return self._timeout

    @timeout.setter
    def timeout(self, t: float):
        self._timeout = t

    def is_pvs_ready(self):
        """Return if all PVs are ready to work or not.
        """
        return self._all_pvs_ready

    def __del__(self):
        """Cleaning up work.
        """
        self.clean_up()

    def clean_up(self):
        [o().remove_callback(idx) for o, idx in zip(self._pvs, self._cb_idx)]

    def pre_setup(self):
        """Preparation for the data fetch procedure.
        """
        # clear the data container
        self._data_list = [[] for _ in range(self._npv)]
        # if all PVs are ready, just return
        self._all_pvs_ready = self.__check_all_pvs()
        if self._all_pvs_ready:
            return
        #
        t0 = time.perf_counter()

        def _cb(idx: int, **kws):
            if self._run:
                val = kws.get('value')
                self._data_list[idx].append(val)
                if self.verbose:
                    ts = kws.get('timestamp')
                    click.secho(
                        f"[{epoch2human(ts)[:-3]}] Get {kws.get('pvname')}: {val:<6g}",
                        fg="blue")

        #
        q = Queue()
        conn_sts = [False] * self._npv

        def _f(i: int, pvname: str, conn: bool, **kws):
            if conn:
                conn_sts[i] = True
                if all(conn_sts):
                    q.put(True)

        for i, pvname in enumerate(self._pvlist):
            o = get_pv(pvname,
                       connection_callback=partial(_f, i),
                       auto_monitor=True)
            if self._cb_idx[i] is None:
                self._cb_idx[i] = o.add_callback(partial(_cb, i),
                                                 with_ctrlvars=False)
            self._pvs[i] = weakref.ref(o)

        while True:
            try:
                v = q.get(timeout=self._timeout)
                if v: raise PVsAreReady
            except Empty:
                print(f"Failed connecting to all PVs in {self._timeout:.1f}s.")
                if self.verbose:
                    not_conn_pvs = [
                        o().pvname for o in self._pvs if not o().connected
                    ]
                    click.secho(
                        f"{len(not_conn_pvs)} PVs are not established in {self._timeout:.1f}s.",
                        fg="red")
                    click.secho("{}".format('\n'.join(not_conn_pvs)), fg="red")
                break
            except PVsAreReady:
                if self.verbose:
                    click.secho(
                        f"Established {self._npv} PVs in {(time.perf_counter() - t0) * 1e3:.1f}ms.",
                        fg="green")
                self._all_pvs_ready = True
                break

    def __call__(self,
                 time_span: float = 5.0,
                 abs_z: float = None,
                 with_data: bool = False,
                 **kws):
        verbose = kws.get('verbose', self.verbose)
        self.verbose = verbose
        # initial data list
        self._data_first_shot = [o().value for o in self._pvs]
        self._data_list = [[] for i in range(self._npv)]
        _tq = Queue()
        _evt = Event()

        def _tick_down(q):
            self._run = True
            while True:
                if _evt.is_set():
                    self._run = False
                    break
                q.put(time.time())
                time.sleep(0.001)

        th = Thread(target=_tick_down, args=(_tq, ))
        th.start()
        t0 = time.time()
        t1 = t0 + time_span
        while True:
            try:
                t = _tq.get(timeout=5)
                if t >= t1: raise FetchDataFinishedException
            except FetchDataFinishedException:
                _evt.set()
                if verbose:
                    click.secho(f"Finished fetching data in {t - t0:.1f}s")
                break
        # amend the first element of the list container with initial shot
        for i in range(self._npv):
            if not self._data_list[i]:
                self._data_list[i] = [self._data_first_shot[i]]
        # raw data
        df0 = pd.DataFrame(self._data_list, index=self._pvlist)
        return DataFetcher.pack_data(df0, abs_z, with_data, expanded=kws.get('expanded', True))
        
        
        