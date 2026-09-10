"""
live_cs_inference.py
====================
One-shot, prior-anchored Courant-Snyder inference for live machine drift monitoring.

Overview
--------
A single passive BPMQ reading at the current operating point is sufficient to
produce an updated CS estimate **when anchored to a reference prior**.  The prior
regularisation keeps the solution physically close to the known reference state
so that small deviations (machine drift) are not swamped by reconstruction
noise.

Key class
---------
LiveCSInference
    Wraps a BPMQscan internally and exposes a simple ``infer(cs_prior)`` call.
    On each call:

    1.  Passively read BPMQ at the *current* quad setting (no knobs touched).
    2.  Run cs_reconstruct with a prior-anchoring regularisation:
    3.  Return an ``InferenceResult`` containing the inferred CS parameters,
        diagnostics (prior mismatch, MMD, fit residuals), and the raw data.

Usage example
-------------
::

    from live_cs_inference import LiveCSInference

    # --- build once (expensive) ---
    infer_app = LiveCSInference(
        lattice_dicts   = lattice_dicts,      # list of element dicts (same as BPMQscan)
        E_MeV_u         = 130.0,
        mass_number     = 124,
        charge_number   = 49,
        quads_to_scan   = ['BDS_BTS:PSQ_D5501', 'BDS_BTS:PSQ_D5509'],
        quads_max_curr  = [150, 150],
        quads_min_curr  = [5,   5  ],
        BPM_names       = ['BDS_BTS:BPM_D5513', 'BDS_BTS:BPM_D5565'],
        machineIO       = my_machineIO,       # None → virtual machine
        BPMQ_model_type = 'TIS161_GP',
    )

    # known-good reference state (e.g. after a full multi-shot scan)
    cs_ref = [0.0, 5.0, 0.16e-6, 0.0, 5.0, 0.16e-6]

    # --- call repeatedly (fast) ---
    result = infer_app.infer(cs_prior=cs_ref, prior_weight=2.0)
    print("Inferred CS:", result.cs)
    print("MMD to prior:", result.mmd_to_prior)
    print("Mismatch x/y:", result.mismatch_x, result.mismatch_y)

    # save full result for logging
    result.save("drift_log_20250101_120000.pkl")
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from pathlib import Path

_script_path = Path(__file__).resolve()
_BPMQpkg_dir = _script_path.parent.parent



# ── Local imports (same package as BAL4BPMQ) ──────────────────────────────────
from .BAL4BPMQ import (
    BPMQscan,
    cs2noise,
    _dtype,
)
from .utils import calculate_mismatch_factor, calculate_MMD4D
from .fmlat import fmname2mpname, combine_lattice_elements_quads_only_w_live_update

# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    cs             : np.ndarray
    cs_ensemble    : np.ndarray
    cs_prior       : Optional[np.ndarray]
    mmd_to_prior   : Optional[float]
    mismatch_x     : Optional[float]
    mismatch_y     : Optional[float]
    fit_residuals  : Optional[np.ndarray]
    lB2_measured   : np.ndarray
    lBPMQ_measured : np.ndarray
    timestamp      : datetime
    prior_weight   : float
    elapsed_seconds: float

    @property
    def drifted(self) -> bool:
        return self.mmd_to_prior is not None and self.mmd_to_prior > 0.05

    def summary(self) -> str:
        cs = self.cs
        prior_line = (
            f"MMD-to-prior={self.mmd_to_prior:.4f}  "
            f"mismatch x/y={self.mismatch_x:.3f}/{self.mismatch_y:.3f}  "
            f"({'DRIFT DETECTED' if self.drifted else 'OK'})"
            if self.mmd_to_prior is not None
            else "no prior supplied"
        )
        return (
            f"[{self.timestamp:%Y-%m-%d %H:%M:%S}] "
            f"CS  x: α={cs[0]:.3f} β={cs[1]:.3f} m  ε={cs[2]*1e6:.4f} μm·rad  |  "
            f"y: α={cs[3]:.3f} β={cs[4]:.3f} m  ε={cs[5]*1e6:.4f} μm·rad\n"
            f"    {prior_line}  "
            f"({self.elapsed_seconds:.1f} s)"
        )

    def to_dict(self, json_safe: bool = True) -> Dict[str, Any]:
        """
        Convert result to a dictionary.

        If json_safe=True, NumPy arrays become lists and datetime becomes ISO string.
        This form can be passed directly to json.dump().
        """
        def arr(x):
            if x is None:
                return None
            return x.tolist() if json_safe else x

        return {
            "schema_version": 1,

            "cs": arr(self.cs),
            "cs_ensemble": arr(self.cs_ensemble),
            "cs_prior": arr(self.cs_prior),

            "mmd_to_prior": None if self.mmd_to_prior is None else float(self.mmd_to_prior),
            "mismatch_x": None if self.mismatch_x is None else float(self.mismatch_x),
            "mismatch_y": None if self.mismatch_y is None else float(self.mismatch_y),

            "fit_residuals": arr(self.fit_residuals),
            "lB2_measured": arr(self.lB2_measured),
            "lBPMQ_measured": arr(self.lBPMQ_measured),

            "timestamp": self.timestamp.isoformat() if json_safe else self.timestamp,
            "prior_weight": float(self.prior_weight),
            "elapsed_seconds": float(self.elapsed_seconds),

            # Derived convenience field.
            "drifted": bool(self.drifted),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        """
        Reconstruct an InferenceResult from a JSON-safe dictionary.
        Ignores derived fields like 'drifted'.
        """
        def arr(x):
            if x is None:
                return None
            return np.asarray(x, dtype=np.float64)

        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            cs=arr(data["cs"]),
            cs_ensemble=arr(data["cs_ensemble"]),
            cs_prior=arr(data.get("cs_prior")),

            mmd_to_prior=data.get("mmd_to_prior"),
            mismatch_x=data.get("mismatch_x"),
            mismatch_y=data.get("mismatch_y"),

            fit_residuals=arr(data.get("fit_residuals")),
            lB2_measured=arr(data["lB2_measured"]),
            lBPMQ_measured=arr(data["lBPMQ_measured"]),

            timestamp=timestamp,
            prior_weight=float(data["prior_weight"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        """Save a portable JSON representation."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(json_safe=True), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "InferenceResult":
        """Load from a portable JSON representation."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        """
        Pickle the result to path.

        Good for quick local Python-only workflows.
        Prefer save_json() for portable logs.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "InferenceResult":
        with open(path, "rb") as f:
            return pickle.load(f)


_INFERENCE_RESULT_DF_COLUMNS = [
    "aflx",
    "betx (m)",
    "nemitx (m.rad)",
    "alfy",
    "bety (m)",
    "nemity (m.rad)",
    "mmd_to_prior",
    "mismatch_x",
    "mismatch_y",
]


def inference_results_to_dataframe(
    results: Sequence[InferenceResult],
    *,
    sort_by_timestamp: bool = False,
) -> pd.DataFrame:
    """
    Convert a list of :class:`InferenceResult` objects to a tidy DataFrame.

    This is intended for post-processing the list returned by
    :meth:`LiveCSInference.run_monitor_loop`.  The output index is the result
    timestamp and the columns are the CS values plus prior-drift diagnostics.

    Parameters
    ----------
    results : sequence of InferenceResult
        Results returned by ``run_monitor_loop()`` or collected from repeated
        ``infer()`` calls.
    sort_by_timestamp : bool, default False
        Preserve input order by default.  Set True to sort the DataFrame by
        timestamp after construction.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``timestamp`` with columns:
        ``['aflx', 'betx (m)', 'nemitx (m.rad)', 'alfy', 'bety (m)',
        'nemity (m.rad)', 'mmd_to_prior', 'mismatch_x', 'mismatch_y']``.

    Examples
    --------
    >>> results = infer_app.run_monitor_loop(cs_prior=cs_ref, n_steps=10)
    >>> df = inference_results_to_dataframe(results)
    """

    def _float_or_nan(value: Optional[float]) -> float:
        return np.nan if value is None else float(value)

    rows = []
    timestamps = []

    for idx, result in enumerate(results):
        if not isinstance(result, InferenceResult):
            raise TypeError(
                f"results[{idx}] must be an InferenceResult, "
                f"got {type(result).__name__}"
            )

        cs = np.asarray(result.cs, dtype=np.float64).reshape(-1)
        if cs.shape != (6,):
            raise ValueError(
                f"results[{idx}].cs must contain 6 values, "
                f"got shape {np.asarray(result.cs).shape}"
            )

        rows.append({
            "aflx": float(cs[0]),
            "betx (m)": float(cs[1]),
            "nemitx (m.rad)": float(cs[2]),
            "alfy": float(cs[3]),
            "bety (m)": float(cs[4]),
            "nemity (m.rad)": float(cs[5]),
            "mmd_to_prior": _float_or_nan(result.mmd_to_prior),
            "mismatch_x": _float_or_nan(result.mismatch_x),
            "mismatch_y": _float_or_nan(result.mismatch_y),
        })
        timestamps.append(pd.to_datetime(result.timestamp))

    df = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
        columns=_INFERENCE_RESULT_DF_COLUMNS,
    )

    if sort_by_timestamp:
        df = df.sort_index()

    return df


# Backwards-compatible short alias for notebook use.
inference_results_to_df = inference_results_to_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# Main application class
# ──────────────────────────────────────────────────────────────────────────────

class LiveCSInference:
    """
    One-shot, prior-anchored Courant-Snyder inference for live drift monitoring.

    Builds a BPMQscan internally (expensive; done once at construction time),
    then exposes a lightweight ``infer(cs_prior, prior_weight)`` method that:

    * performs a **passive** single BPMQ reading (no quad changes),
    * runs ``cs_reconstruct`` with a Gaussian prior anchored to *cs_prior*,
    * returns an :class:`InferenceResult` with the updated CS and diagnostics.

    Parameters
    ----------
    lattice_dicts : list[dict]
        Element dictionaries for :class:`~BAL4BPMQ.LatticeMap`
        (same as the BPMQscan ``lattice_dicts`` parameter).
    E_MeV_u : float
        Kinetic energy per nucleon [MeV/u].
    mass_number : int
    charge_number : int
    machineIO : optional
        If None, a virtual (simulation) machine is used.
    BPMQ_model_type : str
        BPMQ model string passed to BPMQscan (e.g. ``'TIS161_GP'``).
    batch_size : int
        Ensemble size for the optimiser.
    n_batch_padding_factor : int
        Padding multiplier for optimiser (total starts = batch_size * factor).
    num_restarts : int
        Number of optimiser restarts.
    quads_tol_curr : list[float] | None
        Quad current tolerances [A].  Defaults to 0.3 A per quad.
    dtype : torch.dtype
        Float precision.
    seed : int
    **extra_bpmqscan_kwargs
        Any additional kwargs forwarded verbatim to BPMQscan.
    """

    def __init__(
        self,
        E_MeV_u         : float,
        mass_number     : int,
        charge_number   : int,
        machineIO       : Optional[Any] = None,
        flame_filename  : Optional[str] = _BPMQpkg_dir / "test_LS3_Target.lat",
        from_element    : Optional[str] = None,
        to_element      : Optional[str] = "BDS_BTS:PM_D5567",
        BPMQ_model_type : str = 'TIS161_GP',
        batch_size      : int = 8,
        n_batch_padding_factor: int = 16,
        num_restarts    : int = 3,
        fit_err         : bool = False,
        plot_history    : bool = False,
        plot_ellipse    : bool = False,
        seed            : int = 42,
        **extra_bpmqscan_kwargs,
    ):
        lattice_dicts = combine_lattice_elements_quads_only_w_live_update(flame_filename, from_element, to_element)
        for elem_dic in lattice_dicts:
            elem_dic['name'] = fmname2mpname(elem_dic['name'])
        quads_to_scan  = [elem['name'] for elem in lattice_dicts if elem['type']=='quadrupole']
        quads_to_scan  = quads_to_scan[:2]
        quads_max_curr = [5] * len(quads_to_scan)
        quads_min_curr = [150] * len(quads_to_scan)
        quads_tol_curr = [0.3] * len(quads_to_scan)
        BPM_names      = [elem['name'] for elem in lattice_dicts if 'BPM' in elem['name']][1:]
        print("BPM_names",BPM_names)

        self._lattice_dicts    = lattice_dicts
        self._E_MeV_u          = E_MeV_u
        self._mass_number      = mass_number
        self._charge_number    = charge_number
        self._quads_to_scan    = quads_to_scan
        self._quads_max_curr   = quads_max_curr
        self._quads_min_curr   = quads_min_curr
        self._BPM_names        = BPM_names
        self._machineIO        = machineIO
        self._BPMQ_model_type  = BPMQ_model_type if machineIO is not None else 'TIS161'
        self._batch_size       = batch_size
        self._n_batch_padding_factor = n_batch_padding_factor
        self._num_restarts     = num_restarts
        self._fit_err          = fit_err
        self._plot_history     = plot_history       
        self._plot_ellipse     = plot_ellipse   
        self._quads_tol_curr   = quads_tol_curr
        self._dtype            = _dtype
        self._seed             = seed
        self._extra_kwargs     = extra_bpmqscan_kwargs
        self._bpmqscan: Optional[BPMQscan] = None
        self._build_bpmqscan()

    # ── Construction helpers ───────────────────────────────────────────────────

    def _build_bpmqscan(self):
        """Instantiate a fresh BPMQscan (resets all training data)."""
        self._bpmqscan = BPMQscan(
            E_MeV_u               = self._E_MeV_u,
            mass_number           = self._mass_number,
            charge_number         = self._charge_number,
            lattice_dicts         = self._lattice_dicts,
            quads_to_scan         = self._quads_to_scan,
            quads_max_curr        = self._quads_max_curr,
            quads_min_curr        = self._quads_min_curr,
            quads_tol_curr        = self._quads_tol_curr,
            BPM_names             = self._BPM_names,
            machineIO             = self._machineIO,
            BPMQ_model_type       = self._BPMQ_model_type,
            batch_size            = self._batch_size,
            n_batch_padding_factor= self._n_batch_padding_factor,
            num_restarts          = self._num_restarts,
            bootstrap             = False,   # single-shot: no bootstrapping
            sample_model_err      = False,
            fit_err               = self._fit_err,
            plot_history          = self._plot_history,
            plot_ellipse          = self._plot_ellipse,
            n_init                = 1,        # single-shot: one passive measurement
            dtype                 = self._dtype,
            seed                  = self._seed,
            **self._extra_kwargs,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self):
        """
        Rebuild the internal BPMQscan from scratch.

        Call this between inference sessions (e.g. after a tune change) to
        discard stale training data and reset the optimiser state.
        """
        self._build_bpmqscan()

    def infer(
        self,
        cs_prior        : Optional[Sequence[float]],
        prior_weight    : float = 2.0,
        verbose         : bool = True,
    ) -> InferenceResult:
        """
        One-shot CS inference anchored to *cs_prior*.

        Steps
        -----
        1.  **Fresh BPMQscan** is rebuilt to clear any stale training data from
            a previous call.
        2.  ``bpmqscan.initialize(n_init=1, cs_prior=cs_prior,
            prior_weight=prior_weight)`` is called, which:

            a.  Reads BPMQ passively at the current operating point
                (no quads are touched).
            b.  Calls ``train_model(cs_prior=cs_prior, prior_weight=prior_weight)``
                → ``model.cs_reconstruct(…, cs_prior=cs_prior,
                prior_weight=prior_weight)`` — sets ``model.cs_ref = cs_prior``
                so that noise=0 ≡ cs_prior, then minimises::

                    fitloss_bpmQ + prior_weight * mean_j(noise_j^2)

        3.  A final deterministic clean fit (no model-error sampling or
            bootstrapping) overwrites the result.

        Parameters
        ----------
        cs_prior : sequence of 6 floats or None
            Reference Courant-Snyder parameters used as the regularisation
            anchor: [alpha_x, beta_x, emitt_x, alpha_y, beta_y, emitt_y].
            Must have beta_x, emitt_x, beta_y, emitt_y > 0.
            Pass None to run without a prior (pure data fit; may be ill-posed
            for a single scan).

        prior_weight : float
            Regularisation coefficient.  Dimensionally:

            * ``prior_weight = 0``  → pure data fit (ill-posed for 1 scan)
            * ``prior_weight = 1``  → prior ≈ measurement (balanced)
            * ``prior_weight = 2-5``→ tight anchoring (recommended for drift)
            * ``prior_weight ≫ 10`` → essentially ignores the measurement

            See module docstring for scaling rationale.  Ignored when
            *cs_prior* is None.

        verbose : bool
            If True, print a one-line summary on completion.

        Returns
        -------
        InferenceResult
        """
        t0 = datetime.now()

        # Validate and normalise cs_prior -----------------------------------------
        # Keep cs_prior_arr as np.ndarray (or None) as the canonical form used
        # throughout; cs_prior_list is the plain-list form forwarded to BPMQscan.
        cs_prior_arr: Optional[np.ndarray] = None
        cs_prior_list: Optional[list] = None
        if cs_prior is not None:
            cs_prior_arr = np.asarray(cs_prior, dtype=np.float64)
            if cs_prior_arr.shape != (6,):
                raise ValueError(f"cs_prior must have 6 elements, got {cs_prior_arr.shape}")
            if cs_prior_arr[1] <= 0 or cs_prior_arr[2] <= 0 \
                    or cs_prior_arr[4] <= 0 or cs_prior_arr[5] <= 0:
                raise ValueError("cs_prior beta_x, emitt_x, beta_y, emitt_y must be positive.")
            cs_prior_list = cs_prior_arr.tolist()  # plain list for BPMQscan

        # ── 1. Reset state ──────────────────────────────────────────────────
        self.reset()
        b = self._bpmqscan

        # ── 2. Single-shot initialisation + reconstruction with prior ───────
        b.initialize(
            n_init       = 1,
            cs_prior     = cs_prior_list,
            prior_weight = prior_weight,
        )

        # ── 3. Final clean fit (deterministic, no noise sampling) ────────────
        # b.train_model(
        #     sample_model_err = False,
        #     bootstrap        = False,
        #     fit_err          = False,
        #     _record_state    = True,
        #     cs_prior         = cs_prior_list,
        #     prior_weight     = prior_weight,
        # )

        # ── 4. Extract results ───────────────────────────────────────────────
        result = self._extract_result(b, cs_prior_arr, prior_weight, t0)

        if verbose:
            print(result.summary())

        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_result(
        self,
        b            : BPMQscan,
        cs_prior_arr : Optional[np.ndarray],
        prior_weight : float,
        t0           : datetime,
    ) -> InferenceResult:
        """Convert the BPMQscan state into an InferenceResult."""

        model = b.model

        # Best-estimate CS (first member of ensemble)
        cs_np = model.cs.detach().cpu().numpy()      # shape (6,)

        # Full ensemble  (batch_size, 6)
        cs_ensemble = model.noise2cs(
            model.best_noise_ensemble[:, :6]
        ).detach().cpu().numpy()

        # Fit residuals (1, n_bpm)
        fit_residuals = b._compute_fit_residuals()

        # Measurement data captured during initialize()
        lB2_measured = (
            b.train_llB2[0].detach().cpu().numpy()
            if b.train_llB2 is not None else np.array([])
        )
        lBPMQ_measured = (
            b.train_llBPMQ[0].detach().cpu().numpy()
            if b.train_llBPMQ is not None else np.array([])
        )

        # Distance metrics vs prior (all None when no prior was supplied)
        if cs_prior_arr is not None:
            cs_prior     = cs_prior_arr.copy()
            mmd_to_prior = float(calculate_MMD4D(cs_prior_arr, cs_np))
            mismatch_x   = float(calculate_mismatch_factor(cs_prior_arr[:3], cs_np[:3]))
            mismatch_y   = float(calculate_mismatch_factor(cs_prior_arr[3:], cs_np[3:]))
        else:
            cs_prior     = None
            mmd_to_prior = None  # fixed: removed erroneous trailing commas
            mismatch_x   = None
            mismatch_y   = None

        elapsed = (datetime.now() - t0).total_seconds()

        return InferenceResult(
            cs             = cs_np,
            cs_ensemble    = cs_ensemble,
            cs_prior       = cs_prior,
            mmd_to_prior   = mmd_to_prior,
            mismatch_x     = mismatch_x,
            mismatch_y     = mismatch_y,
            fit_residuals  = fit_residuals,
            lB2_measured   = lB2_measured,
            lBPMQ_measured = lBPMQ_measured,
            timestamp      = datetime.now(),
            prior_weight   = prior_weight,
            elapsed_seconds= elapsed,
        )

    # ── Convenience: run a series and log results ─────────────────────────────

    def run_monitor_loop(
        self,
        cs_prior        : Optional[Sequence[float]],
        n_steps         : int,
        prior_weight    : float = 1.0,
        interval_seconds: float = 0.0,
        log_dir         : Optional[Union[str, Path]] = None,
        verbose         : bool = True,
    ) -> List[InferenceResult]:
        """
        Call ``infer()`` *n_steps* times and collect results.

        Useful for offline batch processing or simple live monitoring loops
        without a GUI.  For production use, drive infer() from your own
        event loop.

        Parameters
        ----------
        cs_prior : sequence of 6 floats or None
            Reference CS prior (applied for every step).
        n_steps : int
            Number of inference calls.
        prior_weight : float
        interval_seconds : float
            Sleep between calls (0 = as fast as possible).
        log_dir : str | Path | None
            If given, save each InferenceResult as a timestamped .pkl file.
        verbose : bool

        Returns
        -------
        list[InferenceResult]
        """
        import time

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for step in range(n_steps):
            if verbose:
                print(f"\n── Step {step + 1}/{n_steps} ──")
            result = self.infer(cs_prior, prior_weight=prior_weight, verbose=verbose)
            results.append(result)

            if log_dir is not None:
                fname = log_dir / f"cs_infer_{result.timestamp:%Y%m%d_%H%M%S}_{step:04d}.pkl"
                result.save(fname)

            if interval_seconds > 0 and step < n_steps - 1:
                time.sleep(interval_seconds)

        return results