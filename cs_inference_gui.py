#!/usr/bin/env python3
"""
CS Inference GUI Application — Enhanced
Live monitoring and control for Courant-Snyder parameter inference using BPMQscan.

New features vs. original:
  • Beam Parameters panel (E, A, Q) with EPICS auto-read and manual override
  • Derived physics display (Bρ, β, γ) updated live
  • Rich Inference Status bar (step, training pts, rejected count + details, discriminability)
  • Lattice Context sub-panel (quads + ranges, BPM names)
  • Stop → Data Quality Review dialog
      - Per-scan checkboxes, BPMQ stats, per-BPM popup
      - Retrain with selected scans
      - Resume AL / Save / Start Fresh
  • Selection history tracking on InferenceWorker

Usage:
    python cs_inference_gui.py [config.yaml]
"""

import sys
import os
import yaml
import time
import pickle
import datetime
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QMessageBox,
    QDialog, QDialogButtonBox, QGroupBox, QSplitter,
    QHeaderView, QFrame, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from BPMQ import fmlat, BPMQscan, plot_convergence
    from BPMQ.construct_machineIO import construct_machineIO
    from BPMQ.utils import calculate_mismatch_factor, calculate_MMD4D
except ImportError as e:
    print(f"ERROR: Cannot import BPMQ modules: {e}")
    print("Make sure BPMQ package is in PYTHONPATH.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

_M_U_MEV = 931.494  # atomic mass unit in MeV/c²

def compute_beam_derived(E_MeV_u: float, mass_number: int, charge_number: int):
    """
    Compute relativistic beam parameters from kinetic energy per nucleon,
    mass number A, and charge state Q.

    Returns
    -------
    Brho : float  — magnetic rigidity [T·m]
    beta : float  — v/c
    gamma : float — Lorentz factor
    """
    m_rest  = mass_number  * _M_U_MEV          # total rest energy [MeV]
    E_total = m_rest + E_MeV_u * mass_number    # total energy [MeV]
    p_c     = float(np.sqrt(max(E_total**2 - m_rest**2, 0.0)))  # [MeV]
    beta    = p_c / E_total   if E_total > 0       else 0.0
    gamma   = E_total / m_rest if m_rest  > 0       else 1.0
    # Bρ = p / (Q·e) = (p·c [MeV]) / (Q · 299.792458 [MeV/T/m])
    Brho    = p_c / (charge_number * 299.792458) if charge_number > 0 else 0.0
    return Brho, beta, gamma


def _fmt(val, fmt=".4f"):
    try:
        return format(float(val), fmt)
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# PM Scan Dialog  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class PMScanDialog(QDialog):
    """Dialog for manual PM scan input."""
    def __init__(self, quad_setting, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PM Scan Required")
        self.setModal(True)
        layout = QVBoxLayout()
        info = QLabel(
            f"<b>Perform PM scan at quad setting:</b><br>{quad_setting}<br><br>"
            "After completing the scan, enter measured beam sizes:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QGridLayout()
        form.addWidget(QLabel("x_rms (mm):"), 0, 0)
        self.xrms_input = QDoubleSpinBox()
        self.xrms_input.setDecimals(3); self.xrms_input.setRange(0.001, 100.0); self.xrms_input.setValue(1.0)
        form.addWidget(self.xrms_input, 0, 1)
        form.addWidget(QLabel("y_rms (mm):"), 1, 0)
        self.yrms_input = QDoubleSpinBox()
        self.yrms_input.setDecimals(3); self.yrms_input.setRange(0.001, 100.0); self.yrms_input.setValue(1.0)
        form.addWidget(self.yrms_input, 1, 1)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        return self.xrms_input.value(), self.yrms_input.value()


# ─────────────────────────────────────────────────────────────────────────────
# Retrain Thread  (runs bpmQscan.train_model in a QThread while worker is paused)
# ─────────────────────────────────────────────────────────────────────────────

class RetrainThread(QThread):
    """
    Calls bpmQscan.train_model with a boolean mask selecting a subset of scans.
    Safe to run from the main thread because InferenceWorker is paused (blocked
    in threading.Event.wait()) while this executes.
    """
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, bpmQscan_obj, mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.bpmQscan_obj = bpmQscan_obj
        self.mask = mask.astype(bool)

    def run(self):
        try:
            b = self.bpmQscan_obj
            m = self.mask
            import torch
            fB2   = b.train_llB2[m]     if b.train_llB2   is not None else None
            fBPMQ = b.train_llBPMQ[m]   if b.train_llBPMQ is not None else None
            ftol  = b.train_llBPMQtol[m]     if b.train_llBPMQtol     is not None else None
            fmerr = b.train_llBPMQmodelerr[m] if b.train_llBPMQmodelerr is not None else None
            b.train_model(
                train_llB2=fB2, train_llBPMQ=fBPMQ,
                train_llBPMQtol=ftol, train_llBPMQmodelerr=fmerr,
                _record_state=True
            )
            self.done.emit(b.get_data())
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Rejected Data Dialog
# ─────────────────────────────────────────────────────────────────────────────

class RejectedDataDialog(QDialog):
    """Popup showing all rejected scan candidates and their rejection reasons."""

    def __init__(self, rejection_log: list, quads_to_scan: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rejected Scan Details")
        self.setMinimumSize(750, 380)
        layout = QVBoxLayout(self)

        n_rej = len(rejection_log)
        hdr = QLabel(
            f"<b>{n_rej} scan(s) rejected</b> — these quad settings were <i>not</i> "
            "added to the training set."
        )
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        if n_rej == 0:
            layout.addWidget(QLabel("No rejections recorded yet."))
        else:
            quad_short = [q.split(':')[-1] for q in quads_to_scan]
            n_q = len(quad_short)
            headers = ["ID"] + [f"B2 {s} (T/m²)" for s in quad_short] + ["Reason"]
            table = QTableWidget(n_rej, len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setAlternatingRowColors(True)

            for row, entry in enumerate(rejection_log):
                lB2    = entry.get('lB2', [])
                reason = entry.get('reason', 'Beam loss / |BPMQ| > 40 mm²')
                table.setItem(row, 0, QTableWidgetItem(f"Rej_{row+1}"))
                for qi, b2_val in enumerate(lB2[:n_q]):
                    table.setItem(row, 1 + qi, QTableWidgetItem(f"{float(b2_val):.3f}"))
                ri = QTableWidgetItem(f"🔴 {reason}")
                ri.setForeground(QColor('#cc2200'))
                table.setItem(row, 1 + n_q, ri)

            layout.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality Review Dialog  (opened when user clicks Stop)
# ─────────────────────────────────────────────────────────────────────────────

class DataQualityDialog(QDialog):
    """
    Shown when the user clicks Stop.  Lets the user:
      • Review accepted scans (with per-row BPMQ popup)
      • Uncheck suspicious scans and retrain the model
      • Resume AL from where it paused
      • Save the current result
      • Start Fresh (abort and reset)
    """
    retrain_plot_update = pyqtSignal(dict)   # carries updated data after retrain

    def __init__(self, data: dict, worker: 'InferenceWorker', parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Quality Review — Inference Paused")
        self.setMinimumSize(960, 580)
        self.data    = data
        self.worker  = worker
        self._retrain_thread = None
        self._last_selection_mask = None

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ─ Header ─
        train_llB2 = self.data.get('train_llB2')
        n_acc = len(train_llB2) if train_llB2 is not None else 0
        n_rej = len(self.worker.rejection_log)

        hdr = QLabel(
            f"<h3>📊 Data Quality Review</h3>"
            f"Inference is <b>paused</b> — {n_acc} accepted scan(s), "
            f"{n_rej} rejected. "
            "Uncheck any suspicious scans, then retrain or resume."
        )
        hdr.setWordWrap(True)
        root.addWidget(hdr)

        # ─ Accepted scans table ─
        grp = QGroupBox("Accepted Scans  (uncheck rows to exclude from retrain)")
        gv  = QVBoxLayout(grp)
        self.scan_table = self._build_scan_table()
        gv.addWidget(self.scan_table)
        root.addWidget(grp, stretch=1)

        # ─ Selection controls ─
        sel_row = QHBoxLayout()
        btn_all = QPushButton("✓ All"); btn_all.setMaximumWidth(70)
        btn_all.clicked.connect(self._check_all)
        btn_none = QPushButton("✗ None"); btn_none.setMaximumWidth(70)
        btn_none.clicked.connect(self._uncheck_all)
        self.sel_lbl = QLabel(f"{n_acc}/{n_acc} scans selected")
        sel_row.addWidget(btn_all); sel_row.addWidget(btn_none)
        sel_row.addStretch(); sel_row.addWidget(self.sel_lbl)
        root.addLayout(sel_row)

        # ─ Retrain status ─
        self.retrain_status = QLabel("")
        self.retrain_status.setStyleSheet("color: #0055cc; font-style: italic;")
        root.addWidget(self.retrain_status)

        # ─ Action buttons ─
        btn_row = QHBoxLayout()

        self.retrain_btn = QPushButton("🔄  Retrain with Selected")
        self.retrain_btn.setStyleSheet(
            "font-weight:bold; background:#4a90d9; color:white; padding:6px 14px;"
        )
        self.retrain_btn.clicked.connect(self._do_retrain)

        # Disable Retrain until we know worker is truly paused at _check_pause()
        if self.worker.is_truly_paused:
            self.retrain_btn.setEnabled(True)
            self._pause_wait_lbl = None
        else:
            self.retrain_btn.setEnabled(False)
            self._pause_wait_lbl = QLabel("⏳ Waiting for current step to finish…")
            self._pause_wait_lbl.setStyleSheet("color: orange;")
            self.worker.worker_paused.connect(self._on_worker_truly_paused)

        btn_row.addWidget(self.retrain_btn)
        if self._pause_wait_lbl:
            btn_row.addWidget(self._pause_wait_lbl)

        btn_row.addStretch()

        resume_btn = QPushButton("▶  Resume AL")
        resume_btn.setStyleSheet(
            "font-weight:bold; background:#5cb85c; color:white; padding:6px 14px;"
        )
        resume_btn.clicked.connect(self._do_resume)
        btn_row.addWidget(resume_btn)

        save_btn = QPushButton("💾  Save Current Result")
        save_btn.clicked.connect(self._do_save)
        btn_row.addWidget(save_btn)

        fresh_btn = QPushButton("🔁  Start Fresh")
        fresh_btn.setStyleSheet("color:#cc2200; padding:6px 14px;")
        fresh_btn.clicked.connect(self._do_start_fresh)
        btn_row.addWidget(fresh_btn)

        root.addLayout(btn_row)

    def _build_scan_table(self) -> QTableWidget:
        data         = self.data
        train_llB2   = data.get('train_llB2')    # (n_scan, n_quad) or None
        train_llBPMQ = data.get('train_llBPMQ')  # (n_scan, n_bpm) or None
        quads_to_scan = data.get('quads_to_scan', [])
        bpm_names     = data.get('BPM_names', [])
        n_scans  = len(train_llB2)   if train_llB2   is not None else 0
        n_init   = self.worker.params.get('n_init', 3)

        quad_short = [q.split(':')[-1] for q in quads_to_scan]
        headers = ["", "ID",
                   f"Quads B2 [{', '.join(quad_short)}] (T/m²)",
                   "BPMQ Range (mm²)", "BPMQ  μ ± σ", "n_BPM", "⚠️", "📋"]

        tbl = QTableWidget(n_scans, len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setMinimumHeight(180)

        for i in range(n_scans):
            # ── checkbox ──
            chk = QCheckBox()
            chk.setChecked(True)
            chk.stateChanged.connect(self._update_selection_label)
            cw = QWidget(); cl = QHBoxLayout(cw)
            cl.addWidget(chk); cl.setAlignment(Qt.AlignCenter); cl.setContentsMargins(0,0,0,0)
            tbl.setCellWidget(i, 0, cw)

            # ── ID ──
            scan_id = f"init_{i+1}" if i < n_init else f"AL_{i - n_init + 1}"
            tbl.setItem(i, 1, QTableWidgetItem(scan_id))

            # ── quad B2 ──
            if train_llB2 is not None:
                b2_str = "[" + ", ".join(f"{v:.3f}" for v in train_llB2[i]) + "]"
            else:
                b2_str = "—"
            tbl.setItem(i, 2, QTableWidgetItem(b2_str))

            # ── BPMQ stats ──
            bpmq_row = train_llBPMQ[i] if train_llBPMQ is not None else None
            flag_str = "✓"
            if bpmq_row is not None:
                bq_min  = float(bpmq_row.min())
                bq_max  = float(bpmq_row.max())
                bq_mean = float(bpmq_row.mean())
                bq_std  = float(bpmq_row.std())
                n_bpm   = len(bpmq_row)

                rng_item   = QTableWidgetItem(f"[{bq_min:.1f},  {bq_max:.1f}]")
                stats_item = QTableWidgetItem(f"{bq_mean:.1f} ± {bq_std:.1f}")

                if max(abs(bq_min), abs(bq_max)) > 25:
                    rng_item.setForeground(QColor("darkorange"))
                    flag_str = "⚠️ large |BPMQ|"

                tbl.setItem(i, 3, rng_item)
                tbl.setItem(i, 4, stats_item)
                tbl.setItem(i, 5, QTableWidgetItem(str(n_bpm)))
            else:
                for col in [3, 4, 5]:
                    tbl.setItem(i, col, QTableWidgetItem("—"))
            tbl.setItem(i, 6, QTableWidgetItem(flag_str))

            # ── per-row details button ──
            det_btn = QPushButton("📋")
            det_btn.setMaximumWidth(38)
            det_btn.setToolTip("Show full BPMQ array for this scan")
            _row  = bpmq_row.copy() if bpmq_row is not None else None
            _bpms = list(bpm_names)
            _id   = scan_id
            det_btn.clicked.connect(
                lambda _checked, r=_row, b=_bpms, sid=_id:
                self._show_bpmq_details(r, b, sid)
            )
            tbl.setCellWidget(i, 7, det_btn)

        return tbl

    # ── helpers ────────────────────────────────────────────────────────────

    def _show_bpmq_details(self, bpmq_row, bpm_names, scan_id):
        if bpmq_row is None:
            QMessageBox.information(self, "BPMQ Details", "No BPMQ data available.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"BPMQ Details — {scan_id}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"<b>BPMQ values [mm²]</b> at each BPM for scan <b>{scan_id}</b>:"))
        n = len(bpmq_row)
        hdrs = [(b.split(':')[-1] if b else f"BPM_{j}") for j, b in
                enumerate(bpm_names)] if bpm_names else [f"BPM_{j}" for j in range(n)]
        t = QTableWidget(1, n); t.setHorizontalHeaderLabels(hdrs)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        for j, val in enumerate(bpmq_row):
            item = QTableWidgetItem(f"{float(val):.3f}")
            if abs(float(val)) > 25:
                item.setBackground(QColor("#fff3cd"))
            t.setItem(0, j, item)
        lay.addWidget(t)
        close = QPushButton("Close"); close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.exec_()

    def _get_mask(self) -> np.ndarray:
        n = self.scan_table.rowCount()
        return np.array([
            self.scan_table.cellWidget(i, 0).findChild(QCheckBox).isChecked()
            for i in range(n)
        ], dtype=bool)

    def _update_selection_label(self):
        m = self._get_mask()
        self.sel_lbl.setText(f"{m.sum()}/{len(m)} scans selected")

    def _check_all(self):
        for i in range(self.scan_table.rowCount()):
            self.scan_table.cellWidget(i, 0).findChild(QCheckBox).setChecked(True)

    def _uncheck_all(self):
        for i in range(self.scan_table.rowCount()):
            self.scan_table.cellWidget(i, 0).findChild(QCheckBox).setChecked(False)

    @pyqtSlot()
    def _on_worker_truly_paused(self):
        if self._pause_wait_lbl:
            self._pause_wait_lbl.hide()
        self.retrain_btn.setEnabled(True)

    # ── actions ────────────────────────────────────────────────────────────

    def _do_retrain(self):
        mask = self._get_mask()
        if mask.sum() == 0:
            QMessageBox.warning(self, "Retrain",
                "No scans selected — please check at least one scan.")
            return
        self._last_selection_mask = mask
        self.retrain_btn.setEnabled(False)
        self.retrain_status.setText("⏳  Retraining model in background…")

        self._retrain_thread = RetrainThread(self.worker.bpmQscan, mask, parent=self)
        self._retrain_thread.done.connect(self._on_retrain_done)
        self._retrain_thread.error.connect(self._on_retrain_error)
        self._retrain_thread.start()

    @pyqtSlot(dict)
    def _on_retrain_done(self, updated_data):
        self.retrain_status.setText("✅  Retrain complete.")
        self.retrain_btn.setEnabled(True)
        if self._last_selection_mask is not None:
            self.worker.train_data_selection_history.append(
                self._last_selection_mask.copy()
            )
        # Add augmented fields for status bar updates
        updated_data['rejection_log']        = list(self.worker.rejection_log)
        updated_data['quads_max_curr']        = self.data.get('quads_max_curr')
        updated_data['quads_min_curr']        = self.data.get('quads_min_curr')
        updated_data['scan_quads_init_vals']  = self.data.get('scan_quads_init_vals')
        updated_data['is_finished']           = True   # retrain is always a complete fit
        self.data = updated_data
        self.retrain_plot_update.emit(updated_data)

    @pyqtSlot(str)
    def _on_retrain_error(self, msg):
        self.retrain_status.setText("❌  Retrain failed — see error message.")
        self.retrain_btn.setEnabled(True)
        QMessageBox.critical(self, "Retrain Error", msg)

    def _do_resume(self):
        self.worker.resume()
        self.accept()

    def _do_save(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Current Result",
            f"bpmqscan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if fname:
            # evaluated_dfs was stripped from self.data to avoid deep-copy
            # overhead on the Qt signal boundary.  Re-attach it from the live
            # bpmQscan object before writing the pickle so the saved file is
            # complete.
            save_data = dict(self.data)
            if self.worker.bpmQscan is not None:
                save_data['evaluated_dfs'] = self.worker.bpmQscan.evaluated_dfs
            with open(fname, 'wb') as f:
                pickle.dump(save_data, f)
            QMessageBox.information(self, "Saved", f"Data saved to:\n{fname}")

    def _do_start_fresh(self):
        reply = QMessageBox.question(
            self, "Start Fresh",
            "This will <b>abort the current inference entirely</b>.<br>"
            "All collected data will be discarded unless already saved.<br><br>"
            "Are you sure?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.worker.stop()
            self.worker.resume()   # unblock _check_pause so the thread can exit
            self.reject()          # closes dialog; parent cleans up UI


# ─────────────────────────────────────────────────────────────────────────────
# Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    """Background thread running the BPMQscan inference loop."""

    # Original signals
    progress_update  = pyqtSignal(str)
    iteration_done   = pyqtSignal(dict)
    pm_scan_needed   = pyqtSignal(object, object)
    finished         = pyqtSignal(dict)
    error_occurred   = pyqtSignal(str)

    # New signals
    beam_params_read = pyqtSignal(dict)   # {E_MeV_u, mass_number, charge_number, Brho, beta, gamma, source}
    worker_paused    = pyqtSignal()       # emitted the instant the worker blocks in _check_pause

    def __init__(self, config: Dict[str, Any], params: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.params = params
        self.bpmQscan: Optional[BPMQscan] = None
        self.running = True

        # Pause/resume via threading.Event
        self._pause_event = threading.Event()
        self._pause_event.set()   # start un-paused
        self.is_truly_paused = False

        self._pm_callback_result = None

        # ── New bookkeeping ──────────────────────────────────────────────
        # Selection history: list of bool[n_scan] arrays, one per retrain call
        self.train_data_selection_history: List[np.ndarray] = []
        # Rejection log: list of {lB2, reason} dicts
        self.rejection_log: List[Dict] = []
        self._prev_n_penal = 0

    # ── Pause / resume ────────────────────────────────────────────────────

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self.running = False

    def _check_pause(self):
        """
        Call at every AL iteration boundary.
        Blocks the worker thread while paused; emits worker_paused exactly once.
        """
        if not self._pause_event.is_set():
            self.is_truly_paused = True
            self.worker_paused.emit()
            self._pause_event.wait()   # ← blocks here until resume()
            self.is_truly_paused = False

    # ── Rejection tracking ────────────────────────────────────────────────

    def _track_rejections(self):
        """Check whether bpmQscan.llB2_penal grew since the last call."""
        if self.bpmQscan is None:
            return
        penal = self.bpmQscan.llB2_penal
        if penal is None:
            return
        n_now = len(penal)
        for idx in range(self._prev_n_penal, n_now):
            lB2 = penal[idx].detach().cpu().numpy().tolist()
            self.rejection_log.append({
                'lB2': lB2,
                'reason': 'Beam loss / |BPMQ| > 40 mm²'
            })
        self._prev_n_penal = n_now

    # ── PM callback (unchanged) ───────────────────────────────────────────

    def pm_scan_callback(self, xrms: float, yrms: float):
        self._pm_callback_result = (xrms, yrms)

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self):
        try:
            self._run_inference()
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{e}\n\n{traceback.format_exc()}")

    def _build_augmented_data(self, quads_max, quads_min, scan_init, is_finished=False):
        """
        Build the dict emitted on every iteration_done / finished signal.

        evaluated_dfs is intentionally excluded here.  It is a list of raw
        BPM DataFrames that grows with every accepted scan.  PyQt5 queued
        connections deep-copy the entire signal argument when crossing thread
        boundaries, so including evaluated_dfs would cause an O(N) deep-copy
        on every iteration — easily 30-60 s of latency by mid-run.

        evaluated_dfs is accessed directly from self.bpmQscan when actually
        needed (Data Quality dialog, Save).

        is_finished: True only when emitted by the finished signal (after the
        final clean fit).  Controls whether _update_cs_table labels the last
        row "final" or "AL k".
        """
        d = self.bpmQscan.get_data()
        d.pop('evaluated_dfs', None)          # ← exclude from signal payload
        d['rejection_log']        = list(self.rejection_log)
        d['quads_max_curr']       = quads_max
        d['quads_min_curr']       = quads_min
        d['scan_quads_init_vals'] = scan_init
        d['quad_scan_bound']      = self.params.get('quad_scan_bound', 45.0)
        d['quads_init_rel_size']  = [self.params.get('initial_quad_scan_bound_ratio', 0.2)] * len(quads_max)
        d['is_finished']          = is_finished
        return d

    def _run_inference(self):
        p   = self.params
        cfg = self.config

        self.progress_update.emit("Initializing BPMQscan…")

        # ── Lattice ───────────────────────────────────────────────────────
        flame_file = p['FLAME_file']
        if not Path(flame_file).exists():
            raise FileNotFoundError(f"FLAME file not found: {flame_file}")

        lattice_dicts = fmlat.combine_lattice_elements_quads_only_w_live_update(
            flame_file,
            from_element=p.get('from_elem'),
            to_element=p.get('to_elem', "BDS_BTS:PM_D5567")
        )
        for elem in lattice_dicts:
            elem['name'] = fmlat.fmname2mpname(elem['name'])

        all_quads     = [e['name'] for e in lattice_dicts if e['type'] == 'quadrupole']
        quads_to_scan = p.get('quads_to_scan') or all_quads[:2]
        bpm_names     = [e['name'] for e in lattice_dicts if 'BPM' in e['name']][1:]
        pm_names      = [e['name'] for e in lattice_dicts if ':PM'  in e['name']]

        self.progress_update.emit(
            f"Quads: {quads_to_scan}\nBPMs: {bpm_names}"
        )

        # ── Machine IO ────────────────────────────────────────────────────
        # Fix 1: construct machineIO BEFORE reading beam params so that all PV
        # access in this thread goes through the same object.  Config values are
        # passed directly to the constructor — the old pattern of
        #   construct_machineIO(test=True) + setattr(machineIO, f"_{k}", v)
        # was broken in two ways: (a) test=True silently disabled all ensure_set
        # calls on real hardware, and (b) the setattr keys
        # ("check_chopper_blocking", "fetch_data_resample_rate") did not match
        # any real attribute names, so config was never applied.
        scan_quads_init_vals = None
        quads_CSETs = None
        if p['use_real_machine']:
            mio_cfg = p.get('machine_io', cfg.get('machine_io', {}))
            machineIO = construct_machineIO(
                fetch_data_time_span           = mio_cfg.get('fetch_data_time_span', 2.0),
                ensure_set_timewait_after_ramp = mio_cfg.get('ensure_set_timewait_after_ramp', 0.2),
                sample_interval                = mio_cfg.get('fetch_data_resample_rate', 0.2),
                verbose                        = mio_cfg.get('verbose', False),
                test                           = mio_cfg.get('test', False),
            )
        else:
            machineIO = None

        # ── Beam parameters ───────────────────────────────────────────────
        E_MeV_u       = float(p.get('E_MeV_u',       cfg['virtual']['E_MeV_u']))
        mass_number   = int  (p.get('mass_number',   cfg['virtual']['mass_number']))
        charge_number = int  (p.get('charge_number', cfg['virtual']['charge_number']))
        bp_source     = "Config"

        if p['use_real_machine'] and p.get('auto_read_beam_params', True):
            try:
                # Fix 2: use machineIO.fetch_data (phantasy-managed CA context)
                # instead of bare epics.caget_many.  Calling caget_many directly
                # from a Qt worker thread that also drives phantasy's CA context
                # (via machineIO.fetch_data / ensure_set) corrupts the internal
                # pthread mutex → "Assertion robust || ... FUTEX_OWNER_DIED ...
                # failed. Aborted".
                pv_E = p.get('pv_E_MeV_u',      'BDS_BTS:BEAM_E_MEV_U')
                pv_A = p.get('pv_mass_number',   'BDS_BTS:BEAM_A')
                pv_Q = p.get('pv_charge_number', 'BDS_BTS:BEAM_Q')
                df_bp = machineIO.fetch_data([pv_E, pv_A, pv_Q], time_span=1.0)
                vals  = [df_bp[pv_E].mean(), df_bp[pv_A].mean(), df_bp[pv_Q].mean()]
                if all(v is not None and not np.isnan(float(v)) for v in vals):
                    E_MeV_u, mass_number, charge_number = (
                        float(vals[0]),
                        int(round(float(vals[1]))),
                        int(round(float(vals[2]))),
                    )
                    bp_source = "EPICS ✓"
                    self.progress_update.emit(
                        f"EPICS beam params: E={E_MeV_u} MeV/u, A={mass_number}, Q={charge_number}"
                    )
                else:
                    self.progress_update.emit(
                        "EPICS beam-param PVs returned None — falling back to config values."
                    )
            except Exception as ex:
                self.progress_update.emit(
                    f"EPICS beam-param read failed ({ex}) — using config values."
                )

        Brho, beta, gamma = compute_beam_derived(E_MeV_u, mass_number, charge_number)
        self.beam_params_read.emit({
            'E_MeV_u':      E_MeV_u,
            'mass_number':  mass_number,
            'charge_number':charge_number,
            'Brho': Brho, 'beta': beta, 'gamma': gamma,
            'source': bp_source,
        })

        # ── Read initial quad currents ────────────────────────────────────
        quad_scan_bound           = p.get('quad_scan_bound', 45.0)
        initial_quad_scan_bound_ratio = p.get('initial_quad_scan_bound_ratio', 0.2)

        if p['use_real_machine']:
            quads_CSETs = [pv + ':I_CSET' for pv in quads_to_scan]
            self.progress_update.emit("Reading initial quad currents…")
            df_init = machineIO.fetch_data(quads_CSETs, time_span=2.0)
            scan_quads_init_vals = df_init[quads_CSETs].mean().tolist()
            print("quads_CSETs", quads_CSETs)
            print("scan_quads_init_vals", scan_quads_init_vals)
            quads_max = [v + quad_scan_bound           for v in scan_quads_init_vals]
            quads_min = [max(v - quad_scan_bound, 0.1) for v in scan_quads_init_vals]
        else:
            scan_quads_init_vals = None
            nominal = 75.0
            quads_max = [nominal + quad_scan_bound]            * len(quads_to_scan)
            quads_min = [max(nominal - quad_scan_bound, 0.1)] * len(quads_to_scan)

        quads_init_rel = [initial_quad_scan_bound_ratio] * len(quads_to_scan)

        # ── Construct BPMQscan ────────────────────────────────────────────
        kwargs = dict(
            lattice_dicts          = lattice_dicts,
            quads_to_scan          = quads_to_scan,
            quads_max_curr         = quads_max,
            quads_min_curr         = quads_min,
            quads_tol_curr         = [0.3] * len(quads_to_scan),
            quads_init_rel_size    = quads_init_rel,
            BPM_names              = bpm_names,
            PM_names               = pm_names,
            BPMQ_model_type        = cfg['bpmq']['model_type'],
            machineIO              = machineIO,
            bootstrap              = cfg['model']['bootstrap'],
            sample_model_err       = cfg['model']['sample_model_err'],
            fit_err                = cfg['model']['fit_err'],
            plot_history           = False,   # must be False: plt calls from worker thread crash Qt
            plot_ellipse           = False,   # GUI draws ellipses itself in _update_ellipse_plot
            set_manually           = False,
            cs_ref                 = None,
            n_init                 = p['n_init'],
            batch_size             = cfg['model']['batch_size'],
            n_batch_padding_factor = cfg['model']['n_batch_padding_factor'],
            num_restarts           = cfg['model']['num_restarts'],
            virtual_beamQerr       = cfg['virtual']['beamQ_err'],
            virtual_beamQmodelerr  = cfg['virtual']['beamQ_model_err'],
        )
        self.bpmQscan = BPMQscan(E_MeV_u, mass_number, charge_number, **kwargs)

        # ── Initialization ────────────────────────────────────────────────
        self.progress_update.emit(f"Running initialization (n_init={p['n_init']})…")
        self.bpmQscan.initialize(n_init=p['n_init'])
        self._track_rejections()
        self.iteration_done.emit(
            self._build_augmented_data(quads_max, quads_min, scan_quads_init_vals)
        )

        if not self.running:
            return

        # ── Active-learning loop ──────────────────────────────────────────
        for i in range(p['n_qScan']):
            if not self.running:
                break

            self._check_pause()   # ← pause point; blocks if user clicked Stop
            if not self.running:
                break

            self.progress_update.emit(f"AL step {i+1}/{p['n_qScan']}…")

            if p['do_pmScan']:
                candidate_lB2, _ = self.bpmQscan.query_candidate_for_PMscan()
                self._pm_callback_result = None
                self.pm_scan_needed.emit(candidate_lB2, self.pm_scan_callback)
                while self._pm_callback_result is None and self.running:
                    self.msleep(100)
                if not self.running:
                    break
                xrms, yrms = self._pm_callback_result
                self.bpmQscan.concat_PM_train_data(
                    lB2=candidate_lB2, lxrms=[xrms], lyrms=[yrms]
                )
                self.bpmQscan.train_model(_record_state=True)
            else:
                is_converged = self.bpmQscan.step()
                if is_converged:
                    self.progress_update.emit(
                        f"[Converged] Discriminability below floor at AL {i+1}. Stopping early."
                    )
                    self._track_rejections()
                    self.iteration_done.emit(
                        self._build_augmented_data(quads_max, quads_min, scan_quads_init_vals)
                    )
                    break

            self._track_rejections()
            self.iteration_done.emit(
                self._build_augmented_data(quads_max, quads_min, scan_quads_init_vals)
            )

        if not self.running:
            return

        # ── Final clean fit ───────────────────────────────────────────────
        # Always run regardless of n_qScan (including single-shot n_qScan=0).
        # No model error sampling, no bootstrap, no error fitting — deterministic
        # best estimate from whatever training data was collected.
        self.progress_update.emit("Running final clean fit…")
        self.bpmQscan.train_model(
            sample_model_err=False, bootstrap=False, fit_err=False, _record_state=True
        )

        if p['use_real_machine'] and quads_CSETs is not None and scan_quads_init_vals is not None:
            # Fix 2: use machineIO.caput (one PV at a time) to stay within the
            # same CA context used by fetch_data, instead of bare caput_many.
            for pv, val in zip(quads_CSETs, scan_quads_init_vals):
                machineIO.caput(pv, val)
            self.progress_update.emit("Quads restored to initial currents.")

        self.finished.emit(
            self._build_augmented_data(quads_max, quads_min, scan_quads_init_vals, is_finished=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Immediate EPICS beam-param reader (fires when auto-read checkbox is ticked)
# ─────────────────────────────────────────────────────────────────────────────

class EpicsBeamParamReader(QThread):
    """
    Reads E, A, Q from EPICS immediately when 'Auto-read from EPICS' is toggled.
    Emits result(dict) on success or error(str) on failure.
    """
    result = pyqtSignal(dict)
    error  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from epics import caget
            # Energy: direct PV
            E = caget("ACS_DIAG:BPM_ENGY7E:KE_RD", timeout=3.0)
            if E is None:
                self.error.emit("EPICS caget returned None for: ACS_DIAG:BPM_ENGY7E:KE_RD")
                return

            # Ion source index → mass number + species
            Q = 8  # charge number is fixed at 8 for this machine
            scs = caget("ACS_DIAG:DEST:ACTIVE_ION_SOURCE", timeout=3.0)
            if scs is None:
                self.error.emit("EPICS caget returned None for: ACS_DIAG:DEST:ACTIVE_ION_SOURCE")
                return
            scs = int(scs)
            A = caget(f"FE_ISRC{scs}:BEAM:A_BOOK", timeout=3.0)
            if A is None:
                self.error.emit(f"EPICS caget returned None for: FE_ISRC{scs}:BEAM:A_BOOK")
                return
            A = int(round(float(A)))
            E = float(E)

            Brho, beta, gamma = compute_beam_derived(E, A, Q)
            self.result.emit({
                'E_MeV_u': E, 'mass_number': A, 'charge_number': Q,
                'Brho': Brho, 'beta': beta, 'gamma': gamma,
                'source': 'EPICS ✓',
            })
        except ImportError:
            self.error.emit("epics module not available — cannot auto-read beam params.")
        except Exception as ex:
            import traceback
            self.error.emit(f"EPICS read failed: {ex}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# MachineIO Settings Dialog
# ─────────────────────────────────────────────────────────────────────────────

class MachineIOSettingsDialog(QDialog):
    """
    Pop-up dialog for editing machineIO construction parameters.

    Parameters exposed:
        fetch_data_time_span           – float  (seconds, data collection window)
        fetch_data_resample_rate       – float  (seconds, resample interval)
        ensure_set_timewait_after_ramp – float  (seconds, settle time after ramp)
        test                           – bool   (bypass isOK checks / dry-run)
    """

    def __init__(self, current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MachineIO Settings")
        self.setMinimumWidth(420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(8)

        # ── fetch_data_time_span ──────────────────────────────────────────
        self.fetch_ts = QDoubleSpinBox()
        self.fetch_ts.setRange(0.1, 60.0)
        self.fetch_ts.setDecimals(1)
        self.fetch_ts.setSingleStep(0.5)
        self.fetch_ts.setSuffix("  s")
        self.fetch_ts.setValue(current.get('fetch_data_time_span', 2.0))
        self.fetch_ts.setToolTip(
            "Duration over which PV data is collected at each scan point.\n"
            "Longer → more stable average; shorter → faster scan.\n"
            "construct_machineIO(fetch_data_time_span=…)"
        )
        form.addRow("fetch_data_time_span:", self.fetch_ts)

        # ── fetch_data_resample_rate ──────────────────────────────────────
        self.resample = QDoubleSpinBox()
        self.resample.setRange(0.01, 5.0)
        self.resample.setDecimals(2)
        self.resample.setSingleStep(0.05)
        self.resample.setSuffix("  s")
        self.resample.setValue(current.get('fetch_data_resample_rate', 0.2))
        self.resample.setToolTip(
            "Resampling interval applied to the raw time-series DataFrame.\n"
            "construct_machineIO(sample_interval=…)"
        )
        form.addRow("fetch_data_resample_rate:", self.resample)

        # ── ensure_set_timewait_after_ramp ────────────────────────────────
        self.timewait = QDoubleSpinBox()
        self.timewait.setRange(0.0, 30.0)
        self.timewait.setDecimals(2)
        self.timewait.setSingleStep(0.1)
        self.timewait.setSuffix("  s")
        self.timewait.setValue(current.get('ensure_set_timewait_after_ramp', 0.2))
        self.timewait.setToolTip(
            "Extra sleep after ensure_set confirms the quad has ramped.\n"
            "Allows power-supply transients to settle before data is read.\n"
            "construct_machineIO(ensure_set_timewait_after_ramp=…)"
        )
        form.addRow("ensure_set_timewait_after_ramp:", self.timewait)

        # ── test (bool) ───────────────────────────────────────────────────
        self.test_combo = QComboBox()
        self.test_combo.addItem("False  — live machine (isOK checks active)", False)
        self.test_combo.addItem("True   — dry-run / test mode (skip isOK)",   True)
        cur_test = current.get('test', False)
        self.test_combo.setCurrentIndex(1 if cur_test else 0)
        self.test_combo.setToolTip(
            "When True: _fetch_data_wrapper and _ensure_set_wrapper skip\n"
            "the isOK_PVs chopper-state check, and ensure_set returns\n"
            "immediately without writing to PVs.\n"
            "construct_machineIO(test=…)"
        )
        form.addRow("test:", self.test_combo)

        layout.addLayout(form)

        # ── Hint label ────────────────────────────────────────────────────
        note = QLabel(
            "<i>Changes take effect at the next <b>Start Inference</b>.<br>"
            "machineIO is constructed fresh on each run — no live instance is patched.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555; font-size:8pt;")
        layout.addWidget(note)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_values(self) -> dict:
        """Return the settings dict ready to merge into config['machine_io']."""
        return {
            'fetch_data_time_span':           self.fetch_ts.value(),
            'fetch_data_resample_rate':        self.resample.value(),
            'ensure_set_timewait_after_ramp':  self.timewait.value(),
            'test':                            self.test_combo.currentData(),
        }


class CSInferenceApp(QMainWindow):
    """Main application window — enhanced edition."""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config          = self._load_config(config_path)
        self.machineio_cfg   = dict(self.config.get('machine_io', {}))  # mutable copy
        self.worker: Optional[InferenceWorker] = None
        self.bpmQscan_data: Optional[dict] = None
        self._dq_dialog: Optional[DataQualityDialog] = None
        # Persistent reference to the bpmQscan object — kept alive even after
        # the worker thread exits so that _save_data can re-attach evaluated_dfs.
        self._last_bpmQscan = None

        self.setWindowTitle("CS Inference Tool — Live Monitoring")
        self.setGeometry(80, 80, 1700, 960)

        self._build_ui()
        self._connect_signals()

    # ── Config helpers ────────────────────────────────────────────────────

    def _load_config(self, path: Optional[str]) -> dict:
        if path and Path(path).exists():
            with open(path) as f:
                return yaml.safe_load(f)
        default = Path(__file__).parent / "config_default.yaml"
        if default.exists():
            with open(default) as f:
                return yaml.safe_load(f)
        return self._hardcoded_defaults()

    def _hardcoded_defaults(self) -> dict:
        return {
            'machine_io': {
                'check_chopper_blocking': True,
                'fetch_data_time_span': 5,
                'ensure_set_timewait_after_ramp': 0.5,
                'fetch_data_resample_rate': 0.2,
                'verbose': True,
            },
            'bpmq': {'model_type': 'TIS161_GP'},
            'inference': {
                'n_init': 3, 'n_qScan': 6, 'do_pmScan': False,
                'xnemit_prior': None, 'ynemit_prior': None,
            },
            'lattice': {
                'FLAME_file': 'test_LS3_Target.lat',
                'from_elem': None,
                'to_elem': 'BDS_BTS:PM_D5567',
            },
            'quads': {
                'quads_to_scan':               None,
                'quad_scan_bound':             45.0,
                'initial_quad_scan_bound_ratio': 0.2,
            },
            'virtual': {
                'E_MeV_u': 130, 'mass_number': 124, 'charge_number': 49,
                'beamQ_err': 1.0, 'beamQ_model_err': 0.5,
            },
            'model': {
                'batch_size': 8, 'n_batch_padding_factor': 16,
                'num_restarts': 5, 'bootstrap': False,
                'sample_model_err': True, 'fit_err': True,
            },
            'plotting': {
                'plot_history': False, 'plot_ellipse': True,
            },
        }

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)

        # ── Row 1: Config + MachineIO + Mode ─────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Config:</b>"))
        self.config_label = QLabel("(default)")
        top.addWidget(self.config_label)
        load_cfg = QPushButton("Load Config…")
        load_cfg.clicked.connect(self._load_config_file)
        top.addWidget(load_cfg)

        # MachineIO settings button + compact summary label
        mio_btn = QPushButton("MachineIO…")
        mio_btn.setToolTip("Edit machineIO construction settings")
        mio_btn.clicked.connect(self._open_machineio_dialog)
        top.addWidget(mio_btn)
        self.machineio_summary_lbl = QLabel()
        self.machineio_summary_lbl.setStyleSheet("color:#444; font-size:8pt;")
        top.addWidget(self.machineio_summary_lbl)

        top.addStretch()
        top.addWidget(QLabel("<b>Mode:</b>"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Virtual Machine", "Real Machine"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)
        root.addLayout(top)

        # Initialise summary label text now that the widget exists
        self._update_machineio_btn_label()

        # ── Row 2: Beam Parameters panel ─────────────────────────────────
        root.addWidget(self._build_beam_params_panel())

        # ── Row 3: Main splitter (Input | Plot | Info) ────────────────────
        splitter = QSplitter(Qt.Horizontal)
        self.input_panel = self._build_input_panel()
        splitter.addWidget(self.input_panel)
        self.plot_widget = self._build_plot_widget()
        splitter.addWidget(self.plot_widget)
        self.info_tabs = self._build_info_tabs()
        splitter.addWidget(self.info_tabs)
        splitter.setSizes([370, 720, 480])
        root.addWidget(splitter, stretch=1)

        # ── Row 4: Inference Status bar ───────────────────────────────────
        root.addWidget(self._build_status_bar())

        # ── Row 5: Control buttons ────────────────────────────────────────
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Inference")
        self.start_btn.setStyleSheet("font-weight:bold; padding:6px 16px;")
        self.start_btn.clicked.connect(self._start_inference)
        ctrl.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏸  Stop / Review Data")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_inference)
        ctrl.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾  Save Data…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_data)
        ctrl.addWidget(self.save_btn)

        ctrl.addStretch()


        root.addLayout(ctrl)

    # ── Beam Parameters panel ─────────────────────────────────────────────

    def _build_beam_params_panel(self) -> QGroupBox:
        """
        Top panel showing E, A, Q with derived Bρ/β/γ.
        Auto-read from EPICS or manual, always overridable.
        """
        grp = QGroupBox("Beam Parameters")
        grp.setMaximumHeight(120)
        outer = QHBoxLayout(grp)
        outer.setSpacing(12)

        # ── Editable inputs (E, A, Q) ─────────────────────────────────────
        form = QGridLayout()
        form.setSpacing(6)

        form.addWidget(QLabel("<b>E (MeV/u):</b>"), 0, 0)
        self.beam_E_spin = QDoubleSpinBox()
        self.beam_E_spin.setRange(0.1, 5000.0)
        self.beam_E_spin.setDecimals(2)
        self.beam_E_spin.setSingleStep(1.0)
        self.beam_E_spin.setValue(self.config['virtual']['E_MeV_u'])
        self.beam_E_spin.setMinimumWidth(90)
        self.beam_E_spin.valueChanged.connect(self._update_derived_beam_params)
        form.addWidget(self.beam_E_spin, 0, 1)

        form.addWidget(QLabel("<b>A:</b>"), 0, 2)
        self.beam_A_spin = QSpinBox()
        self.beam_A_spin.setRange(1, 300)
        self.beam_A_spin.setValue(self.config['virtual']['mass_number'])
        self.beam_A_spin.setMinimumWidth(65)
        self.beam_A_spin.valueChanged.connect(self._update_derived_beam_params)
        form.addWidget(self.beam_A_spin, 0, 3)

        form.addWidget(QLabel("<b>Q:</b>"), 0, 4)
        self.beam_Q_spin = QSpinBox()
        self.beam_Q_spin.setRange(1, 120)
        self.beam_Q_spin.setValue(self.config['virtual']['charge_number'])
        self.beam_Q_spin.setMinimumWidth(65)
        self.beam_Q_spin.valueChanged.connect(self._update_derived_beam_params)
        form.addWidget(self.beam_Q_spin, 0, 5)

        # ── Source indicator + auto-read checkbox ─────────────────────────
        self.beam_source_lbl = QLabel("Source: <b>Config</b>")
        form.addWidget(self.beam_source_lbl, 1, 0, 1, 4)

        self.auto_read_check = QCheckBox("Auto-read from EPICS")
        self.auto_read_check.setEnabled(False)
        self.auto_read_check.setToolTip(
            "When checked, immediately reads E, A, Q from EPICS and fills\n"
            "the boxes above. Also re-reads at inference start.\n"
            "PVs: BDS_BTS:BEAM_E_MEV_U / :BEAM_A / :BEAM_Q"
        )
        self.auto_read_check.stateChanged.connect(self._on_auto_read_toggled)
        form.addWidget(self.auto_read_check, 1, 4, 1, 2)

        outer.addLayout(form)

        # ── Derived quantities display ─────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(sep)

        deriv = QGridLayout()
        deriv.setSpacing(6)

        bold = QFont(); bold.setBold(True); bold.setPointSize(10)

        lbl_Brho = QLabel("Bρ (T·m):"); deriv.addWidget(lbl_Brho, 0, 0)
        self.val_Brho = QLabel("—"); self.val_Brho.setFont(bold)
        deriv.addWidget(self.val_Brho, 0, 1)

        lbl_beta = QLabel("β (v/c):"); deriv.addWidget(lbl_beta, 0, 2)
        self.val_beta = QLabel("—"); self.val_beta.setFont(bold)
        deriv.addWidget(self.val_beta, 0, 3)

        lbl_gamma = QLabel("γ:"); deriv.addWidget(lbl_gamma, 1, 0)
        self.val_gamma = QLabel("—"); self.val_gamma.setFont(bold)
        deriv.addWidget(self.val_gamma, 1, 1)

        lbl_species = QLabel("Species:"); deriv.addWidget(lbl_species, 1, 2)
        self.val_species = QLabel("—"); self.val_species.setFont(bold)
        deriv.addWidget(self.val_species, 1, 3)

        outer.addLayout(deriv)
        outer.addStretch()

        self._update_derived_beam_params()  # initial calculation
        return grp

    def _update_derived_beam_params(self):
        """Recompute Bρ/β/γ from current spinbox values and update labels."""
        E   = self.beam_E_spin.value()
        A   = self.beam_A_spin.value()
        Q   = self.beam_Q_spin.value()
        Brho, beta, gamma = compute_beam_derived(E, A, Q)
        self.val_Brho.setText(f"{Brho:.3f}")
        self.val_beta.setText(f"{beta:.4f}")
        self.val_gamma.setText(f"{gamma:.4f}")
        self.val_species.setText(f"¹²⁴Xe⁴⁹⁺" if (A == 124 and Q == 49)
                                 else f"A={A}, Q={Q}+")

    # ── Input panel ───────────────────────────────────────────────────────

    def _build_input_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(340)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        layout.addWidget(QLabel("<h3>Input Parameters</h3>"))

        # ── Inference group ───────────────────────────────────────────────
        grp_inf = QGroupBox("Inference")
        form_inf = QGridLayout()
        form_inf.addWidget(QLabel("n_init:"), 0, 0)
        self.n_init_spin = QSpinBox()
        self.n_init_spin.setRange(1, 20)
        self.n_init_spin.setValue(self.config['inference']['n_init'])
        self.n_init_spin.setToolTip(
            "Number of initialization scans before active learning.\n"
            "The 1st scan is ALWAYS a passive read at the current operating\n"
            "point — quads are never set for that first point (avoids any\n"
            "set/readback discrepancy).\n"
            "n_init=1 + n_qScan=0 → pure single-shot: read once, fit, done."
        )
        form_inf.addWidget(self.n_init_spin, 0, 1)
        form_inf.addWidget(QLabel("n_qScan:"), 1, 0)
        self.n_qScan_spin = QSpinBox()
        self.n_qScan_spin.setRange(0, 50)
        self.n_qScan_spin.setValue(self.config['inference']['n_qScan'])
        self.n_qScan_spin.setToolTip(
            "Active learning budget — number of additional quad scans after initialization.\n"
            "n_qScan=0: no active learning; inference uses only the init scan(s).\n\n"
            "Single-shot mode = n_init=1  AND  n_qScan=0:\n"
            "  one passive read at the current operating point, no quads ever moved."
        )
        form_inf.addWidget(self.n_qScan_spin, 1, 1)
        self.do_pmScan_check = QCheckBox("do_pmScan (manual PM)")
        self.do_pmScan_check.setChecked(self.config['inference']['do_pmScan'])
        form_inf.addWidget(self.do_pmScan_check, 2, 0, 1, 2)
        grp_inf.setLayout(form_inf)
        layout.addWidget(grp_inf)

        # ── Lattice group ─────────────────────────────────────────────────
        grp_lat = QGroupBox("Lattice")
        form_lat = QGridLayout()
        form_lat.addWidget(QLabel("FLAME file:"), 0, 0)
        self.flame_edit = QLineEdit(self.config['lattice']['FLAME_file'])
        form_lat.addWidget(self.flame_edit, 0, 1)
        browse_btn = QPushButton("…"); browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(self._browse_flame_file)
        form_lat.addWidget(browse_btn, 0, 2)
        form_lat.addWidget(QLabel("to_elem:"), 1, 0)
        self.to_elem_edit = QLineEdit(self.config['lattice']['to_elem'])
        form_lat.addWidget(self.to_elem_edit, 1, 1, 1, 2)
        grp_lat.setLayout(form_lat)
        layout.addWidget(grp_lat)

        # ── Quads (optional overrides) ────────────────────────────────────
        grp_quad = QGroupBox("Quads (optional overrides)")
        form_quad = QGridLayout()
        form_quad.setSpacing(6)

        form_quad.addWidget(QLabel("quads_to_scan:"), 0, 0)
        self.quads_edit = QLineEdit()
        self.quads_edit.setPlaceholderText("e.g. LS3_BTS:PSQ_D4713, LS3_BTS:PSQ_D4718")
        _cfg_quads = self.config.get('quads', {}).get('quads_to_scan')
        self.quads_edit.setText(
            ', '.join(_cfg_quads) if _cfg_quads else 'LS3_BTS:PSQ_D4713, LS3_BTS:PSQ_D4718'
        )
        form_quad.addWidget(self.quads_edit, 0, 1, 1, 3)

        form_quad.addWidget(QLabel("Scan bound (A):"), 1, 0)
        self.quad_scan_bound_spin = QDoubleSpinBox()
        self.quad_scan_bound_spin.setRange(1.0, 500.0)
        self.quad_scan_bound_spin.setDecimals(1)
        self.quad_scan_bound_spin.setSingleStep(5.0)
        self.quad_scan_bound_spin.setValue(45.0)
        self.quad_scan_bound_spin.setToolTip(
            "quads_max_curr = init_current + scan_bound\n"
            "quads_min_curr = max(init_current − scan_bound, 0.1)\n"
            "Matches notebook: quad_scan_bound = 45"
        )
        form_quad.addWidget(self.quad_scan_bound_spin, 1, 1)

        form_quad.addWidget(QLabel("Init bound ratio:"), 1, 2)
        self.quad_init_ratio_spin = QDoubleSpinBox()
        self.quad_init_ratio_spin.setRange(0.01, 1.0)
        self.quad_init_ratio_spin.setDecimals(2)
        self.quad_init_ratio_spin.setSingleStep(0.05)
        self.quad_init_ratio_spin.setValue(0.2)
        self.quad_init_ratio_spin.setToolTip(
            "quads_init_rel_size = init_bound_ratio  (fraction of scan_bound)\n"
            "Initial random scans stay within ±ratio×bound of init_current\n"
            "Matches notebook: initial_quad_scan_bound_ratio = 0.2"
        )
        form_quad.addWidget(self.quad_init_ratio_spin, 1, 3)

        _hint = QLabel(
            "<i>max = init + bound,  min = max(init − bound, 0.1),  "
            "init range = ±ratio × bound</i>"
        )
        _hint.setStyleSheet("color:#555; font-size:8pt;")
        _hint.setWordWrap(True)
        form_quad.addWidget(_hint, 2, 0, 1, 4)

        grp_quad.setLayout(form_quad)
        layout.addWidget(grp_quad)

        # ── Scan Quadrupole Settings (populated during inference) ──────────
        self.grp_ctx_quads = QGroupBox("Scan Quadrupole Settings  (live)")
        ctx_quads_layout = QVBoxLayout()
        ctx_quads_layout.setContentsMargins(6, 4, 6, 4)
        self.ctx_quads_lbl = QLabel("<i>Will appear after start</i>")
        self.ctx_quads_lbl.setWordWrap(True)
        ctx_quads_layout.addWidget(self.ctx_quads_lbl)
        self.grp_ctx_quads.setLayout(ctx_quads_layout)
        layout.addWidget(self.grp_ctx_quads)

        # ── BPMs (populated during inference) ─────────────────────────────
        self.grp_ctx_bpms = QGroupBox("BPMs  (live)")
        ctx_bpms_layout = QVBoxLayout()
        ctx_bpms_layout.setContentsMargins(6, 4, 6, 4)
        self.ctx_bpms_lbl = QLabel("<i>Will appear after start</i>")
        self.ctx_bpms_lbl.setWordWrap(True)
        ctx_bpms_layout.addWidget(self.ctx_bpms_lbl)
        self.grp_ctx_bpms.setLayout(ctx_bpms_layout)
        layout.addWidget(self.grp_ctx_bpms)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    # ── Convergence plot ──────────────────────────────────────────────────

    def _build_plot_widget(self) -> QWidget:
        """
        Middle panel: convergence plot inside a QScrollArea.
        The canvas is given a fixed pixel size (1200×820) that is larger than
        a typical panel — the scroll area then provides real scrollbars.
        setWidgetResizable(False) is intentional: it keeps the canvas at its
        declared size so the scrollbars have something to scroll against.
        """
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Placeholder figure — replaced entirely on first inference update
        self.plot_canvas  = FigureCanvas(Figure(figsize=(14, 9)))
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, outer)
        outer_layout.addWidget(self.plot_toolbar)

        # Fixed canvas size → scroll area can actually scroll
        self._plot_canvas_size = (1200, 820)   # (w, h) in pixels; updated on each redraw
        self.plot_canvas.setFixedSize(*self._plot_canvas_size)

        scroll = QScrollArea()
        scroll.setWidget(self.plot_canvas)
        scroll.setWidgetResizable(False)          # MUST be False for scrollbars to work
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        outer_layout.addWidget(scroll, stretch=1)
        self._plot_scroll = scroll               # keep reference for resize

        # Initial placeholder
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Convergence plot will appear here\nafter inference starts.",
                ha='center', va='center', fontsize=12, color='grey')
        ax.axis('off')
        self.plot_canvas.draw()
        return outer

    # ── Info tabs ─────────────────────────────────────────────────────────

    def _build_info_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # ── Tab 0: CS Parameters ──────────────────────────────────────────
        self.cs_table = QTableWidget()
        self.cs_table.setColumnCount(6)
        self.cs_table.setHorizontalHeaderLabels(
            ['αx', 'βx (m)', 'εxn (µm)', 'αy', 'βy (m)', 'εyn (µm)']
        )
        self.cs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.cs_table, "CS Parameters")

        # ── Tab 1: Scan Data (live, mirrors DataQualityDialog table) ──────
        scan_tab = QWidget()
        scan_layout = QVBoxLayout(scan_tab)
        scan_layout.setContentsMargins(4, 4, 4, 4)
        scan_layout.setSpacing(4)

        scan_hdr = QHBoxLayout()
        scan_hdr.addWidget(QLabel("<b>Accepted Scans</b>  (live, read-only)"))
        scan_hdr.addStretch()
        # "Open Review" shortcut — opens the full DataQualityDialog while paused
        self.open_review_btn = QPushButton("⏸ Pause & Review…")
        self.open_review_btn.setEnabled(False)
        self.open_review_btn.setToolTip(
            "Pause inference and open the Data Quality Review dialog\n"
            "to retrain with a subset of scans."
        )
        self.open_review_btn.clicked.connect(self._stop_inference)
        scan_hdr.addWidget(self.open_review_btn)
        scan_layout.addLayout(scan_hdr)

        self.live_scan_table = QTableWidget()
        self.live_scan_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.live_scan_table.setAlternatingRowColors(True)
        self.live_scan_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.live_scan_table.verticalHeader().setVisible(False)
        scan_layout.addWidget(self.live_scan_table)

        # Rejected scans sub-table
        scan_layout.addWidget(QLabel("<b>Rejected Scans</b>"))
        self.live_rej_table = QTableWidget()
        self.live_rej_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.live_rej_table.setAlternatingRowColors(True)
        self.live_rej_table.setMaximumHeight(120)
        self.live_rej_table.verticalHeader().setVisible(False)
        scan_layout.addWidget(self.live_rej_table)

        tabs.addTab(scan_tab, "📊 Scan Data")

        # ── Tab 2: Data Log ───────────────────────────────────────────────
        self.data_log = QTextEdit()
        self.data_log.setReadOnly(True)
        self.data_log.setFont(QFont("Courier", 9))
        tabs.addTab(self.data_log, "Data Log")

        # ── Tab 3: Training Loss ──────────────────────────────────────────
        self.loss_canvas    = FigureCanvas(Figure(figsize=(5, 4)))
        tabs.addTab(self.loss_canvas, "Training Loss")

        # ── Tab 4: Beam Ellipses ──────────────────────────────────────────
        ellipse_tab = QWidget()
        ellipse_layout = QVBoxLayout(ellipse_tab)
        ellipse_layout.setContentsMargins(4, 4, 4, 4)
        ellipse_layout.setSpacing(4)

        range_row = QHBoxLayout(); range_row.setSpacing(6)
        range_row.addWidget(QLabel("x (mm):"))
        self.ellipse_xmin_spin = QDoubleSpinBox()
        self.ellipse_xmin_spin.setRange(-200.0, 0.0); self.ellipse_xmin_spin.setValue(-10.0)
        self.ellipse_xmin_spin.setDecimals(1); self.ellipse_xmin_spin.setSingleStep(1.0)
        self.ellipse_xmin_spin.setMaximumWidth(72)
        range_row.addWidget(self.ellipse_xmin_spin)
        range_row.addWidget(QLabel("to"))
        self.ellipse_xmax_spin = QDoubleSpinBox()
        self.ellipse_xmax_spin.setRange(0.0, 200.0); self.ellipse_xmax_spin.setValue(10.0)
        self.ellipse_xmax_spin.setDecimals(1); self.ellipse_xmax_spin.setSingleStep(1.0)
        self.ellipse_xmax_spin.setMaximumWidth(72)
        range_row.addWidget(self.ellipse_xmax_spin)
        range_row.addSpacing(12)
        range_row.addWidget(QLabel("x' (mrad):"))
        self.ellipse_ymin_spin = QDoubleSpinBox()
        self.ellipse_ymin_spin.setRange(-200.0, 0.0); self.ellipse_ymin_spin.setValue(-10.0)
        self.ellipse_ymin_spin.setDecimals(1); self.ellipse_ymin_spin.setSingleStep(1.0)
        self.ellipse_ymin_spin.setMaximumWidth(72)
        range_row.addWidget(self.ellipse_ymin_spin)
        range_row.addWidget(QLabel("to"))
        self.ellipse_ymax_spin = QDoubleSpinBox()
        self.ellipse_ymax_spin.setRange(0.0, 200.0); self.ellipse_ymax_spin.setValue(10.0)
        self.ellipse_ymax_spin.setDecimals(1); self.ellipse_ymax_spin.setSingleStep(1.0)
        self.ellipse_ymax_spin.setMaximumWidth(72)
        range_row.addWidget(self.ellipse_ymax_spin)
        self.ellipse_autorange_check = QCheckBox("Auto range")
        self.ellipse_autorange_check.setChecked(True)
        self.ellipse_autorange_check.stateChanged.connect(self._on_ellipse_autorange_toggled)
        range_row.addWidget(self.ellipse_autorange_check)
        self._ellipse_apply_btn = QPushButton("Apply")
        self._ellipse_apply_btn.setMaximumWidth(55)
        self._ellipse_apply_btn.clicked.connect(self._on_ellipse_range_apply)
        range_row.addWidget(self._ellipse_apply_btn)
        range_row.addStretch()
        ellipse_layout.addLayout(range_row)

        for w in [self.ellipse_xmin_spin, self.ellipse_xmax_spin,
                  self.ellipse_ymin_spin, self.ellipse_ymax_spin,
                  self._ellipse_apply_btn]:
            w.setEnabled(False)

        self.ellipse_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        ellipse_layout.addWidget(self.ellipse_canvas, stretch=1)
        tabs.addTab(ellipse_tab, "Beam Ellipses")

        return tabs

    # ── Inference Status bar ──────────────────────────────────────────────

    def _build_status_bar(self) -> QFrame:
        """
        Two-line status frame above the control buttons:
          Line 1: step | training pts | rejected count + details button | discriminability | physical flag
          Line 2: Quads detail with scan ranges
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #f0f0f0; border-radius: 4px; padding: 2px; }")
        frame.setMaximumHeight(70)

        outer = QVBoxLayout(frame)
        outer.setSpacing(2)
        outer.setContentsMargins(8, 4, 8, 4)

        # ── Line 1 ────────────────────────────────────────────────────────
        line1 = QHBoxLayout()
        line1.setSpacing(14)

        self.sb_mode_lbl = QLabel("⚪ Ready")
        self.sb_mode_lbl.setFont(QFont("", 10, QFont.Bold))
        line1.addWidget(self.sb_mode_lbl)

        line1.addWidget(_vline())

        self.sb_step_lbl = QLabel("Step: —")
        line1.addWidget(self.sb_step_lbl)

        line1.addWidget(_vline())

        self.sb_pts_lbl = QLabel("Training pts: —")
        line1.addWidget(self.sb_pts_lbl)

        line1.addWidget(_vline())

        # Rejected count + details button
        rej_row = QHBoxLayout(); rej_row.setSpacing(4)
        self.sb_rej_lbl = QLabel("Rejected: 0")
        rej_row.addWidget(self.sb_rej_lbl)
        self.sb_rej_btn = QPushButton("📋 Details")
        self.sb_rej_btn.setMaximumHeight(22)
        self.sb_rej_btn.setEnabled(False)
        self.sb_rej_btn.clicked.connect(self._show_rejected_details)
        rej_row.addWidget(self.sb_rej_btn)
        rej_widget = QWidget(); rej_widget.setLayout(rej_row)
        line1.addWidget(rej_widget)
        line1.addStretch()
        outer.addLayout(line1)

        # ── Line 2 ────────────────────────────────────────────────────────
        self.sb_quads_lbl = QLabel("Quads: —")
        self.sb_quads_lbl.setStyleSheet("color: #444; font-size: 9pt;")
        outer.addWidget(self.sb_quads_lbl)

        return frame

    # ── Signal connections ────────────────────────────────────────────────

    def _connect_signals(self):
        pass  # main connections done in _start_inference; mode connection already done

    def _on_mode_changed(self, idx: int):
        is_real = (idx == 1)
        self.auto_read_check.setEnabled(is_real)
        if not is_real:
            self.auto_read_check.setChecked(False)
            self.beam_source_lbl.setText("Source: <b>Config (Virtual)</b>")
        else:
            self.beam_source_lbl.setText("Source: <b>Config (awaiting EPICS read)</b>")

    def _on_auto_read_toggled(self, state: int):
        """Fire an immediate EPICS read when checkbox is ticked."""
        if state != Qt.Checked:
            for w in [self.beam_E_spin, self.beam_A_spin, self.beam_Q_spin]:
                w.setStyleSheet("")
            self.beam_source_lbl.setText("Source: <b>Config (awaiting EPICS read)</b>")
            return
        _pending = "QDoubleSpinBox,QSpinBox{background:#fff9c4;border:1.5px solid #c8a800;}"
        for w in [self.beam_E_spin, self.beam_A_spin, self.beam_Q_spin]:
            w.setStyleSheet(_pending)
        self.beam_source_lbl.setText(
            "Source: <b style='color:#888800;'>Reading from EPICS…</b>"
        )
        self._epics_reader = EpicsBeamParamReader(parent=self)
        self._epics_reader.result.connect(self._on_epics_read_result)
        self._epics_reader.error.connect(self._on_epics_read_error)
        self._epics_reader.start()

    @pyqtSlot(dict)
    def _on_epics_read_result(self, bp: dict):
        self._on_beam_params_read(bp)
        _filled = "QDoubleSpinBox,QSpinBox{background:#c8f0c8;border:1.5px solid #2a8a2a;}"
        for w in [self.beam_E_spin, self.beam_A_spin, self.beam_Q_spin]:
            w.setStyleSheet(_filled)

    @pyqtSlot(str)
    def _on_epics_read_error(self, msg: str):
        for w in [self.beam_E_spin, self.beam_A_spin, self.beam_Q_spin]:
            w.setStyleSheet("")
        self.beam_source_lbl.setText(
            "Source: <b style='color:#cc0000;'>EPICS read failed</b>"
        )
        self.data_log.append(f"[EPICS auto-read error] {msg}")
        QMessageBox.warning(self, "EPICS Read Failed", msg)



    # ── File / config actions ─────────────────────────────────────────────

    def _browse_flame_file(self):
        cur = self.flame_edit.text()
        start = str(Path(cur).parent) if cur and Path(cur).exists() else str(Path(__file__).parent)
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select FLAME Lattice File", start,
            "Lattice Files (*.lat);;All Files (*)"
        )
        if fname:
            self.flame_edit.setText(fname)

    # ── MachineIO settings dialog ─────────────────────────────────────────

    def _update_machineio_btn_label(self):
        """Show a compact one-line summary of current machineIO settings."""
        c = self.machineio_cfg
        ts   = c.get('fetch_data_time_span',          2.0)
        rs   = c.get('fetch_data_resample_rate',      0.2)
        tw   = c.get('ensure_set_timewait_after_ramp',0.2)
        test = c.get('test', False)
        flag = "  <span style='color:darkorange;'>[TEST]</span>" if test else ""
        self.machineio_summary_lbl.setText(
            f"<i>ts={ts}s  rs={rs}s  wait={tw}s{flag}</i>"
        )

    def _open_machineio_dialog(self):
        dlg = MachineIOSettingsDialog(self.machineio_cfg, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.machineio_cfg.update(dlg.get_values())
            self._update_machineio_btn_label()

    def _load_config_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Config", str(Path(__file__).parent),
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if fname:
            self.config = self._load_config(fname)
            self.machineio_cfg = dict(self.config.get('machine_io', {}))
            self.config_label.setText(Path(fname).name)
            self._update_machineio_btn_label()
            self._populate_inputs_from_config()

    def _populate_inputs_from_config(self):
        self.n_init_spin.setValue(self.config['inference']['n_init'])
        self.n_qScan_spin.setValue(self.config['inference']['n_qScan'])
        self.do_pmScan_check.setChecked(self.config['inference']['do_pmScan'])
        self.flame_edit.setText(self.config['lattice']['FLAME_file'])
        self.to_elem_edit.setText(self.config['lattice']['to_elem'])
        self.beam_E_spin.setValue(self.config['virtual']['E_MeV_u'])
        self.beam_A_spin.setValue(self.config['virtual']['mass_number'])
        self.beam_Q_spin.setValue(self.config['virtual']['charge_number'])
        qcfg = self.config.get('quads', {})
        if qcfg.get('quad_scan_bound') is not None:
            self.quad_scan_bound_spin.setValue(float(qcfg['quad_scan_bound']))
        if qcfg.get('initial_quad_scan_bound_ratio') is not None:
            self.quad_init_ratio_spin.setValue(float(qcfg['initial_quad_scan_bound_ratio']))
        _cfg_quads = qcfg.get('quads_to_scan')
        if _cfg_quads:
            self.quads_edit.setText(', '.join(_cfg_quads))
        self._update_derived_beam_params()

    # ── Gather params ─────────────────────────────────────────────────────

    def _gather_params(self) -> Dict[str, Any]:
        params = {
            'n_init':            self.n_init_spin.value(),
            'n_qScan':           self.n_qScan_spin.value(),
            'do_pmScan':         self.do_pmScan_check.isChecked(),
            'FLAME_file':        self.flame_edit.text(),
            'from_elem':         self.config['lattice']['from_elem'],
            'to_elem':           self.to_elem_edit.text(),
            'use_real_machine':  self.mode_combo.currentIndex() == 1,
            # beam params (may be overridden by EPICS in worker)
            'E_MeV_u':           self.beam_E_spin.value(),
            'mass_number':       self.beam_A_spin.value(),
            'charge_number':     self.beam_Q_spin.value(),
            'auto_read_beam_params': self.auto_read_check.isChecked(),
            # machineIO construction params (from dialog, not config file)
            'machine_io':        dict(self.machineio_cfg),
        }

        # Quad overrides
        if self.quads_edit.text().strip():
            params['quads_to_scan'] = [q.strip() for q in self.quads_edit.text().split(',')]
        else:
            params['quads_to_scan'] = None

        params['quad_scan_bound']               = self.quad_scan_bound_spin.value()
        params['initial_quad_scan_bound_ratio'] = self.quad_init_ratio_spin.value()
        params['quads_max_curr']      = None
        params['quads_min_curr']      = None
        params['quads_init_rel_size'] = None
        return params

    # ── Start / Stop ──────────────────────────────────────────────────────

    def _start_inference(self):
        try:
            params = self._gather_params()
        except Exception as e:
            QMessageBox.critical(self, "Input Error", str(e))
            return

        self.worker = InferenceWorker(self.config, params)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.iteration_done.connect(self._on_iteration)
        self.worker.pm_scan_needed.connect(self._on_pm_scan_needed)
        self.worker.finished.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.beam_params_read.connect(self._on_beam_params_read)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.open_review_btn.setEnabled(True)
        # Total steps = n_init passive reads + n_qScan AL scans + 1 final fit
        self.sb_mode_lbl.setText("🟡 Running…")
        self.sb_mode_lbl.setStyleSheet("color: darkorange; font-weight: bold;")

        self.data_log.append(
            f"\n{'='*60}\n[{datetime.datetime.now()}] Inference started.\n"
        )
        self.worker.start()

    def _stop_inference(self):
        """
        Pause the worker and open the Data Quality Review dialog.
        The user can retrain, resume, save, or start fresh from there.
        """
        if self.worker is None or not self.worker.isRunning():
            return

        self.worker.pause()   # signal worker to block at next _check_pause()

        data = self.bpmQscan_data or {}
        self._dq_dialog = DataQualityDialog(data, self.worker, parent=self)
        self._dq_dialog.retrain_plot_update.connect(self._on_retrain_update)
        self._dq_dialog.finished.connect(self._on_dq_dialog_closed)
        self.sb_mode_lbl.setText("⏸ Paused")
        self.sb_mode_lbl.setStyleSheet("color: #555; font-weight: bold;")
        self._dq_dialog.show()

    @pyqtSlot(int)
    def _on_dq_dialog_closed(self, result: int):
        """Called when DataQualityDialog closes (accept = resume, reject = start fresh)."""
        if self.worker and not self.worker.running:
            # "Start Fresh" was selected
            self.worker.wait(5000)
            self._reset_after_stop()
        else:
            # "Resume AL" — restore running state
            self.sb_mode_lbl.setText("🟡 Running…")
            self.sb_mode_lbl.setStyleSheet("color: darkorange; font-weight: bold;")

    def _reset_after_stop(self):
        """Reset GUI after a complete stop."""
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(bool(self.bpmQscan_data))
        self.sb_mode_lbl.setText("⚫ Stopped")
        self.sb_mode_lbl.setStyleSheet("color: #888; font-weight: bold;")

    # ── Slot: beam params read from EPICS ─────────────────────────────────

    @pyqtSlot(dict)
    def _on_beam_params_read(self, bp: dict):
        """Update spinboxes and derived labels with the authoritative beam params."""
        # Temporarily disconnect to avoid re-triggering _update_derived_beam_params
        self.beam_E_spin.blockSignals(True)
        self.beam_A_spin.blockSignals(True)
        self.beam_Q_spin.blockSignals(True)

        self.beam_E_spin.setValue(bp['E_MeV_u'])
        self.beam_A_spin.setValue(bp['mass_number'])
        self.beam_Q_spin.setValue(bp['charge_number'])

        self.beam_E_spin.blockSignals(False)
        self.beam_A_spin.blockSignals(False)
        self.beam_Q_spin.blockSignals(False)

        self.val_Brho.setText(f"{bp['Brho']:.3f}")
        self.val_beta.setText(f"{bp['beta']:.4f}")
        self.val_gamma.setText(f"{bp['gamma']:.4f}")

        src = bp.get('source', '?')
        color = "#007700" if "EPICS" in src else "#444444"
        self.beam_source_lbl.setText(
            f"Source: <b style='color:{color};'>{src}</b>"
        )

    # ── Slot: per-iteration update ────────────────────────────────────────

    @pyqtSlot(str)
    def _on_progress(self, message: str):
        self.sb_mode_lbl.setText(f"🟡 {message.splitlines()[0][:50]}")
        self.data_log.append(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
        )

    @pyqtSlot(dict)
    def _on_iteration(self, data: dict):
        self.bpmQscan_data = data
        if self.worker and self.worker.bpmQscan is not None:
            self._last_bpmQscan = self.worker.bpmQscan
        self._update_all_views(data)

    @pyqtSlot(dict)
    def _on_retrain_update(self, data: dict):
        """Called after user retrains from DataQualityDialog."""
        self.bpmQscan_data = data
        self._update_all_views(data)

    def _update_all_views(self, data: dict):
        self._update_convergence_plot(data)
        self._update_cs_table(data)
        self._update_scan_data_tab(data)
        self._update_loss_plot(data)
        self._update_ellipse_plot(data)
        self._update_status_bar(data)
        self._update_lattice_context(data)

    # ── Slot: PM scan ──────────────────────────────────────────────────────

    @pyqtSlot(object, object)
    def _on_pm_scan_needed(self, candidate_lB2, callback):
        dialog = PMScanDialog(str(candidate_lB2.detach().numpy()), parent=self)
        if dialog.exec_() == QDialog.Accepted:
            xrms, yrms = dialog.get_values()
            callback(xrms, yrms)
        else:
            self._stop_inference()

    # ── Slot: finished ────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_finished(self, data: dict):
        self.bpmQscan_data = data
        if self.worker and self.worker.bpmQscan is not None:
            self._last_bpmQscan = self.worker.bpmQscan
        self._update_all_views(data)

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.open_review_btn.setEnabled(False)
        self.sb_mode_lbl.setText("🟢 Complete")
        self.sb_mode_lbl.setStyleSheet("color: green; font-weight: bold;")

        self.data_log.append(
            f"\n[{datetime.datetime.now()}] Inference complete.\n"
        )
        QMessageBox.information(self, "Complete", "CS inference finished successfully.")

    # ── Slot: error ───────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_review_btn.setEnabled(False)
        self.sb_mode_lbl.setStyleSheet("color: red; font-weight: bold;")
        QMessageBox.critical(self, "Inference Error", error_msg)

    # ── Plot updaters ─────────────────────────────────────────────────────

    def _update_convergence_plot(self, data: dict):
        cs_ref = data.get('cs_ref')
        if cs_ref is not None:
            try:
                import torch
                if isinstance(cs_ref, torch.Tensor):
                    cs_ref = cs_ref.detach().numpy()
            except ImportError:
                pass
        try:
            # ── Build the plot in its own figure (plot_convergence owns it) ──
            fig_new = plot_convergence(data, cs_ref=cs_ref)

            # ── Fix title clipping: make room above suptitle ─────────────
            try:
                fig_new.subplots_adjust(top=0.90)
            except Exception:
                pass

            # ── Strip budget/step-count lines from ALL text objects ────────
            # Scans every axis AND figure-level texts with NO guards so the
            # scrubbing always runs regardless of what plot_convergence puts in
            # each axis (spines are Line2D objects, so the old
            # "if not ax.lines" guard was always False and the block never ran).
            #
            # Strategy per text object:
            #  - If it contains a "Final CS" line  -> keep from that line onward,
            #    then additionally strip any remaining budget/step-count lines.
            #  - Otherwise                          -> strip budget/step-count lines in-place.
            _BUDGET_KW = (
                'budget', 'n_init meas', 'al steps run',
                'step run', 'steps run',
            )

            def _scrub_text(txt_obj):
                raw = txt_obj.get_text()
                if not raw.strip():
                    return
                lines = raw.split('\n')
                final_cs_idx = next(
                    (i for i, l in enumerate(lines) if 'Final CS' in l), None
                )
                if final_cs_idx is not None:
                    lines = lines[final_cs_idx:]   # drop everything before "Final CS"
                # Remove budget/step-count lines wherever they appear
                kept = [l for l in lines
                        if not any(kw in l.lower() for kw in _BUDGET_KW)]
                txt_obj.set_text('\n'.join(kept))

            # Figure-level texts (suptitle region, etc.)
            for txt in list(fig_new.texts):
                _scrub_text(txt)
            # Every axis
            for ax in fig_new.get_axes():
                for txt in list(ax.texts):
                    _scrub_text(txt)

            # ── Swap the figure into the existing canvas ──────────────────
            # Close the old figure to free memory, then attach the new one.
            old_fig = self.plot_canvas.figure
            plt.close(old_fig)

            self.plot_canvas.figure = fig_new
            fig_new.set_canvas(self.plot_canvas)

            # Resize the canvas widget to match the figure's natural pixel size
            dpi = fig_new.get_dpi()
            fw, fh = fig_new.get_size_inches()
            pw, ph = int(fw * dpi), int(fh * dpi)
            # Ensure at least a minimum so the toolbar stays sensible
            pw = max(pw, 900); ph = max(ph, 600)
            self.plot_canvas.setFixedSize(pw, ph)

        except Exception as e:
            import traceback
            fig_err = Figure(figsize=(8, 5))
            old_fig = self.plot_canvas.figure
            plt.close(old_fig)
            self.plot_canvas.figure = fig_err
            fig_err.set_canvas(self.plot_canvas)
            ax = fig_err.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot error:\n{e}\n\n{traceback.format_exc()}",
                    ha='center', va='top', fontsize=8, family='monospace')
            ax.axis('off')

        self.plot_canvas.draw()

    def _update_cs_table(self, data: dict):
        cs_hist = data.get('reconstructed_cs_history', [])
        if not cs_hist:
            return
        cs_arr = np.array(cs_hist)
        cs_arr[:, [2, 5]] *= 1e6   # → µm·rad
        n_hist = len(cs_arr)
        is_finished = data.get('is_finished', False)

        # "final" label only appears once the process has fully completed and
        # the true final clean-fit result is in the history.  Mid-run, the last
        # row is labelled as the AL step that produced it, not "final".
        if is_finished and n_hist >= 2:
            n_al = n_hist - 2          # init + (n_al AL rows) + final
            row_labels = (
                ["init"]
                + [f"AL {k+1}" for k in range(n_al)]
                + ["final"]
            )[:n_hist]
        else:
            # Still running: label every row after "init" as "AL k"
            n_al = n_hist - 1
            row_labels = (
                ["init"]
                + [f"AL {k+1}" for k in range(n_al)]
            )[:n_hist]
        self.cs_table.setRowCount(n_hist)
        self.cs_table.setVerticalHeaderLabels(row_labels)
        for i in range(n_hist):
            for j in range(6):
                self.cs_table.setItem(i, j, QTableWidgetItem(f"{cs_arr[i, j]:.3g}"))
        self.cs_table.resizeColumnsToContents()

    def _update_scan_data_tab(self, data: dict):
        """
        Refresh the live Scan Data tab in the right panel.
        Rebuilds both the accepted-scans table and the rejected-scans table
        from the latest data snapshot.  Called on every iteration_done signal
        so it always reflects current state without needing to open a dialog.
        """
        train_llB2   = data.get('train_llB2')
        train_llBPMQ = data.get('train_llBPMQ')
        quads_to_scan = data.get('quads_to_scan', [])
        bpm_names     = data.get('BPM_names', [])
        rejection_log = data.get('rejection_log', [])
        n_init        = self.n_init_spin.value()

        quad_short = [q.split(':')[-1] for q in quads_to_scan]
        n_scans    = len(train_llB2) if train_llB2 is not None else 0

        # ── Accepted scans table ──────────────────────────────────────────
        acc_headers = (
            ["ID"] +
            [f"B2 {s} (T/m²)" for s in quad_short] +
            ["BPMQ Range (mm²)", "μ ± σ", "n_BPM", "⚠️", "📋"]
        )
        tbl = self.live_scan_table
        tbl.setColumnCount(len(acc_headers))
        tbl.setHorizontalHeaderLabels(acc_headers)
        tbl.setRowCount(n_scans)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        if len(acc_headers) > 2:
            # stretch the BPMQ range column
            tbl.horizontalHeader().setSectionResizeMode(
                1 + len(quad_short), QHeaderView.Stretch
            )

        for i in range(n_scans):
            scan_id = f"init_{i+1}" if i < n_init else f"AL_{i - n_init + 1}"
            col = 0

            # ID
            tbl.setItem(i, col, QTableWidgetItem(scan_id)); col += 1

            # B2 values
            if train_llB2 is not None:
                for qi in range(len(quad_short)):
                    tbl.setItem(i, col, QTableWidgetItem(
                        f"{float(train_llB2[i][qi]):.3f}"
                    )); col += 1
            else:
                for _ in quad_short:
                    tbl.setItem(i, col, QTableWidgetItem("—")); col += 1

            # BPMQ stats
            bpmq_row = train_llBPMQ[i] if train_llBPMQ is not None else None
            flag_str = "✓"
            if bpmq_row is not None:
                bq_min  = float(bpmq_row.min())
                bq_max  = float(bpmq_row.max())
                bq_mean = float(bpmq_row.mean())
                bq_std  = float(bpmq_row.std())
                n_bpm   = len(bpmq_row)

                rng_item   = QTableWidgetItem(f"[{bq_min:.1f}, {bq_max:.1f}]")
                stats_item = QTableWidgetItem(f"{bq_mean:.1f} ± {bq_std:.1f}")
                nbpm_item  = QTableWidgetItem(str(n_bpm))

                if max(abs(bq_min), abs(bq_max)) > 25:
                    rng_item.setForeground(QColor("darkorange"))
                    flag_str = "⚠️"

                tbl.setItem(i, col,     rng_item);   col += 1
                tbl.setItem(i, col,     stats_item); col += 1
                tbl.setItem(i, col,     nbpm_item);  col += 1
            else:
                for _ in range(3):
                    tbl.setItem(i, col, QTableWidgetItem("—")); col += 1

            tbl.setItem(i, col, QTableWidgetItem(flag_str)); col += 1

            # Per-row BPMQ detail button
            det_btn = QPushButton("📋")
            det_btn.setMaximumWidth(36)
            det_btn.setToolTip("Show full BPMQ array for this scan")
            _row  = bpmq_row.copy() if bpmq_row is not None else None
            _bpms = list(bpm_names)
            _sid  = scan_id
            det_btn.clicked.connect(
                lambda _c, r=_row, b=_bpms, s=_sid:
                    self._show_bpmq_details_main(r, b, s)
            )
            tbl.setCellWidget(i, col, det_btn)

        # ── Rejected scans table ──────────────────────────────────────────
        rej_headers = ["ID"] + [f"B2 {s}" for s in quad_short] + ["Reason"]
        rtbl = self.live_rej_table
        rtbl.setColumnCount(len(rej_headers))
        rtbl.setHorizontalHeaderLabels(rej_headers)
        rtbl.setRowCount(len(rejection_log))
        rtbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        if rejection_log:
            rtbl.horizontalHeader().setSectionResizeMode(
                len(rej_headers) - 1, QHeaderView.Stretch
            )

        for i, entry in enumerate(rejection_log):
            lB2    = entry.get('lB2', [])
            reason = entry.get('reason', '?')
            rtbl.setItem(i, 0, QTableWidgetItem(f"Rej_{i+1}"))
            for qi, b2v in enumerate(lB2[:len(quad_short)]):
                rtbl.setItem(i, 1 + qi, QTableWidgetItem(f"{float(b2v):.3f}"))
            reason_item = QTableWidgetItem(f"🔴 {reason}")
            reason_item.setForeground(QColor('#cc2200'))
            rtbl.setItem(i, 1 + len(quad_short), reason_item)

    def _show_bpmq_details_main(self, bpmq_row, bpm_names, scan_id):
        """Per-BPM BPMQ popup from the live Scan Data tab."""
        if bpmq_row is None:
            QMessageBox.information(self, "BPMQ Details", "No BPMQ data available.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"BPMQ Details — {scan_id}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            f"<b>BPMQ values [mm²]</b> at each BPM for scan <b>{scan_id}</b>:"
        ))
        n    = len(bpmq_row)
        hdrs = ([(b.split(':')[-1] if b else f"BPM_{j}") for j, b in enumerate(bpm_names)]
                if bpm_names else [f"BPM_{j}" for j in range(n)])
        t = QTableWidget(1, n)
        t.setHorizontalHeaderLabels(hdrs)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        for j, val in enumerate(bpmq_row):
            item = QTableWidgetItem(f"{float(val):.3f}")
            if abs(float(val)) > 25:
                item.setBackground(QColor("#fff3cd"))
            t.setItem(0, j, item)
        lay.addWidget(t)
        close = QPushButton("Close"); close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.exec_()

    def _update_loss_plot(self, data: dict):
        self.loss_canvas.figure.clear()
        ax = self.loss_canvas.figure.add_subplot(111)
        loss_hist = data.get('train_loss', [])
        if loss_hist:
            ax.plot(loss_hist, 'o-', label='Train Loss')
            ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
            ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Loss history not available",
                    ha='center', va='center', color='grey')
            ax.axis('off')
        self.loss_canvas.figure.tight_layout()
        self.loss_canvas.draw()

    def _on_ellipse_autorange_toggled(self, state: int):
        manual = (state != Qt.Checked)
        for w in [self.ellipse_xmin_spin, self.ellipse_xmax_spin,
                  self.ellipse_ymin_spin, self.ellipse_ymax_spin,
                  self._ellipse_apply_btn]:
            w.setEnabled(manual)
        if not manual and self.bpmQscan_data:
            self._update_ellipse_plot(self.bpmQscan_data)

    def _on_ellipse_range_apply(self):
        if self.bpmQscan_data:
            self._update_ellipse_plot(self.bpmQscan_data)

    def _get_ellipse_limits(self):
        if self.ellipse_autorange_check.isChecked():
            return None, None
        return ((self.ellipse_xmin_spin.value(), self.ellipse_xmax_spin.value()),
                (self.ellipse_ymin_spin.value(), self.ellipse_ymax_spin.value()))

    def _update_ellipse_plot(self, data: dict):
        self.ellipse_canvas.figure.clear()
        covs_hist = data.get('reconstructed_covs_history', [])
        cs_hist   = data.get('reconstructed_cs_history', [])
        cs_ref    = data.get('cs_ref')

        if not covs_hist and not cs_hist:
            ax = self.ellipse_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Ellipse plot not available",
                    ha='center', va='center', color='grey')
            ax.axis('off'); self.ellipse_canvas.draw(); return

        if cs_ref is not None:
            try:
                import torch
                if isinstance(cs_ref, torch.Tensor):
                    cs_ref = cs_ref.detach().numpy()
            except ImportError:
                pass

        ax1 = self.ellipse_canvas.figure.add_subplot(121)
        ax2 = self.ellipse_canvas.figure.add_subplot(122)

        if covs_hist:
            xcovs_arr, ycovs_arr = covs_hist[-1]
            for xcov in xcovs_arr:
                self._plot_ellipse_from_cov(ax1, xcov, color='steelblue', alpha=0.25, lw=0.8)
            for ycov in ycovs_arr:
                self._plot_ellipse_from_cov(ax2, ycov, color='steelblue', alpha=0.25, lw=0.8)
            from matplotlib.lines import Line2D
            ax1.add_line(Line2D([], [], color='steelblue', alpha=0.7, lw=1.5,
                                label=f'Posterior ({len(xcovs_arr)} samples)'))
            ax2.add_line(Line2D([], [], color='steelblue', alpha=0.7, lw=1.5,
                                label=f'Posterior ({len(ycovs_arr)} samples)'))

        if cs_hist:
            cs_mean = np.array(cs_hist[-1])
            self._plot_cs_ellipse(ax1, cs_mean[:3], color='black', ls='--', lw=1.8, label='Mean recon.')
            self._plot_cs_ellipse(ax2, cs_mean[3:], color='black', ls='--', lw=1.8, label='Mean recon.')

        if cs_ref is not None:
            self._plot_cs_ellipse(ax1, cs_ref[:3], color='red', ls='-', lw=2.0, label='Reference')
            self._plot_cs_ellipse(ax2, cs_ref[3:], color='red', ls='-', lw=2.0, label='Reference')

        xlim, ylim = self._get_ellipse_limits()
        for ax, title in [(ax1, 'x phase space'), (ax2, 'y phase space')]:
            ax.set_title(title)
            ax.set_xlabel('position  (mm)')
            ax.set_ylabel("angle  (mrad)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)

        self.ellipse_canvas.figure.tight_layout()
        self.ellipse_canvas.draw()

    def _plot_ellipse_from_cov(self, ax, cov_2x2: np.ndarray, **kwargs):
        """Draw ellipse from 2×2 covariance matrix in SI (m, rad) → mm/mrad."""
        cov = np.array(cov_2x2, dtype=float)
        for _ in range(20):
            try:
                L = np.linalg.cholesky(cov); break
            except np.linalg.LinAlgError:
                cov[0, 0] += 1e-12; cov[1, 1] += 1e-12
        else:
            return
        t = np.linspace(0, 2 * np.pi, 120)
        e = L.dot(np.array([np.cos(t), np.sin(t)]))
        ax.plot(e[0] * 1e3, e[1] * 1e3, **kwargs)

    def _plot_cs_ellipse(self, ax, cs_3: np.ndarray, **kwargs):
        """Draw ellipse from [alpha, beta, nemit_norm]; βγ from spinboxes."""
        alpha, beta, nemit_norm = float(cs_3[0]), float(cs_3[1]), float(cs_3[2])
        _, beta_rel, gamma_rel = compute_beam_derived(
            self.beam_E_spin.value(), self.beam_A_spin.value(), self.beam_Q_spin.value())
        bg = max(beta_rel * gamma_rel, 1e-9)
        nemit_geom = nemit_norm / bg
        t  = np.linspace(0, 2 * np.pi, 120)
        x  =  np.sqrt(max(nemit_geom * beta, 0.0)) * np.cos(t)
        xp = -np.sqrt(max(nemit_geom / max(beta, 1e-12), 0.0)) * (alpha * np.cos(t) + np.sin(t))
        ax.plot(x * 1e3, xp * 1e3, **kwargs)

    # ── Status bar updaters ───────────────────────────────────────────────

    def _update_status_bar(self, data: dict):
        """Refresh the inference status bar from the latest data snapshot."""
        cs_hist = data.get('reconstructed_cs_history', [])
        n_al    = max(len(cs_hist) - 2, 0)
        n_qScan = self.n_qScan_spin.value()
        if len(cs_hist) == 0:
            step_txt = "Initializing…"
        elif len(cs_hist) == 1:
            step_txt = "init done"
        else:
            step_txt = f"AL {n_al}/{n_qScan}"
        self.sb_step_lbl.setText(f"Step: {step_txt}")

        train_B2 = data.get('train_llB2')
        n_pts = len(train_B2) if train_B2 is not None else 0
        self.sb_pts_lbl.setText(f"Training pts: {n_pts}")

        rej_log = data.get('rejection_log', [])
        n_rej   = len(rej_log)
        self.sb_rej_lbl.setText(f"Rejected: {n_rej}")
        self.sb_rej_btn.setEnabled(n_rej > 0)

        # Quads line
        quads      = data.get('quads_to_scan', [])
        q_max      = data.get('quads_max_curr', [])
        q_min      = data.get('quads_min_curr', [])
        q_init     = data.get('scan_quads_init_vals')
        q_scan_bnd = data.get('quad_scan_bound')
        q_init_rel = data.get('quads_init_rel_size', [])
        if quads:
            parts = []
            for qi, q in enumerate(quads):
                short  = q.split(':')[-1]
                lo_f   = q_min[qi] if q_min and qi < len(q_min) else None
                hi_f   = q_max[qi] if q_max and qi < len(q_max) else None

                # Scan bounds
                scan_str = (f"[{lo_f:.1f}, {hi_f:.1f}] A"
                            if lo_f is not None and hi_f is not None else "—")

                # Init bounds
                ratio = q_init_rel[qi] if q_init_rel and qi < len(q_init_rel) else 0.2
                if q_init and qi < len(q_init) and q_scan_bnd is not None:
                    # Real machine: centre on live initial current
                    ib     = q_scan_bnd * ratio
                    iv     = q_init[qi]
                    ib_lo  = iv - ib if lo_f is None else max(iv - ib, lo_f)
                    ib_hi  = iv + ib if hi_f is None else min(iv + ib, hi_f)
                    init_str = f"[{ib_lo:.1f}, {ib_hi:.1f}] A"
                elif lo_f is not None and hi_f is not None and q_scan_bnd is not None:
                    # Virtual machine: centre on midpoint of scan range
                    iv    = (lo_f + hi_f) / 2.0
                    ib    = q_scan_bnd * ratio
                    ib_lo = max(iv - ib, lo_f)
                    ib_hi = min(iv + ib, hi_f)
                    init_str = f"[{ib_lo:.1f}, {ib_hi:.1f}] A"
                else:
                    init_str = "—"

                parts.append(f"{short}  init {init_str}  scan {scan_str}")
            self.sb_quads_lbl.setText("Quads: " + ",    ".join(parts))

    def _update_lattice_context(self, data: dict):
        """Populate the split Lattice Context sub-panels in the input panel."""
        quads      = data.get('quads_to_scan', [])
        bpms       = data.get('BPM_names', [])
        q_max      = data.get('quads_max_curr', [])
        q_min      = data.get('quads_min_curr', [])
        q_init     = data.get('scan_quads_init_vals')
        q_scan_bnd = data.get('quad_scan_bound')
        q_init_rel = data.get('quads_init_rel_size', [])

        # ── Scan Quadrupole Settings ──────────────────────────────────────
        if quads:
            q_lines = []
            for qi, q in enumerate(quads):
                lo_f = q_min[qi]  if q_min  and qi < len(q_min)  else None
                hi_f = q_max[qi]  if q_max  and qi < len(q_max)  else None
                lo   = f"{lo_f:.1f}" if lo_f is not None else "?"
                hi   = f"{hi_f:.1f}" if hi_f is not None else "?"
                init_s = f"{q_init[qi]:.1f} A" if q_init and qi < len(q_init) else "virtual"

                # Init bound window
                ratio = q_init_rel[qi] if q_init_rel and qi < len(q_init_rel) else 0.2
                if q_init and qi < len(q_init) and q_scan_bnd is not None:
                    # Real machine: init ± ratio×bound, clipped to scan range
                    ib   = q_scan_bnd * ratio
                    iv   = q_init[qi]
                    ib_lo = iv - ib if lo_f is None else max(iv - ib, lo_f)
                    ib_hi = iv + ib if hi_f is None else min(iv + ib, hi_f)
                    ib_str = f"[{ib_lo:.1f}, {ib_hi:.1f}] A"
                elif lo_f is not None and hi_f is not None and q_scan_bnd is not None:
                    # Virtual machine: use midpoint of scan range as nominal centre
                    iv    = (lo_f + hi_f) / 2.0
                    ib    = q_scan_bnd * ratio
                    ib_lo = max(iv - ib, lo_f)
                    ib_hi = min(iv + ib, hi_f)
                    ib_str = f"[{ib_lo:.1f}, {ib_hi:.1f}] A"
                else:
                    ib_str = "—"

                short = q.split(':')[-1]
                q_lines.append(
                    f"<b>{short}</b>:  init {init_s}<br>"
                    f"&nbsp;&nbsp;init bound {ib_str}"
                    f" &nbsp;|&nbsp; scan [{lo}, {hi}] A"
                )
            self.ctx_quads_lbl.setText("<br>".join(q_lines))
        else:
            self.ctx_quads_lbl.setText("<i>Will appear after start</i>")

        # ── BPMs ─────────────────────────────────────────────────────────
        if bpms:
            # One per line for readability
            bpm_lines = [b.split(':')[-1] for b in bpms]
            self.ctx_bpms_lbl.setText("<br>".join(bpm_lines))
        else:
            self.ctx_bpms_lbl.setText("<i>Will appear after start</i>")

    # ── Rejected details popup ────────────────────────────────────────────

    def _show_rejected_details(self):
        if self.bpmQscan_data is None:
            return
        rej_log   = self.bpmQscan_data.get('rejection_log', [])
        quads     = self.bpmQscan_data.get('quads_to_scan', [])
        dlg = RejectedDataDialog(rej_log, quads, parent=self)
        dlg.exec_()

    # ── Save ──────────────────────────────────────────────────────────────

    def _save_data(self):
        if not self.bpmQscan_data:
            QMessageBox.warning(self, "Save", "No data to save.")
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Data",
            f"bpmqscan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if fname:
            # evaluated_dfs is excluded from bpmQscan_data (stripped at emit
            # time to avoid O(N) deep-copy on the Qt queued-connection boundary).
            # Re-attach it from the persistent bpmQscan reference before saving
            # so the pickle file is complete and identical to what BPMQscan
            # save_data() would produce.
            save_data = dict(self.bpmQscan_data)
            if self._last_bpmQscan is not None:
                save_data['evaluated_dfs'] = self._last_bpmQscan.evaluated_dfs
            with open(fname, 'wb') as f:
                pickle.dump(save_data, f)
            QMessageBox.information(self, "Saved", f"Data saved to:\n{fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vline() -> QFrame:
    """Thin vertical separator for the status bar."""
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFrameShadow(QFrame.Sunken)
    f.setMaximumHeight(18)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = CSInferenceApp(config_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()