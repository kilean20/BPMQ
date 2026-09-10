"""
plot_convergence_from_pkl.py  v4
5-panel convergence figure for BPMQscan.

History layout (robust):
  initialize() always produces exactly ONE cs_ensemble_history entry.
  Layout: [init | AL_1 ... AL_k | final]  =>  n_al_actual = len - 2
  n_init / n_qScan from data dict are used only for annotations.

Layout (2x3 grid):
  Row 0: alpha_x,y | beta_x,y | emittance_x,y
  Row 1: posterior volume+delta | discriminability+MMD4D | text summary
"""

import sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional


def _ens_from_covs(covs_history, bg):
    out = []
    for (xc, yc) in covs_history:
        xc, yc = np.asarray(xc), np.asarray(yc)
        cs = np.zeros((xc.shape[0], 6))
        for b in range(xc.shape[0]):
            x, y = xc[b], yc[b]
            xne = np.sqrt(max(x[0,0]*x[1,1]-x[0,1]**2, 1e-30))*bg
            yne = np.sqrt(max(y[0,0]*y[1,1]-y[0,1]**2, 1e-30))*bg
            cs[b] = [-x[0,1]*bg/xne, x[0,0]*bg/xne, xne,
                     -y[0,1]*bg/yne, y[0,0]*bg/yne, yne]
        out.append(cs)
    return out


def _logdet(ens):
    if ens.shape[0] < 7: return np.nan
    s, ld = np.linalg.slogdet(np.cov(ens.T) + 1e-30*np.eye(6))
    return float(ld) if s > 0 else np.nan


def _mmd(a, b):
    try:
        from BPMQ.utils import calculate_MMD4D
        return float(calculate_MMD4D(np.asarray(a,float), np.asarray(b,float)))
    except Exception:
        a, b = np.asarray(a,float), np.asarray(b,float)
        return float(np.linalg.norm((a-b)/(np.abs(a)+np.abs(b)+1e-30)))


def plot_convergence(data: dict,
                     figsize=(15, 8),
                     cs_ref: Optional[np.ndarray] = None,
                     n_qScan: Optional[int] = None) -> plt.Figure:
    """
    Parameters
    ----------
    data     : dict from bpmQscan.get_data()
    figsize  : (width, height) in inches
    cs_ref   : shape (6,) ground-truth CS array (virtual runs only)
    n_qScan  : override the AL budget (needed if your BAL4BPMQ version
               does not store n_qScan in get_data(); pass the notebook
               variable directly)
    """
    # beam beta-gamma
    try:
        from BPMQ.utils import calculate_betagamma
        bg = calculate_betagamma(data.get('E_MeV_u',130), data.get('mass_number',18))
    except Exception:
        bg = 0.5

    # ensemble history
    ens = data.get('cs_ensemble_history') or []
    if not ens:
        covs = data.get('reconstructed_covs_history', [])
        if not covs: raise ValueError("No CS history found.")
        ens = _ens_from_covs(covs, bg)
    N = len(ens)

    # robust label building:
    # initialize() -> always 1 history entry; final fit -> 1 entry; rest are AL
    n_al = max(N - 2, 0)
    iters = np.arange(N)
    labels = (["init"]
              + [f"AL {k+1}" for k in range(n_al)]
              + (["final"] if N >= 2 else []))[:N]

    # budget from data OR caller override OR fallback
    n_init_meas = data.get('n_init', '?')
    # n_qScan: prefer caller arg, then data dict (v4 BAL4BPMQ), then n_al
    n_budget = (n_qScan
                or data.get('n_qScan', None)
                or n_al)

    # CS statistics
    mu  = np.array([e.mean(0) for e in ens])   # (N, 6)
    sig = np.array([e.std(0)  for e in ens])

    # log det posterior volume
    ld = np.array(data.get('log_det_cs_history') or [_logdet(e) for e in ens], float)

    # MMD4D change rate
    mmd_ch = data.get('mmd4d_change_history') or []
    if not mmd_ch:
        ch = data.get('reconstructed_cs_history', [])
        if len(ch) >= 2:
            mmd_ch = [_mmd(ch[k-1], ch[k]) for k in range(1, len(ch))]

    # discriminability (recorded only when step() is used)
    disc = np.array(data.get('discriminability_history') or [], float)
    has_disc = len(disc) > 0

    # cs_ref
    if cs_ref is None: cs_ref = data.get('cs_ref', None)
    if cs_ref is not None: cs_ref = np.asarray(cs_ref, float)

    # metadata for title
    E  = data.get('E_MeV_u', '?')
    A  = data.get('mass_number', '?')
    quad_lbl = ', '.join(q.split(':')[-1] for q in data.get('quads_to_scan', []))

    # colour conventions
    CX, CY = '#1f6fbf', '#c0392b'
    FX, FY = '#aec7e8', '#f4a58a'

    # figure
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, hspace=0.54, wspace=0.42,
                            height_ratios=[1, 0.9])
    ax_a  = fig.add_subplot(gs[0, 0])
    ax_b  = fig.add_subplot(gs[0, 1])
    ax_e  = fig.add_subplot(gs[0, 2])
    ax_ld = fig.add_subplot(gs[1, 0])
    ax_dc = fig.add_subplot(gs[1, 1])
    ax_su = fig.add_subplot(gs[1, 2])

    # shared vertical lines
    def vl(ax):
        ax.axvline(0.5, color='#888', lw=0.9, ls='--', alpha=0.45)
        if N >= 2:
            ax.axvline(N - 1.5, color='#c00', lw=0.8, ls=':', alpha=0.40)

    def sty(ax, title, ylabel):
        ax.set_xticks(iters)
        ax.set_xticklabels(labels, rotation=38, ha='right', fontsize=7)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.22)

    # Row 0: three CS panels
    def cs_panel(ax, jx, jy, scale, title, ylabel):
        vl(ax)
        mx, sx = mu[:, jx]*scale, sig[:, jx]*scale
        my, sy = mu[:, jy]*scale, sig[:, jy]*scale
        ax.plot(iters, mx, 'o-', color=CX, ms=4, lw=1.6, label='x')
        ax.fill_between(iters, mx-sx, mx+sx, color=FX, alpha=0.35)
        ax.plot(iters, my, 's-', color=CY, ms=4, lw=1.6, label='y')
        ax.fill_between(iters, my-sy, my+sy, color=FY, alpha=0.35)
        if cs_ref is not None:
            ax.axhline(cs_ref[jx]*scale, color=CX, lw=0.9, ls='--', alpha=0.65)
            ax.axhline(cs_ref[jy]*scale, color=CY, lw=0.9, ls='--', alpha=0.65)
            ax.text(0.98, 0.03, 'dashed = truth',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=6.5, color='grey')
        ax.legend(fontsize=7, loc='best')
        sty(ax, title, ylabel)

    cs_panel(ax_a, 0, 3, 1.0,  r"$\alpha_{x,y}$  (±1σ)",          r"$\alpha$")
    cs_panel(ax_b, 1, 4, 1.0,  r"$\beta_{x,y}$  (±1σ)",           r"$\beta$  (m/rad)")
    cs_panel(ax_e, 2, 5, 1e6,  r"$\varepsilon_{n,x/y}$  (±1σ)",   r"$\varepsilon_n$  (µm)")

    # Row 1, panel 0: posterior volume
    vl(ax_ld)
    ax_ld.plot(iters, ld, 's-', color='#6a3d9a', ms=5.5, lw=1.8,
               label=r'$\log\det\,\Sigma_{CS}$')
    if N > 1:
        dld = np.diff(ld)
        ax2 = ax_ld.twinx()
        ax2.bar(iters[1:], dld, color='#cab2d6', alpha=0.45, width=0.38,
                label=r'$\Delta\log\det$')
        ax2.axhline(0, color='#6a3d9a', lw=0.5, ls=':')
        ax2.set_ylabel(r"$\Delta\log\det$ / step", color='#6a3d9a', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='#6a3d9a', labelsize=7)
        ax2.legend(fontsize=7, loc='lower left')
    ax_ld.legend(fontsize=8, loc='upper right')
    sty(ax_ld, "Posterior volume\n(lower = tighter)", r"$\log\det\,\Sigma_{CS}$")

    # Row 1, panel 1: discriminability + MMD4D
    vl(ax_dc)
    ax_dc2 = ax_dc.twinx()

    if has_disc:
        # discriminability[k] belongs to AL step k+1 → history index k+1
        di = np.arange(1, 1 + len(disc))
        ax_dc.bar(di, disc, color='#1f78b4', alpha=0.55, width=0.40,
                  label='BPMQ discrim. (mm²)')
        ax_dc.axhline(0.25, color='#1f78b4', lw=1.1, ls='--', alpha=0.75,
                      label='early-stop floor (0.25 mm²)')
        # annotate early-stop if it fired
        below = np.where(disc <= 0.25)[0]
        if len(below) > 0:
            stop_iter = di[below[0]]
            ax_dc.annotate(f'early stop\n(AL {below[0]+1})',
                           xy=(stop_iter, disc[below[0]]),
                           xytext=(stop_iter + 0.3, 0.30),
                           fontsize=7, color='#1f78b4',
                           arrowprops=dict(arrowstyle='->', color='#1f78b4', lw=0.8))
        ax_dc.set_ylabel("BPMQ discriminability (mm²)", fontsize=8)
    else:
        ax_dc.text(0.5, 0.5,
                   'discriminability not recorded\n(use step() instead of\nexplicit loop)',
                   transform=ax_dc.transAxes, ha='center', va='center',
                   fontsize=8, color='grey', style='italic')
        ax_dc.set_yticks([])

    if mmd_ch:
        ma = np.array(mmd_ch)
        mi = np.arange(1, 1 + len(ma))
        ax_dc2.plot(mi, ma, 'd--', color='#e31a1c', ms=5.5, lw=1.4,
                    label='MMD4D(k, k-1)')
        ax_dc2.set_ylabel("CS change rate (MMD4D)", color='#e31a1c', fontsize=8)
        ax_dc2.tick_params(axis='y', labelcolor='#e31a1c', labelsize=7)
        ax_dc2.legend(fontsize=7, loc='upper right')

    la, na = ax_dc.get_legend_handles_labels()
    ax_dc.legend(la, na, fontsize=7, loc='upper left')
    sty(ax_dc, "Convergence signals", "BPMQ discriminability (mm²)")

    # Row 1, panel 2: text summary
    ax_su.axis('off')
    fm, fs = mu[-1], sig[-1]

    early_stopped = has_disc and n_al < n_budget
    status = (f"Early stop at AL {n_al}\n  (discrim. < 0.25 mm2)"
              if early_stopped else
              f"Full budget used ({n_al}/{n_budget})")

    lines = [
        f"n_init meas  : {n_init_meas}",
        f"AL steps run : {n_al} / {n_budget}",
        "",
        "Final CS (mean +/- 1sigma)",
        f"  ax  {fm[0]:+.3f} +/- {fs[0]:.3f}",
        f"  bx  {fm[1]:.3f} +/- {fs[1]:.3f}  m/rad",
        f"  exn {fm[2]*1e6:.3f} +/- {fs[2]*1e6:.3f}  um",
        f"  ay  {fm[3]:+.3f} +/- {fs[3]:.3f}",
        f"  by  {fm[4]:.3f} +/- {fs[4]:.3f}  m/rad",
        f"  eyn {fm[5]*1e6:.3f} +/- {fs[5]*1e6:.3f}  um",
    ]
    if cs_ref is not None:
        lines += ["", f"MMD4D vs truth : {_mmd(cs_ref, fm):.5f}"]
    lines += ["", status]

    ax_su.text(0.05, 0.96, '\n'.join(lines),
               transform=ax_su.transAxes, va='top', ha='left',
               fontsize=8, family='monospace',
               bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='#f8f8f8', edgecolor='#ccc', alpha=0.9))

    fig.suptitle(
        f"CS inference convergence  --  {A}X  {E} MeV/u  |  quads: {quad_lbl}",
        fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_convergence_from_pkl.py <file.pkl> [out.png]")
        sys.exit(1)
    pkl_path = Path(sys.argv[1])
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    fig = plot_convergence(data)
    out = (Path(sys.argv[2]) if len(sys.argv) > 2
           else pkl_path.with_suffix('.convergence.png'))
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out}")
    plt.show()
