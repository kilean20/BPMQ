"""
Two targeted changes to EnvelopeEnsembleModel in envelope_model.py
===================================================================

CHANGE 1: Add _mmd4d_batch_torch after the existing cs/covar helper functions
          (i.e., after covar2cs, before drift_maps_2x2)

CHANGE 2: Replace the L2-in-noise-space regloss_prior in _get_cs_reconst_loss_ftn
          with the MMD4D-in-phase-space version.

Design rationale
----------------
Current regloss_prior = mean((noise - 0)^2, dim=1)
  - Operates in noise-parameterisation space, which is arbitrary
  - Penalises alpha/beta/emittance deviations equally by noise coordinate,
    not by physical beam distinguishability

New regloss_prior = MMD4D(xcov_i, ycov_i | xcov_prior, ycov_prior) / mmd4d_norm
  - Operates in beam phase-space covariance: physically meaningful,
    parameterisation-independent
  - Uses the same closed-form Gaussian-kernel MMD used as the convergence
    metric elsewhere, so prior penalty and convergence tracking are in the
    same space
  - O(batch) cost — each particle's MMD to the prior is one set of 2×2
    determinants (no pairwise N² computation needed)
  - Fully differentiable via torch.linalg.det / inv

Normalisation
-------------
The normalisation constant mmd4d_norm = std(MMD4D(z)) where z ~ N(0,I)
(500 samples, computed once at loss-function construction time, no-grad).

Why this normalisation?
  x0 = torch.randn(...) — the initial particle cloud — is exactly N(0,I).
  So mmd4d_norm is the std of regloss_prior over the actual initial distribution
  of particles. After dividing, regloss_prior has std≈1 at initialisation,
  which matches the O(1) scale of fitloss_bpmQ (set by the BPMQ tolerance
  normalisation). This makes prior_weight directly interpretable:
    prior_weight=1  →  prior and data-fit contribute equally in gradient scale
    prior_weight=0  →  no prior (original behaviour)
"""

import torch
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — New module-level function.
# Insert after covar2cs(), before drift_maps_2x2().
# ══════════════════════════════════════════════════════════════════════════════

def _mmd4d_batch_torch(
    xcovs: torch.Tensor,
    ycovs: torch.Tensor,
    xcov_prior: torch.Tensor,
    ycov_prior: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form, differentiable, O(batch)-cost MMD4D between each particle's
    reconstructed beam and the prior beam.

    Uses the Gaussian-kernel MMD identity for zero-mean Gaussian distributions
    in 4D beam phase space (factored as x-subspace ⊗ y-subspace):

        MMD²(p_i, p_prior) = E_{p_i}[k(z,z)]
                            - 2·E_{p_i, p_prior}[k(z,w)]
                            + E_{p_prior}[k(w,w)]

    with k(z,w) = exp(-½ zᵀ S z), S = Σ_prior⁻¹  (Gaussian kernel).

    For Gaussian distributions the expectations have closed-form determinant
    expressions (see calculate_MMD4D_from_covs).  Everything is O(1) per
    particle in terms of matrix size (all matrices are 2×2).

    Args
    ----
    xcovs      : (batch, 2, 2) — per-particle x phase-space covariance
    ycovs      : (batch, 2, 2) — per-particle y phase-space covariance
    xcov_prior : (2, 2)        — prior x covariance (also sets kernel bandwidth)
    ycov_prior : (2, 2)        — prior y covariance

    Returns
    -------
    mmd : (batch,) ∈ [0, ~1/3], differentiable w.r.t. xcovs / ycovs
          (and therefore w.r.t. x[:,:6] through noise2covar)
    """
    I2  = torch.eye(2, dtype=xcovs.dtype, device=xcovs.device)
    S_x = torch.linalg.inv(xcov_prior)          # (2,2)
    S_y = torch.linalg.inv(ycov_prior)
    Sx  = S_x.unsqueeze(0)                       # (1,2,2) — broadcasts over batch
    Sy  = S_y.unsqueeze(0)
    xp  = xcov_prior.unsqueeze(0)               # (1,2,2)
    yp  = ycov_prior.unsqueeze(0)

    # ── term 1 : E_{p_i}[k(z,z)]  (one value per particle) ──────────────────
    det1 = (torch.linalg.det(I2 + 2.0 * (Sx @ xcovs)) *    # (batch,)
            torch.linalg.det(I2 + 2.0 * (Sy @ ycovs)))

    # ── term 2 : E_{p_prior}[k(w,w)]  (scalar, same for all particles) ───────
    det2 = (torch.linalg.det(I2 + 2.0 * S_x @ xcov_prior) *  # scalar
            torch.linalg.det(I2 + 2.0 * S_y @ ycov_prior))

    # ── term 3 : -2·E_{p_i, p_prior}[k(z,w)]  (one value per particle) ──────
    det3 = (torch.linalg.det(I2 + Sx @ xcovs + Sx @ xp) *  # (batch,)
            torch.linalg.det(I2 + Sy @ ycovs + Sy @ yp))

    mmd2 = 1.0 / det1.sqrt() + 1.0 / det2.sqrt() - 2.0 / det3.sqrt()

    # Clamp before sqrt: numerical noise can push mmd2 slightly negative
    # when xcov_i ≈ xcov_prior (true MMD = 0).
    return mmd2.clamp(min=0.0).sqrt()   # (batch,)


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — Modified _get_cs_reconst_loss_ftn (only the changed sections).
#
# In the full file, replace:
#   (a) the block that starts with the comment about cs_prior_noise and ends
#       before "def loss_fun(x):"        → insert the MMD4D setup block below
#   (b) the regloss_prior computation inside loss_fun  → replace with the
#       MMD4D version below
# ══════════════════════════════════════════════════════════════════════════════

# ── (a) MMD4D setup block — insert right before  "def loss_fun(x):" ──────────
#
#   (Everything before this in _get_cs_reconst_loss_ftn stays unchanged.)

_SETUP_BLOCK = '''
        # ── MMD4D prior setup ──────────────────────────────────────────────
        # Replaces the old L2-in-noise-space prior.
        # We compute:
        #   xcov_prior_t, ycov_prior_t : the prior beam covariance (noise=0)
        #   _mmd4d_norm                : std of MMD4D when particles ~ N(0,I)
        #
        # _mmd4d_norm calibrates regloss_prior to std≈1 at initialisation
        # (x0=randn ~ N(0,I)), matching the O(1) scale of fitloss_bpmQ that
        # comes from the BPMQ tolerance normalisation.  With this, prior_weight
        # has a consistent meaning across different datasets and prior choices.
        _xcov_prior_t = _ycov_prior_t = _mmd4d_norm = None
        if cs_prior_noise is not None:
            with torch.no_grad():
                # Prior beam: noise=0 maps to cs_ref via noise2covar
                _xc, _yc = noise2covar(
                    torch.zeros(1, 6, dtype=self.dtype),
                    *self.cs_ref, bg=self.bg
                )
                _xcov_prior_t = _xc.squeeze(0).detach()   # (2,2), no grad
                _ycov_prior_t = _yc.squeeze(0).detach()

                # Calibration: draw 500 particles from N(0,I) — the same
                # distribution as x0 — and compute their MMD4D to the prior.
                # Normalising by the resulting std makes the loss scale-invariant
                # and comparable to fitloss_bpmQ at initialisation.
                _z_cal = torch.randn(500, 6, dtype=self.dtype)
                _xc_cal, _yc_cal = noise2covar(
                    _z_cal, *self.cs_ref, bg=self.bg
                )
                _mmd_cal  = _mmd4d_batch_torch(
                    _xc_cal, _yc_cal, _xcov_prior_t, _ycov_prior_t
                )
                _mmd4d_norm = _mmd_cal.std().clamp(min=1e-8)
        # ──────────────────────────────────────────────────────────────────────
'''


# ── (b) Replacement for the regloss_prior block inside loss_fun ───────────────
#
# REMOVE this (old code):
_OLD_REGLOSS_PRIOR = '''
            regloss_prior = None
            if cs_prior_noise is not None:
                diff = x[:, :6] - cs_prior_noise           # (batch_size, 6)
                regloss_prior = torch.mean(diff**2, dim=1)  # (batch_size,)
'''

# INSERT this (new code) in its place:
_NEW_REGLOSS_PRIOR = '''
            # MMD4D prior: penalises each particle for having a beam distribution
            # that is distinguishable from the prior beam in 4D phase space.
            # Normalised by _mmd4d_norm so that std(regloss_prior) ≈ 1
            # at initialisation (x0 ~ N(0,I)), matching fitloss_bpmQ scale.
            regloss_prior = None
            if cs_prior_noise is not None:
                regloss_prior = (
                    _mmd4d_batch_torch(
                        xcovs, ycovs,        # already computed above in loss_fun
                        _xcov_prior_t, _ycov_prior_t,
                    )
                    / _mmd4d_norm            # → std≈1 at N(0,I) initialisation
                )
'''


# ══════════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED UNIT TEST
# Run with:  python mmd4d_prior_changes.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)
    dtype = torch.float32

    # ── helpers (copied from envelope_model for standalone test) ──────────────
    def _noise2covar_simple(noise, xbeta=5.0, ybeta=5.0, xnemit=0.16e-6,
                             ynemit=0.16e-6, bg=0.566):
        """Minimal noise→covar for testing (alpha=0 reference)."""
        # noise[:, 1] → log(beta_x/xbeta)/0.8,  noise[:, 2] → log(emit_x/xnemit)/0.3
        bx  = xbeta  * torch.exp(noise[:, 1] * 0.8)
        ex  = xnemit * torch.exp(noise[:, 2] * 0.3)
        ax  = noise[:, 0] * 1.5
        by_ = ybeta  * torch.exp(noise[:, 4] * 0.8)
        ey  = ynemit * torch.exp(noise[:, 5] * 0.3)
        ay  = noise[:, 3] * 1.5
        N   = noise.shape[0]
        xcov = torch.zeros(N, 2, 2, dtype=dtype)
        ycov = torch.zeros(N, 2, 2, dtype=dtype)
        xcov[:, 0, 0] = bx;  xcov[:, 0, 1] = -ax
        xcov[:, 1, 0] = -ax; xcov[:, 1, 1] = (ax**2 + 1) / bx
        xcov *= (ex / bg).unsqueeze(-1).unsqueeze(-1)
        ycov[:, 0, 0] = by_; ycov[:, 0, 1] = -ay
        ycov[:, 1, 0] = -ay; ycov[:, 1, 1] = (ay**2 + 1) / by_
        ycov *= (ey / bg).unsqueeze(-1).unsqueeze(-1)
        return xcov, ycov

    # ── Test 1: MMD4D = 0 when particle = prior ───────────────────────────────
    print("Test 1: MMD4D = 0 at prior")
    z0 = torch.zeros(1, 6, dtype=dtype)
    xc0, yc0 = _noise2covar_simple(z0)
    xcov_p, ycov_p = xc0[0], yc0[0]
    mmd_at_prior = _mmd4d_batch_torch(xc0, yc0, xcov_p, ycov_p)
    assert mmd_at_prior.item() < 1e-6, f"Expected ~0, got {mmd_at_prior.item():.6f}"
    print(f"  MMD4D at prior = {mmd_at_prior.item():.2e}  ✓")

    # ── Test 2: MMD4D > 0 for a different beam ───────────────────────────────
    print("Test 2: MMD4D > 0 for different beam")
    z_far = torch.tensor([[2.0, 1.5, 0.5, -2.0, 1.5, 0.5]], dtype=dtype)
    xc_far, yc_far = _noise2covar_simple(z_far)
    mmd_far = _mmd4d_batch_torch(xc_far, yc_far, xcov_p, ycov_p)
    assert mmd_far.item() > 0.01, f"Expected >0.01, got {mmd_far.item():.4f}"
    print(f"  MMD4D at 2-sigma = {mmd_far.item():.4f}  ✓")

    # ── Test 3: Normalisation makes std≈1 at N(0,I) ──────────────────────────
    print("Test 3: Normalisation → std≈1 at N(0,I)")
    with torch.no_grad():
        z_cal = torch.randn(500, 6, dtype=dtype)
        xc_cal, yc_cal = _noise2covar_simple(z_cal)
        mmd_cal  = _mmd4d_batch_torch(xc_cal, yc_cal, xcov_p, ycov_p)
        mmd_norm = mmd_cal.std().clamp(min=1e-8)
        mmd_normalised = mmd_cal / mmd_norm
    print(f"  Raw MMD4D      : mean={mmd_cal.mean():.4f}, std={mmd_cal.std():.4f}")
    print(f"  Normalised     : mean={mmd_normalised.mean():.4f}, std={mmd_normalised.std():.4f}  ✓")
    assert 0.8 < mmd_normalised.std().item() < 1.2, "std after normalisation should be ≈1"

    # ── Test 4: Gradient flows through MMD4D → xcovs → noise ─────────────────
    print("Test 4: Gradient flows from MMD4D through noise2covar")
    z_grad = torch.randn(8, 6, dtype=dtype, requires_grad=True)
    xc_g, yc_g = _noise2covar_simple(z_grad)
    mmd_g = _mmd4d_batch_torch(xc_g, yc_g, xcov_p, ycov_p)
    mmd_g.sum().backward()
    assert z_grad.grad is not None, "No gradient!"
    assert not z_grad.grad.isnan().any(), "NaN in gradient!"
    print(f"  Gradient norm = {z_grad.grad.norm():.4f}  ✓")

    # ── Test 5: Consistency with numpy reference (calculate_MMD4D_from_covs) ──
    print("Test 5: Consistency with numpy reference")
    def _mmd4d_numpy_ref(xcov1, ycov1, xcov2, ycov2, xcov_ref, ycov_ref):
        I2 = np.eye(2)
        S_x = np.linalg.inv(xcov_ref); S_y = np.linalg.inv(ycov_ref)
        d1x = np.linalg.det(I2 + 2*S_x@xcov1); d1y = np.linalg.det(I2 + 2*S_y@ycov1)
        d2x = np.linalg.det(I2 + 2*S_x@xcov2); d2y = np.linalg.det(I2 + 2*S_y@ycov2)
        d3x = np.linalg.det(I2 + S_x@xcov1 + S_x@xcov2)
        d3y = np.linalg.det(I2 + S_y@ycov1 + S_y@ycov2)
        t1 = 1/np.sqrt(d1x*d1y); t2 = 1/np.sqrt(d2x*d2y); t3 = -2/np.sqrt(d3x*d3y)
        return max(t1+t2+t3, 0)**0.5

    for _ in range(10):
        z1 = torch.randn(1, 6, dtype=dtype)
        z2 = torch.randn(1, 6, dtype=dtype)
        xc1, yc1 = _noise2covar_simple(z1)
        xc2, yc2 = _noise2covar_simple(z2)
        torch_val = _mmd4d_batch_torch(
            xc1, yc1,
            xc2[0], yc2[0]        # prior = second beam
        ).item()
        numpy_val = _mmd4d_numpy_ref(
            xc1[0].numpy(), yc1[0].numpy(),
            xc2[0].numpy(), yc2[0].numpy(),
            xc2[0].numpy(), yc2[0].numpy()
        )
        assert abs(torch_val - numpy_val) < 1e-5, \
            f"Mismatch: torch={torch_val:.6f}, numpy={numpy_val:.6f}"
    print("  All 10 random pairs match numpy reference  ✓")

    # ── Test 6: Monotonicity — further beam = larger MMD ─────────────────────
    print("Test 6: Monotonicity (larger deviation → larger MMD)")
    scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    mmds = []
    for s in scales:
        z_s = torch.tensor([[s, s*0.5, 0.0, -s, s*0.5, 0.0]], dtype=dtype)
        xc_s, yc_s = _noise2covar_simple(z_s)
        mmds.append(_mmd4d_batch_torch(xc_s, yc_s, xcov_p, ycov_p).item())
        print(f"  scale={s:.1f}  →  MMD4D={mmds[-1]:.4f}")
    assert all(mmds[i] <= mmds[i+1] for i in range(len(mmds)-1)), "Not monotone!"
    print("  Monotone  ✓")

    print("\nAll tests passed.")
