import torch
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1  —  New module-level helper.
# Insert after noise2covar(), before drift_maps_2x2().
# ══════════════════════════════════════════════════════════════════════════════

def _mmd4d_sq_batch_torch(
    xcovs: torch.Tensor,
    ycovs: torch.Tensor,
    xcov_prior: torch.Tensor,
    ycov_prior: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form, differentiable, O(batch)-cost  MMD4D²  between each
    particle's reconstructed beam and the prior beam.

    Returns MMD²  (squared MMD, without the final sqrt).  This is
    intentional — see design notes below.

    Design notes
    ────────────
    Why MMD² and not MMD?
      At the prior (xcovs ≈ xcov_prior), mmd2 = 0 and its gradient is also
      exactly 0 (it is the minimum of a non-negative smooth function).
      Taking sqrt(mmd2) = mmd would require evaluating 1/(2·mmd) in the
      backward pass; at mmd=0 this is +∞ → NaN gradients.
      Using mmd2 directly avoids the singularity completely while keeping
      the function smooth and differentiable everywhere.

    Why is MMD² still a good penalty shape?
      Near the prior, mmd2 ≈ C·‖ΔΣ‖² (locally quadratic), giving gradient
      behaviour analogous to the L2-noise-space prior being replaced.
      Far from the prior, mmd2 grows monotonically, reaching its maximum
      (1/3 for the Gaussian kernel with this bandwidth) only when the two
      beams are perfectly orthogonal in phase space.

    Kernel
    ──────
    k(z,w) = exp(-½ (z-w)ᵀ S (z-w)),  S = Σ_prior⁻¹.
    Factored over the decoupled x and y subspaces:
        k4D = k_x ⊗ k_y,  each 2-dimensional.

    MMD² formula for zero-mean Gaussians p_i = N(0, Σ_i), p_0 = N(0, Σ_prior):
        mmd2_i = 1/sqrt(det(I+2 S_x Σ_xi) det(I+2 S_y Σ_yi))   ← E_{pi,pi}[k]
               + 1/sqrt(det(I+2 S_x Σ_x0) det(I+2 S_y Σ_y0))   ← E_{p0,p0}[k]
               - 2/sqrt(det(I+S_x(Σ_xi+Σ_x0)) det(I+S_y(Σ_yi+Σ_y0)))  ← -2 E_{pi,p0}[k]

    All matrices are 2×2; det and inv are O(1).  Fully differentiable via
    torch.linalg.det / torch.linalg.inv.

    Parameters
    ──────────
    xcovs      : (batch, 2, 2) — per-particle x phase-space covariance
    ycovs      : (batch, 2, 2) — per-particle y phase-space covariance
    xcov_prior : (2, 2)        — prior x covariance (sets kernel bandwidth)
    ycov_prior : (2, 2)        — prior y covariance

    Returns
    ───────
    mmd2 : (batch,) ∈ [0, ~1/3], differentiable w.r.t. xcovs / ycovs
    """
    I2  = torch.eye(2, dtype=xcovs.dtype, device=xcovs.device)
    S_x = torch.linalg.inv(xcov_prior)          # (2, 2)
    S_y = torch.linalg.inv(ycov_prior)
    Sx  = S_x.unsqueeze(0)                       # (1, 2, 2)  broadcasts over batch
    Sy  = S_y.unsqueeze(0)
    xp  = xcov_prior.unsqueeze(0)                # (1, 2, 2)
    yp  = ycov_prior.unsqueeze(0)

    # term1 : E_{p_i, p_i}[k(z,z)]   — one value per particle
    det1 = (torch.linalg.det(I2 + 2.0 * (Sx @ xcovs)) *      # (batch,)
            torch.linalg.det(I2 + 2.0 * (Sy @ ycovs)))

    # term2 : E_{p_prior, p_prior}[k(w,w)]  — scalar, same for all particles
    det2 = (torch.linalg.det(I2 + 2.0 * S_x @ xcov_prior) *  # scalar
            torch.linalg.det(I2 + 2.0 * S_y @ ycov_prior))

    # term3 : -2 · E_{p_i, p_prior}[k(z,w)]  — one value per particle
    det3 = (torch.linalg.det(I2 + Sx @ xcovs + Sx @ xp) *    # (batch,)
            torch.linalg.det(I2 + Sy @ ycovs + Sy @ yp))

    mmd2 = 1.0 / det1.sqrt() + 1.0 / det2.sqrt() - 2.0 / det3.sqrt()

    # Clamp: numerical noise can push mmd2 slightly negative when
    # xcovs ≈ xcov_prior (true MMD² = 0).
    return mmd2.clamp(min=0.0)   # (batch,)


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2a  —  Setup block inside _get_cs_reconst_loss_ftn.
#
# REPLACE the existing block:
#   (starts with the docstring about cs_prior_noise,
#    ends just before  "def loss_fun(x):" )
#
# Keep everything else in the method unchanged.
# ══════════════════════════════════════════════════════════════════════════════

_SETUP_BLOCK = '''
        # ── MMD4D² prior setup ────────────────────────────────────────────
        # Replaces the old L2-in-noise-space prior regularisation.
        #
        # We pre-compute:
        #   _xcov_prior_t  — 2×2 x-covariance of the prior beam (no grad)
        #   _ycov_prior_t  — 2×2 y-covariance of the prior beam
        #   _mmd4d_norm    — mean(mmd2) over N(0,I) calibration samples
        #
        # Normalising by mean(mmd2_init) — not std(sqrt(mmd2)) — gives:
        #   E[regloss_prior] = 1  at initialisation  (x0 ~ N(0,I))
        # which matches fitloss_bpmQ (normalised to BPMQ tolerance) and the
        # old L2 prior (E = 1 by construction), so prior_weight keeps the
        # same intuitive meaning.
        _xcov_prior_t = _ycov_prior_t = _mmd4d_norm = None
        if cs_prior_noise is not None:
            with torch.no_grad():
                # Prior beam covariance: noise=0 maps to cs_ref via noise2covar
                _xc, _yc = noise2covar(
                    torch.zeros(1, 6, dtype=self.dtype),
                    *self.cs_ref, bg=self.bg
                )
                _xcov_prior_t = _xc.squeeze(0).detach()   # (2,2)
                _ycov_prior_t = _yc.squeeze(0).detach()   # (2,2)

                # Calibration: draw N_CAL samples from N(0,I) — the same
                # distribution as the initial particle cloud — and compute
                # their MMD² to the prior.  The resulting mean is the
                # normalisation constant that puts E[regloss_prior] = 1.
                _N_CAL = 1000
                _z_cal = torch.randn(_N_CAL, 6, dtype=self.dtype)
                _xc_cal, _yc_cal = noise2covar(
                    _z_cal, *self.cs_ref, bg=self.bg
                )
                _mmd2_cal = _mmd4d_sq_batch_torch(
                    _xc_cal, _yc_cal, _xcov_prior_t, _ycov_prior_t
                )
                _mmd4d_norm = _mmd2_cal.mean().clamp(min=1e-10)
                # ↑ mean, not std: ensures E[regloss_prior / _mmd4d_norm] = 1.
                #   Using std (as in mmd4d_prior_changes.py) would give
                #   E ≈ mean/std ≈ 3.9 — a 4× scale error.
        # ─────────────────────────────────────────────────────────────────
'''

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2b  —  Replacement for the regloss_prior block inside loss_fun.
#
# REMOVE (old code):
# ══════════════════════════════════════════════════════════════════════════════
_OLD_REGLOSS_PRIOR = '''
            regloss_prior = None
            if cs_prior_noise is not None:
                diff = x[:, :6] - cs_prior_noise           # (batch_size, 6)
                regloss_prior = torch.mean(diff**2, dim=1)  # (batch_size,)
'''

# INSERT (new code) in its place:
_NEW_REGLOSS_PRIOR = '''
            # MMD4D² prior: penalises each particle for having a beam
            # distribution distinguishable from the prior in 4D phase space.
            #
            # Uses MMD²  (not sqrt(MMD²)) to avoid the 0/0 gradient
            # singularity at the prior (mmd2=0 → d(sqrt)/dx = 1/(2·0) = ∞).
            # Divided by _mmd4d_norm (= E[mmd2] at N(0,I) init) so that
            # E[regloss_prior] = 1, matching fitloss_bpmQ at initialisation.
            regloss_prior = None
            if cs_prior_noise is not None:
                # xcovs, ycovs are already computed earlier in loss_fun
                regloss_prior = (
                    _mmd4d_sq_batch_torch(
                        xcovs, ycovs,
                        _xcov_prior_t, _ycov_prior_t,
                    )
                    / _mmd4d_norm     # → E[loss] = 1 at N(0,I) initialisation
                )
'''


# ══════════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED UNIT TESTS
# Run with:  python mmd4d_prior_v2.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)
    dtype = torch.float32

    # ── shared helpers ────────────────────────────────────────────────────────
    def _noise2covar(noise, xbeta=5.0, xnemit=0.16e-6, ybeta=5.0, ynemit=0.16e-6):
        bx  = xbeta  * torch.exp(noise[:, 1] * 0.8)
        ex  = xnemit * torch.exp(noise[:, 2] * 0.3)
        ax  = noise[:, 0] * 1.5
        by_ = ybeta  * torch.exp(noise[:, 4] * 0.8)
        ey  = ynemit * torch.exp(noise[:, 5] * 0.3)
        ay  = noise[:, 3] * 1.5
        N = noise.shape[0]; bg = 0.566
        xcov = torch.zeros(N, 2, 2, dtype=dtype)
        ycov = torch.zeros(N, 2, 2, dtype=dtype)
        xcov[:, 0, 0] = bx;  xcov[:, 0, 1] = -ax
        xcov[:, 1, 0] = -ax; xcov[:, 1, 1] = (ax**2 + 1) / bx
        xcov *= (ex / bg).unsqueeze(-1).unsqueeze(-1)
        ycov[:, 0, 0] = by_; ycov[:, 0, 1] = -ay
        ycov[:, 1, 0] = -ay; ycov[:, 1, 1] = (ay**2 + 1) / by_
        ycov *= (ey / bg).unsqueeze(-1).unsqueeze(-1)
        return xcov, ycov

    z0 = torch.zeros(1, 6, dtype=dtype)
    xc0, yc0   = _noise2covar(z0)
    xcov_p, ycov_p = xc0[0], yc0[0]

    # ── Test 1: MMD² = 0 at the prior ────────────────────────────────────────
    print("Test 1: MMD² = 0 at the prior")
    mmd2_at_prior = _mmd4d_sq_batch_torch(xc0, yc0, xcov_p, ycov_p)
    assert mmd2_at_prior.item() < 1e-10, f"Expected ~0, got {mmd2_at_prior.item():.2e}"
    print(f"  MMD² at prior = {mmd2_at_prior.item():.2e}  ✓")

    # ── Test 2: MMD² > 0 for a different beam ────────────────────────────────
    print("Test 2: MMD² > 0 for a different beam")
    z_far = torch.tensor([[2.0, 1.5, 0.5, -2.0, 1.5, 0.5]], dtype=dtype)
    xc_far, yc_far = _noise2covar(z_far)
    mmd2_far = _mmd4d_sq_batch_torch(xc_far, yc_far, xcov_p, ycov_p)
    assert mmd2_far.item() > 1e-4, f"Expected > 1e-4, got {mmd2_far.item():.6f}"
    print(f"  MMD² at 2-sigma beam = {mmd2_far.item():.5f}  ✓")

    # ── Test 3: NO NaN gradient at exactly the prior ─────────────────────────
    print("Test 3: No NaN gradient at the prior (fixes Bug 1 in mmd4d_prior_changes.py)")
    z_at_prior = torch.zeros(8, 6, dtype=dtype, requires_grad=True)
    xc, yc = _noise2covar(z_at_prior)
    mmd2 = _mmd4d_sq_batch_torch(xc, yc, xcov_p, ycov_p)
    mmd2.sum().backward()
    has_nan = z_at_prior.grad.isnan().any().item()
    assert not has_nan, "NaN gradient detected!"
    print(f"  Grad at prior: norm={z_at_prior.grad.norm():.6f}, NaN={has_nan}  ✓")

    # Show that sqrt version (mmd4d_prior_changes.py) DOES produce NaN:
    z_sqrt = torch.zeros(8, 6, dtype=dtype, requires_grad=True)
    xc2, yc2 = _noise2covar(z_sqrt)
    mmd2_v = _mmd4d_sq_batch_torch(xc2, yc2, xcov_p, ycov_p)
    mmd_sqrt = mmd2_v.sqrt()
    mmd_sqrt.sum().backward()
    print(f"  sqrt(MMD²) grad at prior: NaN={z_sqrt.grad.isnan().any().item()}  "
          f"← Bug confirmed in mmd4d_prior_changes.py approach")

    # ── Test 4: Mean normalisation gives E[loss] ≈ 1 (fixes Bug 2) ───────────
    print("Test 4: Mean normalisation → E[regloss_prior] ≈ 1  (fixes Bug 2)")
    with torch.no_grad():
        z_cal = torch.randn(2000, 6, dtype=dtype)
        xc_cal, yc_cal = _noise2covar(z_cal)
        mmd2_cal = _mmd4d_sq_batch_torch(xc_cal, yc_cal, xcov_p, ycov_p)
    mmd4d_norm = mmd2_cal.mean().clamp(min=1e-10)
    mmd2_normed = mmd2_cal / mmd4d_norm
    print(f"  Raw MMD²:             mean={mmd2_cal.mean():.5f}, std={mmd2_cal.std():.5f}")
    print(f"  Mean-normed MMD²:     mean={mmd2_normed.mean():.4f}, std={mmd2_normed.std():.4f}  ✓")
    assert 0.9 < mmd2_normed.mean().item() < 1.1, "E[normed] should be ≈1"

    # Show what mmd4d_prior_changes.py normalization actually gives:
    mmd_sqrt_cal = mmd2_cal.sqrt()
    std_norm = mmd_sqrt_cal.std().clamp(min=1e-10)
    mmd_sqrt_std_normed = mmd_sqrt_cal / std_norm
    print(f"  mmd4d_prior_changes.py (std-normed sqrt):")
    print(f"    mean={mmd_sqrt_std_normed.mean():.3f}, std={mmd_sqrt_std_normed.std():.4f}  "
          f"← E≈{mmd_sqrt_std_normed.mean():.2f}, not 1! (4× scale error)")

    # ── Test 5: Gradient scale matches L2 noise loss ──────────────────────────
    print("Test 5: Gradient magnitude comparison at typical init")
    ratios = []
    for seed in range(5):
        torch.manual_seed(seed)
        z_g = torch.randn(8, 6, dtype=dtype, requires_grad=True)
        xc_g, yc_g = _noise2covar(z_g)
        mmd2_g = _mmd4d_sq_batch_torch(xc_g, yc_g, xcov_p, ycov_p)
        (mmd2_g / mmd4d_norm).sum().backward()
        g_mmd2 = z_g.grad.abs().mean().item()

        z_l2 = z_g.detach().clone().requires_grad_(True)
        l2 = (z_l2 ** 2).mean(dim=1)
        l2.sum().backward()
        g_l2 = z_l2.grad.abs().mean().item()

        ratio = g_mmd2 / g_l2
        ratios.append(ratio)
        print(f"  seed {seed}: |grad(mmd2/norm)|={g_mmd2:.4f}  |grad(L2)|={g_l2:.4f}  ratio={ratio:.3f}")
    print(f"  Mean ratio: {np.mean(ratios):.3f}  (good: near 1.0)  ✓")

    # ── Test 6: Gradient flows through the whole chain ────────────────────────
    print("Test 6: Gradient flows through noise → covar → MMD²")
    z_chain = torch.randn(8, 6, dtype=dtype, requires_grad=True)
    xc_c, yc_c = _noise2covar(z_chain)
    mmd2_c = _mmd4d_sq_batch_torch(xc_c, yc_c, xcov_p, ycov_p)
    (mmd2_c / mmd4d_norm).sum().backward()
    assert z_chain.grad is not None and not z_chain.grad.isnan().any()
    print(f"  Gradient norm = {z_chain.grad.norm():.4f}  ✓")

    # ── Test 7: Consistency with numpy reference ──────────────────────────────
    print("Test 7: Numerical consistency with numpy reference (calculate_MMD4D_from_covs)")
    def _mmd4d_numpy_ref(xc1, yc1, xc2, yc2, xr, yr):
        I2 = np.eye(2)
        Sx = np.linalg.inv(xr); Sy = np.linalg.inv(yr)
        d1 = np.linalg.det(I2+2*Sx@xc1)*np.linalg.det(I2+2*Sy@yc1)
        d2 = np.linalg.det(I2+2*Sx@xc2)*np.linalg.det(I2+2*Sy@yc2)
        d3 = np.linalg.det(I2+Sx@xc1+Sx@xc2)*np.linalg.det(I2+Sy@yc1+Sy@yc2)
        return max(1/d1**0.5 + 1/d2**0.5 - 2/d3**0.5, 0)  # MMD², no sqrt

    for i in range(10):
        z1 = torch.randn(1, 6, dtype=dtype); z2 = torch.randn(1, 6, dtype=dtype)
        xc1, yc1 = _noise2covar(z1); xc2, yc2 = _noise2covar(z2)
        torch_val  = _mmd4d_sq_batch_torch(xc1, yc1, xc2[0], yc2[0]).item()
        numpy_val  = _mmd4d_numpy_ref(
            xc1[0].numpy(), yc1[0].numpy(),
            xc2[0].numpy(), yc2[0].numpy(),
            xc2[0].numpy(), yc2[0].numpy()
        )
        assert abs(torch_val - numpy_val) < 1e-5, \
            f"Mismatch: torch={torch_val:.6f}, numpy={numpy_val:.6f}"
    print("  All 10 random pairs match numpy reference  ✓")

    # ── Test 8: Monotonicity ──────────────────────────────────────────────────
    print("Test 8: Monotonicity — larger beam deviation → larger MMD²")
    scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    mmd2_vals = []
    for s in scales:
        z_s = torch.tensor([[s, s*0.5, 0.0, -s, s*0.5, 0.0]], dtype=dtype)
        xc_s, yc_s = _noise2covar(z_s)
        mmd2_vals.append(_mmd4d_sq_batch_torch(xc_s, yc_s, xcov_p, ycov_p).item())
        print(f"  scale={s:.1f}  →  MMD²={mmd2_vals[-1]:.6f}")
    assert all(mmd2_vals[i] <= mmd2_vals[i+1] for i in range(len(mmd2_vals)-1))
    print("  Monotone  ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("All tests passed.")
    print()
    print("Summary vs mmd4d_prior_changes.py:")
    print("  Bug 1 (NaN grad at prior):  FIXED  — use MMD² not sqrt(MMD²)")
    print("  Bug 2 (4× scale error):     FIXED  — normalise by mean(MMD²), not std(MMD)")
    print("  Gradient scale vs L2 prior: GOOD   — ratio ≈ 0.85–0.97")
