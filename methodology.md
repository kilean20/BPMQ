# REAL Framework: Algorithm and Methodology

**Randomized Ensemble Active Learning for Latent-State Inference in Control Systems**

---

## 1. Overview

REAL is an iterative Bayesian active-learning framework for inferring an unknown latent state $z^\star$ of a dynamical system equipped with a forward model (or digital twin). The system is controlled through an input $x$ drawn from a controllable input space $\mathcal{X}$, and diagnostic devices return observations $y$ from an observation space $\mathcal{Y}$, or alternatively a Virtual Diagnostics (VD) model reads raw signals $o \in \mathcal{O}$ that are not directly modeled by the forward model and maps them to observations $y \in \mathcal{Y}$ together with the uncertainty $\sigma_y$. The forward model $f(x, z)$ predicts the diagnostic response given a control setting and a candidate latent state.

At each iteration, REAL selects the most informative control setting to query next, measures the system, updates a multi-modal posterior over the latent state, and checks for convergence. The framework comprises two coupled components: Randomized Maximum A Posteriori (rMAP) Ensemble for model training and approximate Mutual Information for active query.

---

## 2. Problem Setup

Let:
- $z^\star \in \mathcal{Z} \subseteq \mathbb{R}^{n_z}$ be the unknown latent state.
- $x \in \mathcal{X} \subseteq \mathbb{R}^{n_x}$ be the controllable input (the variable selected at each active-learning step).
- $y \in \mathcal{Y} \subseteq \mathbb{R}^{n_y}$ be the observation from diagnostic devices.

When raw diagnostic signals $o \in \mathcal{O}$ are not directly modeled by the forward model, a Virtual Diagnostics (VD) model processes them:
$$\text{VD}(o) = (y,\, \sigma_y), \qquad y \in \mathcal{Y},\quad \sigma_y \in \mathbb{R}_{>0}^{n_y}$$
where $y$ are the mapped observations in the space the forward model predicts, and $\sigma_y$ is the VD model's per-component output uncertainty. All tolerances and perturbations below are defined in this mapped space $\mathcal{Y}$.

Assume that the true system generates observations according to:
$$y_{ij} = f_j(x_i, z^\star) + \xi_{ij}, \qquad \xi_{ij} \sim \mathcal{N}(0,\, \sigma_{y,ij}^2)$$

where $i$ indexes the control setting (scan step) and $j \in \{1,\ldots,n_y\}$ indexes the observation component. The uncertainty $\sigma_{y,ij}$ is realized as a known scalar only once measurement $(i,j)$ is completed, and is thereafter treated as a fixed constant in all subsequent objectives.

The dataset at iteration $t$ is $\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^t$ with associated realized uncertainties $\{\sigma_{y,ij}\}$.

The negative log-posterior used as the inference objective is:
$$\mathcal{L}(z;\mathcal{D}_t) = R(z) + \sum_{i=1}^{t} \sum_{j=1}^{n_y} \ell\!\left(y_{ij} - f_j(x_i, z);\, \sigma_{y,ij}\right)$$

where $R(z) = -\log p(z)$ is the negative log-prior and $\ell(\cdot\,;\sigma_y)$ is an admissible loss kernel. Suitable choices include the squared (Gaussian) loss $\ell(r;\sigma_y) = r^2/(2\sigma_y^2)$, smooth approximations of the absolute (Laplace) loss $\ell(r;\sigma_y) = |r|/\sigma_y$, and other continuously twice differentiable coercive kernels.

---

## 3. Algorithm

### Algorithm 1: REAL — Main Loop

```
Input:
    forward model f, prior p(z), loss kernel ℓ
    budget T, ensemble size S, representative size K ≤ S
    controllable input space X

Initialization:
    Collect n_init observations at diverse control settings $x_1, …, x_{n_init}$.
    (Settings may be chosen by a random sampling over $X$, a space-filling
     design, or any other strategy independent of the model.)
    Record realized uncertainties $\sigma_{y,1}, …, \sigma_{y,n_init}$.
    D_t ← {(x_i, y_i, \sigma_{y,i})}_{i=1}^{n_init},   t ← n_init.

Active-learning loop:
    for t = n_init, n_init + 1, …, T − 1 do

        Step 1 — Posterior Approximation
            Construct ensemble {z^(s)}_{s=1}^S  ←  Algorithm 2(D_t, {σ_{y,ij}})

        Step 2 — Representative Subset Selection
            Select K-member index set K*  ←  Algorithm 3({z^(s)}, K)

        Step 3 — Acquisition
            x_{t+1} ← argmax_{x ∈ X}  score_K(x)
            where score_K is computed by Algorithm 4 using {z^(s) : s ∈ K*}

        Step 4 — Evaluate
            Apply control x_{t+1} to the system.
            Observe y_{t+1}; record realized uncertainties {σ_{y,t+1,j}}.
            D_{t+1} ← D_t ∪ {(x_{t+1}, y_{t+1})},   t ← t + 1.

        Step 5 — Convergence Check
            if  max_{x ∈ X} score_K(x)  ≤  noise_floor  then
                break
            end if

    end for

Final fit:
    Re-run MAP inference on the full dataset D_T with no observation
    perturbations (η_{ij} = 0 for all i, j) to obtain the deterministic point
    estimate z_hat and ensemble mean for reporting.

Output:  ensemble {z^(s)},  point estimate z_hat,  convergence diagnostics.
```

---

### Algorithm 2: rMAP Ensemble Construction

The rMAP ensemble approximates the multi-modal posterior by independently seeding $S$ optimization runs with (i) diverse initializations drawn from the prior and (ii) calibrated perturbations of the observed targets, so that different members may converge to different posterior basins.

```
Input:
    D_t = {(x_i, y_i, σ_{y,i})}_{i=1}^t,  ensemble size S,
    loss kernel ℓ,  prior R(z),  latent space Z.

for s = 1, …, S do

    1. Initialize:
       Draw z_0^(s) ~ p(z).
       Diverse initialization across Z promotes coverage of all posterior basins.

    2. Perturb observations:
       For each scan step i = 1, …, t and each observation component j = 1, …, n_y,
       draw an independent scalar perturbation:
           η_{ij}^(s) ~ N(0, σ_{y,ij}^2)
       All (i, j) draws are mutually independent across scan steps, observation
       components, and ensemble members. Channels with small σ_{y,ij} receive
       small perturbations; channels with large σ_{y,ij} permit larger variation
       across members, reflecting lower confidence in those observations.

    3. Solve the perturbed MAP problem:
       z^(s) = argmin_{z ∈ Z}  R(z) + Σ_{i=1}^t Σ_{j=1}^{n_y}  ℓ(y_{ij} + η_{ij}^(s) − f_j(x_i, z); σ_{y,ij})
       starting from z_0^(s), using any suitable local optimizer.

end for

Output:  ensemble  {z^(s)}_{s=1}^S.
```

**Theoretical guarantee.** Within the basin of any local minimizer $z_m^\star$ of $\mathcal{L}(\cdot\,;\mathcal{D}_t)$, each member initialized in that basin is approximately Gaussian with mean $z_m^\star$ and local covariance:
$$\Sigma_m^{\text{rMAP}} = H_m^{-1} \!\left(\sum_{i=1}^t \sum_{j=1}^{n_y} \frac{J_{m,ij}^\top J_{m,ij}}{\sigma_{y,ij}^2}\right)\! H_m^{-1}$$
where $H_m = \nabla_z^2 \mathcal{L}(z_m^\star;\mathcal{D}_t) \succ 0$ is the Hessian at the minimizer and $J_{m,ij} = \partial_z f_j(x_i, z_m^\star)$ is the Jacobian of the $j$-th forward model output. This covariance shrinks as more reliable observations are added, consistent with Bayesian posterior contraction.

**Implementation guidance.**
- Run more than $S$ parallel optimization trajectories (e.g., $S \times p$ for an oversampling factor $p \geq 1$) and retain the $S$ members with the lowest perturbed objective values. This improves basin coverage without altering the theoretical guarantees.
- Multiple restarts from independent random initializations further reduce the risk of missing posterior modes.
- When a reparameterization $z = g(u)$ with $u \in \mathbb{R}^{n_u}$ unconstrained is available that automatically enforces the support of $p(z)$, optimization may be performed in the unconstrained space $u$.
- A bootstrap variant — subsampling a fraction of the data per restart instead of (or in addition to) perturbing observations — provides an alternative diversification strategy when the noise model is uncertain.

---

### Algorithm 3: K-Representative Subset Selection

When $S$ is large, evaluating the acquisition score over a dense candidate grid using all $S$ members is expensive. A representative subset of size $K$ is selected to capture most of the ensemble's predictive spread at lower cost.

```
Input:
    ensemble {z^(s)}_{s=1}^S,  subset size K,
    latent-space distance metric  d : Z × Z → R_{≥0}.

Select K-subset K* ⊂ {1, …, S} maximizing pairwise diversity:
    K* = argmax_{K ⊂ {1,…,S}, |K|=K}  min_{s ≠ r ∈ K}  d(z^(s), z^(r))

Output:  representative index set K*.
```

**Special case K = 2 (recommended for computational efficiency).** Select the single pair of members maximally separated in latent space:
$$(z^{(a)}, z^{(b)}) = \arg\max_{s \neq r} \; d\!\left(z^{(s)}, z^{(r)}\right)$$

K = 2 is a *computational shortcut*, not an assumption about the number of posterior modes. The posterior may have any number of modes; the pair acts as the most extreme representatives for evaluating observational discriminability.

**Metric selection.** The metric $d$ should be *observation-informative*: the most $d$-distant pair should also produce a large spread in predicted observations across $\mathcal{X}$. Metrics defined in the forward-model output space, rather than the raw parameter space, tend to satisfy this requirement more reliably. When this holds with constant $c_d \in (0, 1]$, the K = 2 score satisfies $\text{score}_2(x) \geq c_d \cdot \text{score}_S(x)$ for all $x \in \mathcal{X}$, providing a controlled approximation of the full-ensemble score.

---

### Algorithm 4: Epistemic Acquisition Scoring and Argmax

For a candidate control setting $x$, the acquisition score measures how much the representative ensemble members disagree on the predicted observation $f(x, z)$. The control setting that maximizes disagreement is selected as the next query.

**Full-ensemble score** (using all $S$ members):
$$\text{score}_S(x) = \sqrt{\frac{1}{n_y} \operatorname{tr}\!\left(\widehat{\Sigma}_t(x)\right)}$$

where the ensemble sample covariance of the forward model output is:
$$\bar{f}_t(x) = \frac{1}{S}\sum_{s=1}^S f(x,z^{(s)}), \qquad \widehat{\Sigma}_t(x) = \frac{1}{S-1} \sum_{s=1}^S \Bigl(f(x,z^{(s)}) - \bar{f}_t(x)\Bigr)\Bigl(f(x,z^{(s)}) - \bar{f}_t(x)\Bigr)^\top$$

This score is a monotone surrogate for the Gaussian predictive entropy and for the mutual information $I(Z; \mathcal{Y} \mid x, \mathcal{D}_t)$. By the law of total covariance it simultaneously captures both inter-basin spread (mode ambiguity) and within-basin spread (residual parameter uncertainty after mode identification).

**K = 2 score** (using the representative pair from Algorithm 3):
$$\text{score}_2(x) = \frac{1}{\sqrt{2 n_y}} \left\|f(x, z^{(a)}) - f(x, z^{(b)})\right\|_2$$

The factor $1/\sqrt{2 n_y}$ places $\text{score}_2$ on the same scale as $\text{score}_S$: for a two-member ensemble the sample covariance satisfies $(1/n_y)\operatorname{tr}(\widehat{\Sigma}_t(x)) = \text{score}_2(x)^2$.

**Acquisition argmax:**
```
Input:
    forward model f,  representative ensemble {z^(s) : s ∈ K*},
    controllable input space X,  score function score_K.

Solve:
    x_{t+1} = argmax_{x ∈ X}  score_K(x)

Implementation:
    Continuous X — solve via gradient-based optimization with multiple
    random restarts to avoid local optima of the score landscape.
    Finite discrete X — evaluate score_K at each candidate and select
    the maximizer directly.

Output:  next query  x_{t+1}.
```

---

### Algorithm 5: Convergence Check

```
Input:
    representative ensemble {z^(s) : s ∈ K*},  forward model f,
    noise floor σ_noise > 0.

Compute:
    max_score = max_{x ∈ X}  score_K(x)

if  max_score  ≤  σ_noise  then
    return  CONVERGED
else
    return  NOT CONVERGED
end if
```

As the posterior concentrates toward the true latent state, the ensemble collapses, $\text{score}_K(x) \to 0$ for all $x$, and the predictive distribution approaches $\mathcal{N}(f(x, z^\star), \sigma_\text{noise}^2 I)$. The maximum ensemble spread therefore provides a natural, parameter-free stopping criterion: when $\max_x \text{score}_K(x)$ falls to or below the noise floor $\sigma_\text{noise}$, remaining epistemic uncertainty is negligible relative to measurement noise and further queries are uninformative.

---

## 4. Theoretical Properties

**Surrogate monotonicity.** Under an isotropic Gaussian predictive approximation with noise floor $\sigma_\text{noise}^2$:
$$\arg\max_{x \in \mathcal{X}} \text{score}_S(x) = \arg\max_{x \in \mathcal{X}} \tilde{h}_t(x)$$
because $\tilde{h}_t(x) = (n_y/2)\log(2\pi e\,(\text{score}_S(x)^2 + \sigma_\text{noise}^2))$ is strictly increasing in $\text{score}_S(x)$.

**Joint surrogate for both MI terms.** Let $M = g(Z)$ denote the basin label. By the law of total covariance:
$$\text{score}_S(x)^2 = \underbrace{\frac{1}{n_y}\operatorname{tr}\!\left(\operatorname{Cov}\!\left(\mathbb{E}[f(x,Z)\mid M]\right)\right)}_{\text{inter-basin}} + \underbrace{\frac{1}{n_y}\operatorname{tr}\!\left(\mathbb{E}\!\left[\operatorname{Cov}(f(x,Z)\mid M)\right]\right)}_{\text{within-basin}}$$

Both summands are non-negative, so $\text{score}_S(x)$ simultaneously accounts for mode ambiguity and residual within-basin uncertainty. Maximizing it addresses both in a single greedy step, corresponding to a joint surrogate for the two-term mutual information decomposition $I(M;\mathcal{Y}\mid x,\mathcal{D}_t) + I(Z;\mathcal{Y}\mid M,x,\mathcal{D}_t)$.

**Expected uncertainty reduction.** Selecting $x_t = \arg\max_x \text{score}_S(x)$ and observing $y_{t+1}$:
$$\mathbb{E}[\mathsf{H}(M \mid \mathcal{D}_{t+1}) \mid x_t] \;=\; \mathsf{H}(M \mid \mathcal{D}_t) - I(M;\mathcal{Y}\mid x_t,\mathcal{D}_t) \;\leq\; \mathsf{H}(M \mid \mathcal{D}_t)$$
$$\mathbb{E}[h(Z \mid M, \mathcal{D}_{t+1}) \mid x_t] \;=\; h(Z \mid M, \mathcal{D}_t) - I(Z;\mathcal{Y}\mid M, x_t,\mathcal{D}_t) \;\leq\; h(Z \mid M, \mathcal{D}_t)$$

Both mode ambiguity and within-basin spread are non-increasing in expectation at every step.

**Asymptotic contraction.** Under standard Bayesian consistency conditions and persistent informativeness of the selected queries:
$$\mathsf{H}(M \mid \mathcal{D}_t) \to 0 \quad \text{and} \quad \operatorname{tr}\!\left(\operatorname{Cov}(Z \mid M, \mathcal{D}_t)\right) \to 0 \quad \text{as } t \to \infty$$

**K = 2 quality bound.** When the metric $d$ satisfies the observation-informative assumption with constant $c_d \in (0,1]$, the surrogate information gain from the K = 2 query satisfies:
$$I_\text{sur}(Z;\mathcal{Y}\mid x_t^{(2)},\mathcal{D}_t) \;\geq\; \frac{n_y}{2}\log\!\left(1 + \frac{c_d^2\left(\max_x \text{score}_S(x)\right)^2}{\sigma_\text{noise}^2}\right)$$

so greedy maximization of $\text{score}_2$ achieves a controlled fraction of the full-ensemble information gain at substantially lower computational cost.

---

## 5. Summary of Approximations

| Approximation | Role in the framework | Becomes exact when | Improves as |
|---|---|---|---|
| rMAP ensemble $\{z^{(s)}\}$ with realized uncertainties $\{\sigma_{y,ij}\}$ | Represents $p(z\mid\mathcal{D}_t)$ via mode-centered samples | Local objective is well approximated by a quadratic | $S\to\infty$; posterior concentrates |
| Gaussian predictive surrogate $V_t(x)$ | Approximates predictive spread to form the acquisition score | Predictive distribution is Gaussian | Posterior concentrates (unimodal regime) |
| Epistemic-only score | Avoids the unobservable aleatoric term at candidate $x$ | Observation noise is homoscedastic in $x$ | Low-data regime where epistemic uncertainty dominates |
| K = 2 representative pair (metric assumption with constant $c_d$) | Reduces acquisition cost; selects the most latent-distant pair by metric $d$ | $c_d = 1$ (metric perfectly predicts observational separation) | Mode separation large relative to noise floor |