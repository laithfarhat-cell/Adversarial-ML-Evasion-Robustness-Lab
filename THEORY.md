# Mathematical Foundations of Domain-Constrained Adversarial Evasion & Layered IDS Hardening

This document derives the formal threat models, constrained optimization operators, and statistical defense mechanisms implemented in the **Adversarial ML Evasion & Robustness Lab**.

---

## 1. Tabular Cybersecurity Threat Model & Domain Constraints

Unlike unconstrained continuous domains such as digital imagery—where every pixel $x_{i,j} \in [0, 1]$ can be independently perturbed within an $\ell_p$-ball without invalidating the file format—network Intrusion Detection System (IDS) classifiers operate on **feature-space projections of structured network packets** (Pierazzi et al., *IEEE S&P 2020*).

Let $\mathcal{Z}$ denote the problem space of valid bidirectional network packet sequences and $\phi : \mathcal{Z} \to \mathbb{R}^d$ denote the flow feature extractor producing a $d$-dimensional tabular vector $r = \phi(z)$ paired with traffic class $y \in \mathcal{Y} = \{0, 1, \dots, C-1\}$ (`Benign`, `DoS`, `PortScan`, `BruteForce`).

### 1.1 Feature Partitioning & Mutability Mask

We partition the coordinate index set $\{1, \dots, d\} = \mathcal{I}_{\text{imm}} \sqcup \mathcal{I}_{\text{mut}}$ via a binary **mutability mask** $M \in \{0, 1\}^d$:

$$
M_j = \begin{cases}
0, & j \in \mathcal{I}_{\text{imm}} \quad \text{(Immutable / Discrete Protocol & Header Features)}, \\
1, & j \in \mathcal{I}_{\text{mut}} \quad \text{(Mutable Continuous Flow Kinematic Features)}.
\end{cases}
$$

* **Immutable Protocol Features ($M_j = 0$):** Transport protocol indicators (`TCP`, `UDP`), destination service port tier (`80/443`, `22/3389`, DNS/system), and TCP state-machine control flags (`SYN`, `ACK`, `RST/PSH`). Modifying these features either breaks session establishment, alters the functional semantics of the attack, or produces non-realizable fractional values (e.g., `protocol_tcp = 0.37`).
* **Mutable Flow Features ($M_j = 1$):** Packet inter-arrival times (IAT mean and standard deviation), forward/backward packet rates, aggregate byte throughput, packet length statistics, active burst durations, and idle gaps. An adversary can modulate these statistics via packet padding, traffic shaping, burst splitting, or deliberate inter-packet delays, subject to physical telemetry bounds $r_j \in [r_{\min, j}, r_{\max, j}]$ (e.g., non-negative durations and Ethernet MTU $\le 1500$ bytes).

### 1.2 Invertible Log-Physical Normalization

Because network rates and durations span five to seven orders of magnitude with heavy Pareto/log-normal tails, uniform $\ell_\infty$ budgets in raw units are ill-posed. For heavy-tailed coordinates ($j \in \mathcal{I}_{\log}$), we define the strictly increasing bijection $\psi_j : [r_{\min, j}, r_{\max, j}] \to \mathbb{R}_{\ge 0}$:

$$
\psi_j(r_j) = \begin{cases}
\log\!\big(1 + \max(0,\, r_j - r_{\min, j})\big), & j \in \mathcal{I}_{\log}, \\
r_j, & j \notin \mathcal{I}_{\log},
\end{cases}
$$

followed by affine normalization into $[x_{\min, j}, x_{\max, j}] = [0, 1]$:

$$
x_j = \frac{\psi_j(\text{clip}(r_j, r_{\min, j}, r_{\max, j})) - \psi_j(r_{\min, j})}{\psi_j(r_{\max, j}) - \psi_j(r_{\min, j})} \in [0, 1].
$$

The inverse map $\psi_j^{-1}$ recovers exact physical units (`DomainBoundedScaler.inverse_transform`), guaranteeing that every normalized point $x' \in [0, 1]^d$ with $x'_{\mathcal{I}_{\text{imm}}} = x_{\mathcal{I}_{\text{imm}}}$ maps to physically bounded telemetry within $[r_{\min}, r_{\max}]$.

### 1.3 Feasible Evasion Polytope & Exact Projection

For a clean flow $x \in [x_{\min}, x_{\max}]$ and $\ell_\infty$ budget $\epsilon \ge 0$, the **domain-valid perturbation set** is the hyper-rectangular polytope:

$$
\mathcal{S}(x, \epsilon) = \Big\{ x' \in \mathbb{R}^d \;:\; \|x' - x\|_\infty \le \epsilon, \;\; x_{\min} \le x' \le x_{\max}, \;\; (1 - M) \odot (x' - x) = \mathbf{0} \Big\}.
$$

**Proposition 1 (Exact Coordinate-Wise Projection).** *Let $x \in [x_{\min}, x_{\max}]$. Because $\mathcal{S}(x, \epsilon) = \prod_{j=1}^d \mathcal{S}_j(x_j, \epsilon)$ is a Cartesian product of closed intervals*

$$
\mathcal{S}_j(x_j, \epsilon) = \begin{cases}
\{x_j\}, & M_j = 0, \\
[\max(x_{\min, j},\, x_j - \epsilon),\; \min(x_{\max, j},\, x_j + \epsilon)], & M_j = 1,
\end{cases}
$$

*the Euclidean projection $\Pi_{\mathcal{S}(x, \epsilon)}(u) = \arg\min_{x' \in \mathcal{S}(x, \epsilon)} \|x' - u\|_2^2$ (and simultaneously the $\ell_1$/$\ell_\infty$ metric projection) is given in closed form by:*

$$
\Pi_{\mathcal{S}(x, \epsilon)}(u) = \text{Clip}_{[x_{\min}, x_{\max}]}\!\Big( x + M \odot \text{Clip}_{[-\epsilon, +\epsilon]}(u - x) \Big).
$$

*Proof.* For $M_j = 0$, the inner term vanishes and $\text{Clip}_{[x_{\min, j}, x_{\max, j}]}(x_j) = x_j$ since $x \in [x_{\min}, x_{\max}]$. For $M_j = 1$, $\delta_j = \text{Clip}_{[-\epsilon, +\epsilon]}(u_j - x_j)$ satisfies $x_j + \delta_j \in [x_j - \epsilon, x_j + \epsilon]$, and clamping $x_j + \delta_j$ onto $[x_{\min, j}, x_{\max, j}]$ yields $\max(x_{\min, j}, \min(x_{\max, j}, \max(x_j - \epsilon, \min(x_j + \epsilon, u_j))))$, which is the unique nearest point in $\mathcal{S}_j(x_j, \epsilon)$ to $u_j$. Moreover, since $x_j \in [x_{\min, j}, x_{\max, j}]$, clamping toward an interval containing $x_j$ is non-expansive around $x_j$, preserving $|\Pi_{\mathcal{S}_j}(u_j) - x_j| \le |\delta_j| \le \epsilon$. $\blacksquare$

---

## 2. White-Box First-Order Gradient Evasion

Let $f_\theta : \mathbb{R}^d \to \mathbb{R}^C$ denote the neural IDS classifier parameterized by weights $\theta$, and let $\mathcal{L}(\theta, x, y) = -\log \frac{\exp(f_\theta(x)_y)}{\sum_{c=0}^{C-1} \exp(f_\theta(x)_c)}$ denote the cross-entropy loss (weighted by $w_y = \frac{N}{C \cdot N_y}$ during training). The white-box adversary solves the constrained maximization problem:

$$
x_{\text{adv}}^* \in \arg\max_{x' \in \mathcal{S}(x, \epsilon)} \mathcal{L}(\theta, x', y).
$$

### 2.1 Constrained Fast Gradient Sign Method (FGSM)

Approximating $\mathcal{L}(\theta, x + \delta, y)$ to first order around $x$:

$$
\mathcal{L}(\theta, x + \delta, y) \approx \mathcal{L}(\theta, x, y) + \langle \nabla_x \mathcal{L}(\theta, x, y),\, \delta \rangle.
$$

Maximizing the linear form over $\delta = M \odot \tilde{\delta}$ with $\|\tilde{\delta}\|_\infty \le \epsilon$ uses the $\ell_1$–$\ell_\infty$ Hölder duality:

$$
\max_{\|\tilde{\delta}\|_\infty \le \epsilon} \langle \nabla_x \mathcal{L}(\theta, x, y),\, M \odot \tilde{\delta} \rangle
= \max_{\|\tilde{\delta}\|_\infty \le \epsilon} \langle M \odot \nabla_x \mathcal{L}(\theta, x, y),\, \tilde{\delta} \rangle
= \epsilon \, \| M \odot \nabla_x \mathcal{L}(\theta, x, y) \|_1,
$$

achieved at $\delta_{\text{FGSM}} = \epsilon \cdot \big(M \odot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))\big)$. Projecting onto physical telemetry bounds yields:

$$
x_{\text{adv}}^{\text{FGSM}} = \text{Clip}_{[x_{\min}, x_{\max}]}\!\Big(x + \epsilon \cdot \big(M \odot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))\big)\Big).
$$

### 2.2 Multi-Step Projected Gradient Descent (PGD)

Because deep ReLU/LeakyReLU networks exhibit non-linear curvature inside $\mathcal{S}(x, \epsilon)$, single-step FGSM underestimates worst-case evasion risk. Masked PGD initializes randomly inside the mutable $\epsilon$-ball to escape local flat regions:

$$
\eta^{(0)} \sim \mathcal{U}(-\epsilon, +\epsilon)^d, \qquad x^{(0)} = \Pi_{\mathcal{S}(x, \epsilon)}\!\big(x + M \odot \eta^{(0)}\big),
$$

and iterates $K$ projected sign-gradient ascent steps with step size $\alpha > 0$:

$$
x^{(t+1)} = \Pi_{\mathcal{S}(x, \epsilon)}\!\Big( x^{(t)} + \alpha \cdot \big(M \odot \text{sign}(\nabla_{x^{(t)}} \mathcal{L}(\theta, x^{(t)}, y))\big) \Big), \qquad t = 0, \dots, K-1.
$$

Across $R \ge 1$ independent random restarts, we retain per-sample $x_{\text{adv}} = \arg\max_{r \in \{1,\dots,R\}} \mathcal{L}(\theta, x^{(K, r)}, y)$.

---

## 3. Defense 1: Min-Max Saddle-Point Adversarial Training (PGD-AT)

### 3.1 Constrained Saddle-Point Formulation

Following Madry et al. (*ICLR 2018*), we harden the IDS parameters $\theta$ via hybrid clean/adversarial min-max optimization with trade-off coefficient $\lambda_{\text{adv}} \in [0, 1]$:

$$
\min_{\theta} \; \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ (1 - \lambda_{\text{adv}}) \, \mathcal{L}(\theta, x, y) + \lambda_{\text{adv}} \max_{x' \in \mathcal{S}(x, \epsilon_{\text{train}})} \mathcal{L}(\theta, x', y) \right].
$$

By **Danskin's Theorem**, for a compact constraint set $\mathcal{S}(x, \epsilon_{\text{train}})$ and continuously differentiable $\mathcal{L}(\cdot, x', y)$, if $x^*(\theta) \in \arg\max_{x' \in \mathcal{S}(x, \epsilon_{\text{train}})} \mathcal{L}(\theta, x', y)$, then $\nabla_\theta \mathcal{L}(\theta, x^*(\theta), y)$ is a valid descent subgradient of the outer envelope $\Phi(\theta) = \max_{x' \in \mathcal{S}(x, \epsilon_{\text{train}})} \mathcal{L}(\theta, x', y)$.

### 3.2 Implicit Input-Jacobian Regularization & Robust vs. Non-Robust Features

Expanding the inner maximum to first order reveals the implicit regularizer induced by PGD-AT:

$$
\max_{x' \in \mathcal{S}(x, \epsilon)} \mathcal{L}(\theta, x', y) \approx \mathcal{L}(\theta, x, y) + \epsilon \, \big\| M \odot \nabla_x \mathcal{L}(\theta, x, y) \big\|_1.
$$

Thus, PGD-AT penalizes the **masked $\ell_1$ norm of the input loss gradient**. Following Tsipras et al. (*ICLR 2019*), consider two groups of mutable flow features:
1. **Macro-kinematic features** (e.g., `fwd_pkt_rate`, `byte_rate`, `bwd_pkt_len_mean`) with large inter-class separation $\Delta \mu_j \gg 2\epsilon$ and moderate variance $\sigma_j$.
2. **Micro-timing/ratio features** (e.g., `pkt_iat_std_ms`, `fwd_bwd_byte_ratio`, `header_payload_ratio`) with small absolute margin $\Delta \mu_j < \epsilon$ but very small variance $\sigma_j \ll \Delta \mu_j$ (high clean SNR $\Delta \mu_j / \sigma_j$).

An unhardened empirical-risk minimizer assigns large weights to micro-timing features because aggregating many high-SNR coordinates minimizes clean cross-entropy. However, an $\ell_\infty$ adversary with budget $\epsilon > \Delta \mu_j$ flips the sign of every micro-feature correlation simultaneously, collapsing accuracy. PGD-AT drives $|\partial \mathcal{L} / \partial x_j| \to 0$ on small-margin micro-features and concentrates weight on large-margin macro-kinematic and immutable protocol features, incurring a modest clean accuracy reduction in exchange for a massive robust accuracy gain.

### 3.3 BatchNorm Running-Statistics Isolation

Clean flows $x$ and adversarial flows $x_{\text{adv}}$ follow distinct activation moments $(\mu_{\text{clean}}, \sigma^2_{\text{clean}}) \neq (\mu_{\text{adv}}, \sigma^2_{\text{adv}})$ (Xie et al., *CVPR 2020*). Updating `BatchNorm1d` running statistics during inner PGD steps skews inference normalization and degrades clean accuracy. Our implementation runs inner PGD ascent in `eval()` mode and disables `track_running_stats` on the adversarial forward pass during outer updates so BatchNorm tracks the clean traffic distribution accurately.

---

## 4. Defense 2: Inference-Time Feature Squeezing & Metric-Space Smoothing

Feature Squeezing (Xu et al., *NDSS 2018*) reduces the degrees of freedom available to an adversary in the mutable continuous subspace without model retraining.

### 4.1 Uniform $b$-Bit Quantization & Grid-Crossing Geometry

For mutable coordinate $j \in \mathcal{I}_{\text{mut}}$ normalized to $u_j \in [0, 1]$, uniform $b$-bit quantization with $L = 2^b - 1$ intervals defines:

$$
Q_b(u_j) = \frac{\text{round}(L \cdot u_j)}{L}, \qquad \Delta_b = \frac{1}{L}.
$$

Any perturbation $|\delta_j| < \text{dist}(u_j, \mathcal{G}_b)$, where $\mathcal{G}_b = \{(k + \tfrac{1}{2})\Delta_b\}_{k=0}^{L-1}$ is the set of quantization midpoints, satisfies $Q_b(u_j + \delta_j) = Q_b(u_j)$ exactly. However, by Widrow's quantization theorem, if $u_j \bmod \Delta_b$ is approximately uniform over $[0, \Delta_b)$, a deterministic shift $+\epsilon$ crosses a bin threshold with probability $\min(1, \epsilon / \Delta_b)$, yielding expected post-quantization drift $\mathbb{E}[Q_b(u_j + \epsilon) - Q_b(u_j)] = \Delta_b \cdot (\epsilon / \Delta_b) = \epsilon$ unless high-frequency coordinate excursions are smoothed prior to rounding.

### 4.2 Metric-Space Gaussian & Local-Median Smoothing

Unlike 2D image grids, adjacent columns in a tabular vector represent physically distinct quantities (`fwd_pkt_rate` vs. `bwd_pkt_len_mean`) and must not be blurred across unrelated columns. Instead, given a clean reference bank $\mathcal{R} = \{x^{(i)}\}_{i=1}^{N_{\text{ref}}}$, the natural metric-space generalization of Gaussian and median filtering smooths mutable coordinates over the local neighborhood in joint protocol-kinematic space:

* **Nadaraya-Watson Gaussian Kernel Smoother:**
  $$
  \hat{u}_{\text{mut}}^{\text{NW}}(x) = \sum_{i=1}^{N_{\text{ref}}} w_i(x) \, u_{\text{mut}}^{(i)}, \qquad w_i(x) = \frac{\exp\!\big(-\|x - x^{(i)}\|_2^2 / (2\sigma^2)\big)}{\sum_{\ell=1}^{N_{\text{ref}}} \exp\!\big(-\|x - x^{(\ell)}\|_2^2 / (2\sigma^2)\big)}.
  $$
* **Local $k$-NN Coordinate Median Smoother:**
  $$
  \hat{u}_{\text{mut}, j}^{\text{Med}}(x) = \operatorname*{median}_{i \in \mathcal{N}_k(x)} u_{\text{mut}, j}^{(i)}.
  $$

Blending $u_{\text{mut}}^{\text{blend}} = (1 - \beta) u_{\text{mut}} + \beta \hat{u}_{\text{mut}}(x)$ prior to $Q_b(\cdot)$ contracts sign-aligned $\pm \epsilon$ excursions back toward the clean class manifold while leaving immutable coordinates exact: $T_{\text{squeeze}}(x)_{\mathcal{I}_{\text{imm}}} \equiv x_{\mathcal{I}_{\text{imm}}}$. Furthermore, the $\ell_1$ prediction disagreement $\|p_\theta(x) - p_\theta(T_{\text{squeeze}}(x))\|_1$ provides a retraining-free evasion indicator.

---

## 5. Defense 3: Class-Conditional Mahalanobis Manifold & Entropy Rejection

Discriminative softmax classifiers partition $\mathbb{R}^m$ using open half-spaces and can assign arbitrarily high confidence to points far from the support of training data. Following Lee et al. (*NeurIPS 2018*), we equip the penultimate representation $z = f_{\text{enc}}(x) \in \mathbb{R}^m$ with a generative **tied-covariance Gaussian model**.

### 5.1 Class-Conditional Gaussian & Shrinkage Precision Estimation

Under the generative assumption $z \mid (y = c) \sim \mathcal{N}(\mu_c, \Sigma)$, the log-posterior matches the linear softmax head up to a quadratic density term. From clean training embeddings $\{(z_i, y_i)\}_{i=1}^N$, we compute empirical class centroids:

$$
\hat{\mu}_c = \frac{1}{N_c} \sum_{i : y_i = c} z_i, \qquad c \in \{0, \dots, C-1\},
$$

and fit the centered residual matrix $R = [z_i - \hat{\mu}_{y_i}]_{i=1}^N$ using **Ledoit-Wolf optimal linear shrinkage** with ridge floor $\lambda_{\text{ridge}} > 0$:

$$
\hat{\Sigma} = (1 - \rho^*) \frac{1}{N} R^\top R + \rho^* \frac{\operatorname{tr}(R^\top R / N)}{m} I_m + \lambda_{\text{ridge}} I_m \succ 0.
$$

Because $\hat{\Sigma}$ is strictly symmetric positive-definite, its precision matrix $\hat{\Sigma}^{-1}$ exists uniquely and satisfies $v^\top \hat{\Sigma}^{-1} v > 0$ for all $v \neq \mathbf{0}$. The penultimate **Mahalanobis distance** to the closest known traffic manifold is:

$$
D_M^{(z)}(x) = \min_{c \in \{0, \dots, C-1\}} \sqrt{(z - \hat{\mu}_c)^\top \hat{\Sigma}^{-1} (z - \hat{\mu}_c)}.
$$

### 5.2 Protocol-Kinematic Input Covariance & Predictive Entropy Fusion

When an adversary perturbs mutable flow statistics ($M_j = 1$) while discrete protocol headers ($M_j = 0$) remain locked, physical cross-feature correlations (e.g., inverse coupling between packet rate and inter-arrival time, or TCP flag vs. response payload consistency) break down. Evaluating input-space Mahalanobis distance conditional on predicted class $\hat{c} = \arg\max_c p_\theta(y=c \mid x)$:

$$
D_M^{(x)}(x) = \sqrt{(x - \hat{\mu}_{\hat{c}}^{(x)})^\top \hat{\Sigma}_{(x)}^{-1} (x - \hat{\mu}_{\hat{c}}^{(x)})},
$$

alongside normalized predictive Shannon entropy:

$$
H(x) = -\frac{1}{\log C} \sum_{c=0}^{C-1} p_\theta(y = c \mid x) \log p_\theta(y = c \mid x) \in [0, 1],
$$

yields the validation-standardized composite anomaly score:

$$
S(x) = w_z \frac{D_M^{(z)}(x) - \mu_z}{\sigma_z} + w_x \frac{D_M^{(x)}(x) - \mu_x}{\sigma_x} + w_H \frac{H(x) - \mu_H}{\sigma_H}.
$$

Calibrating $\tau = \operatorname{Quantile}_{1 - \alpha_{\text{FPR}}}\big(\{S(x_{\text{val}})\}\big)$ on clean validation flows targets a clean false-rejection rate near $\alpha_{\text{FPR}}$ under representative calibration data (without a finite-sample guarantee) while quarantining off-manifold adversarial evasions and unseen zero-day traffic archetypes ($S(x) > \tau \implies \text{REJECT / SOC ESCALATION}$).

---

## References

1. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2015). *Explaining and Harnessing Adversarial Examples.* ICLR.
2. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.** (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
3. **Xu, W., Evans, D., & Qi, Y.** (2018). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks.* NDSS.
4. **Lee, K., Lee, K., Lee, H., & Shin, J.** (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS.
5. **Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Madry, A.** (2019). *Robustness May Be at Odds with Accuracy.* ICLR.
6. **Pierazzi, F., Pendlebury, F., Cortellazzi, J., & Cavallaro, L.** (2020). *Intriguing Properties of Adversarial ML Attacks in the Problem Space.* IEEE Symposium on Security and Privacy (S&P).
7. **Xie, C., Tan, M., Gong, B., Wang, J., Yuille, A. L., & Le, Q. V.** (2020). *Adversarial Examples Improve Image Recognition.* CVPR.

## Evaluation limitations

The benchmark attacks the underlying classifier, not the complete squeezing or rejection pipeline. These non-adaptive attacks do not establish robustness against an adversary aware of preprocessing and detection; adaptive attacks such as BPDA and detector-aware optimization are needed. The full-stack adversarial defense rate counts rejection as success, unlike classification accuracy. Physical feature bounds and immutable coordinates alone do not guarantee realizable traffic.
