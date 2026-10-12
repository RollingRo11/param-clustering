# Sparse sufficient computation under a GIM-sized budget

## Finding

The missing target is **joint sufficiency of a selected computation**, rather than concentration of individual importance scores. A component can have a large correct derivative, a large exact single-ablation effect, and still belong to a group that can be removed together. Conversely, every single-ablation effect can be negligible while removing the whole group is destructive. Both behaviors occur without any error in the gradient calculation.

There is a mathematically grounded route to a better objective: measure the signed reconstruction defect of an entire omitted set, and account for how the retained computation propagates that defect. The defect is inexpensive to compute from intact activations and rank-one factors. Zero defect gives an exact sufficiency theorem. Small defect gives a finite-error guarantee only when accompanied by valid bounds on amplification through the selected network.

This is **not yet a demonstrated solution that beats VPD**. Three gaps remain: a tight certificate on this transformer, a way to find low-budget supports efficiently, and an optimization procedure that learns suitable shared factors. A proof that a small certificate implies low KL does not prove that the certificate is small, that optimization finds it, or that its component count beats VPD. The existing experiments actually warn that a purely local reconstruction criterion can reject an output-faithful VPD solution.

There is also a constraint boundary. An algebraic subset-reconstruction loss needs subset membership, even though it can avoid all masked target forwards, adversarial ablations, learned importance networks, and Gaussian probes. It satisfies a ban on **masked target-model training forwards**, but not a literal ban on **any subset-conditioned/masking algebra in the loss**. Under the latter interpretation, the cheap sufficient bounds identified here reduce to conservative activity-tail bounds closely related to a write-norm experiment already tried. They do not justify claiming a new lower floor.

## 1. The actual optimization problem

Let the frozen target contain matrices

$$
W_l=\sum_{c=1}^{C_l}U_{lc}R_{lc}^{\mathsf T}.
$$

The factorization is shared across inputs. For one sequence $x$, a binary support $m_{ltc}$ determines which original rank-one pieces operate at source position $t$. Define the selected matrix

$$
W^m_{lt}=\sum_c m_{ltc}U_{lc}R_{lc}^{\mathsf T}.
$$

Let $p_t$ be the intact model's next-token distribution and $p_t^m$ the distribution obtained by actually executing these selected matrices, including recomputed attention, normalization, and GELU. The deployed sparsity–fidelity frontier is

$$
K_\epsilon(x)=\min_{m\in\{0,1\}^{\sum_l TC_l}}
\frac1T\sum_{l,t,c}m_{ltc}
\quad\text{subject to}\quad
\frac1T\sum_t D_{\rm KL}(p_t\Vert p_t^m)\le\epsilon.
\tag{1}
$$

Equation (1) fixes several otherwise easy-to-miss choices. Retained coefficients stay exactly one; arbitrary rescaling is not permitted. Primitive rank-one pieces count individually; a route with ten pieces costs ten occurrences. There is no uncharged remainder. The output is a distribution, not just the top predicted label. The factor dictionary is shared, but the support may depend on the input.

The usual “components per token” is the average number retained **at each source position**. It is not generally the total number of component occurrences needed to explain one later prediction: that prediction can depend on components at many earlier positions. A theorem about one output position must count its entire selected causal prefix, or explicitly use the average-position convention in (1).

KL has a direct coding interpretation. The expected extra code length from using $p_t^m$ instead of $p_t$ is $D_{\rm KL}(p_t\Vert p_t^m)/\ln 2$ bits. Component count is only a proxy for the other part of description length: support identifiers cost approximately $\sum_{l,t}\log_2\binom{C_l}{k_{lt}}$ bits under a simple fixed-cardinality code. Shared factor storage and the selector's complexity are additional terms if the aim is total description length rather than the published L0 convention. A learned, highly complex selector cannot be treated as free in a literal minimum-description-length claim.

### Endpoint sufficiency and robust sufficiency differ

Equation (1) asks for one successful selected endpoint. A stronger condition is

$$
\sup_{a:\,m\le a\le\mathbf1}
D_{\rm KL}(p\Vert p^a)\le\epsilon.
\tag{2}
$$

This demands that every partial restoration of omitted pieces remain harmless. VPD explicitly motivates its stochastic and adversarial reconstruction losses using this stronger condition, including fractional restorations. Its paper also explicitly distinguishes jointly cancelling pieces from pieces individually safe to remove.[^vpd]

A cancellation-based gain in (1) need not satisfy (2). A method can beat VPD's measured endpoint L0 while being worse under its stated robustness criterion. Conversely, insisting on a stronger mechanistic criterion can require more pieces. A lower endpoint KL alone cannot certify mechanism preservation; the restoration-reversal experiments demonstrate this limitation directly.

## 2. What the experiments already establish

The primary gap audit used the same frozen 66,931,200-parameter target and eight validation contexts, with FP32 endpoint evaluation. The following comparisons belong to that slice; the later IFR slice is different and should not be numerically pooled with it.[^gap]

| Experiment | Observation | Supported inference |
|---|---|---|
| Frozen GIM5k, 512 pieces/token | Fisher ranking 2.65 KL; bounded endpoint search 1.59 | Ranking leaves useful combinations undiscovered |
| Frozen published VPD, same budget | Fisher 1.30; bounded endpoint search 0.67; native CI 0.29 | Even the endpoint search misses a known better support |
| VPD converted to exactly our per-matrix capacities and exact sums | About 193 pieces/token, KL 0.293 | Rank-one capacity and exact reconstruction do not necessarily cause the gap |
| 200 GIM updates from that conversion | GIM loss 9.38→6.41; inherited-support KL 0.293→1.304; refined KL 1.009 | The current objective locally pushes against this known sparse solution |
| Signed reader-conditioned IFR prototype, new contexts | Some GIM gains; converted VPD 3.59 KL versus native 0.235 | Changing the scalar attribution and adding bounded reader conditioning is insufficient |

The conversion borrows VPD-trained factors and its selector. It establishes representability on the evaluated inputs, not discovery by the present training method. The bounded support searches do not prove an optimal floor for any checkpoint. Five thousand updates do not imply equal training compute across methods.

The restoration test also needs a narrower interpretation than “only VPD can repair itself.” On the gap-audit slice, restoring the last layer increased KL for the endpoint-selected GIM support, from about 1.61 to 2.00, as well as for native VPD, from about 0.293 to 0.368. The usual magnitude-selected GIM support behaved differently. Compensation is therefore a property of the particular retained computation and support, not an ability established to be exclusive to VPD. This weakens a purely VPD-specific explanation of the KL gap.

The write-norm variant was already tested. It replaces $|R_c^T h||g^T U_c|$ by $|R_c^T h|\|U_c\|$, without a target backward pass. At approximately 430 retained pieces it had KL 5.78 versus 4.94 for its GIM comparator on that experiment's own evaluation set. Reintroducing that score as a new “certified” idea would obscure this negative result.[^write]

## 3. Why single-component importance cannot solve the general problem

### 3.1 A complete finite-ablation counterexample

Consider a two-class predictor with logits $(y,0)$, where

$$
y=z_1+z_2+z_3=1+10-10=1.
$$

Keeping component 1 alone reconstructs the target exactly. Individually removing components 2 or 3 causes a much larger KL than removing component 1. Ranking by exact individual KL therefore retains one of the cancelling large components first, producing a poor one-piece explanation.

The arithmetic audit obtains individual KL values $(0.11094,5.99745,2.37617)$. The highest-ranked single component gives KL 2.10726, while the optimal single component gives zero. This is a failure of the ranking-to-subset inference, not a Taylor approximation failure. Signs are necessary to notice the cancellation, but attaching a signed number to each component does not automatically solve the subset problem.

### 3.2 Identical individual observations can hide different joint behavior

Let two gate-response functions be

$$
f_0(m_1,m_2)=1,\qquad
f_1(m_1,m_2)=1+A(1-m_1)(1-m_2).
$$

They agree at the intact point and at both individual ablations. Their intact first derivatives also agree. Yet at joint removal they differ by $A$. More generally, a term $A\prod_{j=1}^{q+1}(1-m_j)$ is invisible to all intact derivatives through order $q$, and to interventions removing at most $q$ of those gates.

**Conclusion:** no guarantee for arbitrary joint deletions can follow from those observations alone without additional structural assumptions. This is not a lower bound against algorithms that inspect and exploit the whole network. It is an information limitation of the particular summaries being used.

Related attribution impossibility results apply to complete and linear attribution methods under specified model-richness assumptions. Their scope should not be exaggerated into a theorem that every gradient-based algorithm fails.[^impossibility] Conditional Co-Ablation provides empirical evidence that conditioning an ablation on other removals reveals effects invisible in the intact model, but its repeated interventions do not supply a GIM-cost training solution.[^coax]

### 3.3 The subset objective is not generally monotone

The exact subset $\{1\}$ in the cancellation example has zero error. Restoring component 2 worsens it. Thus “add more pieces” need not improve fidelity. Classical monotone submodular-greedy guarantees cannot be imported directly.

Weak-submodularity results for sparse regression optimize the coefficients **within each selected support** and require restricted curvature conditions. Their set function is $S\mapsto\max_{\mathrm{supp}(\beta)\subseteq S}\ell(\beta)$. Here the coefficients are fixed to zero or one and the selected network changes nonlinearly. Both differences matter.[^weak]

### 3.4 Positive signs do not imply cheap sparsity

Take 1,000 identical positive scalar contributions, each equal to 0.001. The intact logit is one. At KL tolerance 0.01, at least 689 original contributions must be retained in the two-class example. If reweighting were allowed, one contribution could simply be scaled to one.

Approximate Carathéodory results give sparse **convex combinations**; their coefficients are part of the construction. They are powerful for reweighted approximation, but do not promise a comparably sparse unweighted subset of these original pieces.[^caratheodory] Spectral sparsification and randomized compression results likewise need their exact coefficient and stability assumptions checked before being applied here.

Sparse approximation is computationally hard in broad settings.[^natarajan] This rules out a universal optimizer guarantee without assumptions; it does not establish hardness of this specific 67M checkpoint or prevent exploiting its structure. The existence witness from VPD remains a concrete reason to investigate that structure.

## 4. GIM is not just a noisy measurement of endpoint sensitivity

GIM's temperature-adjusted softmax backward is designed to recover credit suppressed by attention redundancy. Its normalization freeze deliberately removes the derivative of the normalization scale. These are useful attribution choices for identifying interacting participants, but they do not define the true derivative of endpoint KL.[^gim]

An elementary normalization example makes the distinction precise. Let

$$
N_\epsilon(h)=\frac{h}{\sqrt{\|h\|_2^2/d+\epsilon}},\qquad
r^2=\|h\|_2^2/d.
$$

Its actual radial derivative is

$$
DN_\epsilon(h)h=\frac{\epsilon}{(r^2+\epsilon)^{3/2}}h.
$$

Freezing the denominator instead gives $h/\sqrt{r^2+\epsilon}$. The ratio is $\epsilon/(r^2+\epsilon)$: the actual derivative can be tiny while the modified one is large. At $\epsilon=0$, positive radial rescaling is exactly invisible to the normalized output.

Therefore, moving from ordinary gradients to GIM can intentionally increase credit for changes that normalization compensates. This is not inherently an implementation bug. It does mean that sparsifying those credits is a different optimization target from finding a minimum sufficient endpoint.

DeepLIFT and related finite-reference decompositions help with saturation and distinguish positive and negative contributions. They still allocate a total difference relative to a reference; completeness alone does not imply that arbitrary subsets of the allocated contributions can be physically deleted.[^deeplift] The IFR experiment exhibits the same distinction in a different form.

### The scalar objective, output metric, and finite deletion are three separate issues

For intact probabilities p, the gradient of $D_{\rm KL}(p\Vert\mathrm{softmax}(z+\delta))$ with respect to $\delta$ is **zero at $\delta=0$**. Its first nonzero local term is

$$
\frac12\delta^T F\delta,\qquad F=\mathrm{diag}(p)-pp^T.
$$

This follows by differentiating equation (11) below twice. A target-label cross-entropy gradient instead measures the direction $p-e_y$. A direction orthogonal to that vector can change the output distribution substantially. Moving to a Fisher metric addresses this scalar-metric mismatch; it does not automatically address finite deletion or selected-network response.

Let $j_c$ be the actual intact logit derivative of a component gate. The local joint prediction for deleting a set D is $\frac12\|\sum_{c\in D}F^{1/2}j_c\|^2$. Summing individual Fisher magnitudes discards its cross terms. Keeping those signed vector cross terms repairs a second loss of information, but still uses an intact local approximation.

Finally, a finite ablation changes routing, normalizers, and the inputs to later selected pieces. More probes reduce estimation noise in the chosen Fisher quantity; they do not remove these nonlinear remainder terms. In the existing two-context audit, native VPD's intact joint quadratic quantity was about 29.3, compared with about 23.1 for a Fisher-selected support, despite native VPD having much better endpoint KL. The metric can order known supports incorrectly even when estimated accurately. This is evidence against solving the gap by increasing probe count alone, not evidence that derivatives are unusable inside a better objective.

## 5. A theorem that directly targets a jointly sufficient set

### 5.1 Signed synthesis defect

At matrix occurrence $v=(l,t)$, let $h_v$ be its **intact** input. For a fixed support $S_v$, define

$$
a_{vc}=R_{lc}^{\mathsf T}h_v,
\qquad
d_v(S_v)=W_lh_v-\sum_{c\in S_v}U_{lc}a_{vc}
=\sum_{c\notin S_v}U_{lc}a_{vc}.
\tag{3}
$$

Unlike the sum of individual magnitudes,

$$
\|d_v\|_2^2
=\sum_{c\notin S_v}a_{vc}^2\|U_{lc}\|_2^2
+2\sum_{c<c';\,c,c'\notin S_v}
 a_{vc}a_{vc'}U_{lc}^{\mathsf T}U_{lc'}.
\tag{4}
$$

The second sum is the exact signed interaction between omitted writes at that occurrence. Large cancelling omitted pieces have small joint defect. Computing (3) by synthesis avoids materializing the pairwise matrix in (4).

**Proposition 1 — exact sufficiency.** For a deterministic feed-forward computational DAG with unchanged embeddings and non-matrix operations, if every selected matrix occurrence satisfies $d_v(S_v)=0$, executing all selected matrices produces exactly the original outputs on that input.

**Proof.** Order the graph topologically. Input nodes agree. If a node is unchanged and all its parents agree, its output agrees. If a node is a selected matrix, its parents equal the intact input by induction and equation (3) says its selected output equals the intact output. Thus all nodes, including the logits, agree. The proof includes attention, cross-position connections, residual additions, normalization and GELU because it does not linearize them. □

This is an actual subset-sufficiency statement, not a statement about attribution mass. It is also only sufficient, not necessary. Components may be dispensable because a later operation ignores or compensates their differences; requiring zero defect at every matrix disallows such explanations.

### 5.2 Why small local defects require another term

For a changed input $h_v+e_v$, a selected linear map produces

$$
W_v^S(h_v+e_v)-W_vh_v=-d_v+W_v^Se_v.
\tag{5}
$$

Consequently,

$$
\|\hat y_v-y_v\|\le\|d_v\|+\|W_v^S\|\,\|e_v\|.
\tag{6}
$$

The amplification factor is the **selected** operator. Replacing it by the intact operator's norm is not valid.

For example, an upstream error of $(0,0.01)$ entering $\mathrm{diag}(1,1000)$ creates an output error of ten. The selected downstream map agrees perfectly with the original at the intact input $(1,0)$, so its own intact-input defect is zero. The original downstream map can be the identity, reconstructed by adding a cancelling second component. An analysis that uses only the original downstream norm predicts 0.01 and is wrong by a factor of 1,000.

Equation (5) contains the selected network's changed response to upstream errors. This is the finite effect that disappears when all deletions are replaced by independent clean-input error injections.

### 5.3 A finite-error DAG certificate

Assign each node a radius $r_v$ that bounds the difference between its selected and intact outputs. Initialize input radii at zero. For selected linear nodes use (6). For unchanged nonlinear nodes use a valid Lipschitz or multivariate perturbation bound over the entire region permitted by parent radii. Residual addition adds radii. All quantities must refer to the same fixed support.

**Proposition 2 — certified execution bound.** If every local radius rule bounds its operation for all inputs in its stated parent regions, induction over the DAG proves that each $r_v$ bounds the actual execution error. In particular, the final radius bounds the true finite logit difference.

**Proof.** The induction hypothesis places every actual parent value inside its region. The node's validated local bound then contains its actual output. This argument is independent of how the support was found. □

The result certifies any proposed support. It does not require a global support optimizer or a target-model gradient. It also does not guarantee the radii remain small enough to be useful.

## 6. Required bounds for this transformer

The following are explicit safe rules, not proposed tight implementations. They demonstrate that the certificate can include the actual operations rather than silently freezing them.

**Rank-one selected linear maps.** A cheap gauge-invariant upper bound is

$$
\|W_v^S\|_2\le\sum_{c\in S_v}\|U_c\|_2\|R_c\|_2.
$$

A sometimes tighter certified upper bound is $\|W_v^S\|_F$, calculated without forming the dense selected matrix:

$$
\|W_v^S\|_F^2
=\sum_{c,c'\in S_v}
(U_c^TU_{c'})(R_c^TR_{c'}).
\tag{7}
$$

Equation (7) costs pairwise work in the **selected** count. That can be manageable when the counts per matrix are small, but it is not automatically cheap when supports concentrate in a few matrices. Spectral power iteration alone provides no certified upper bound unless an additional valid error bound is supplied.

**RMS normalization.** For $N(h)=\gamma\odot h/\sqrt{\|h\|^2/d+\epsilon}$, a valid Euclidean Lipschitz constant on a radius-$r$ ball around $h$ is

$$
L_N\le\frac{\|\gamma\|_\infty}
{\sqrt{(\max\{\|h\|-r,0\})^2/d+\epsilon}}.
\tag{8}
$$

The Jacobian is $\mathrm{diag}(\gamma)(I/s-hh^T/(ds^3))$. The symmetric factor has tangential eigenvalue $1/s$ and radial eigenvalue $\epsilon/s^3$, giving the bound. If the uncertainty ball approaches zero norm, (8) can become enormous. Treating normalization as a constant would conceal exactly this failure mode.

**The implemented tanh-GELU.** For

$$
g(x)=\tfrac12x[1+\tanh(a(x+bx^3))],
\quad a=\sqrt{2/\pi},\ b=0.044715,
$$

$L_g\le7/4$ is a valid global derivative bound. For $t=a(|x|+b|x|^3)$, $|x|a(1+3bx^2)\le3t$ and $t\operatorname{sech}^2t\le t/(1+t^2)\le1/2$. The two derivative terms are therefore bounded in absolute value by 1 and 3/4. A certified interval-specific bound would be tighter. The numerical derivative audit observes a maximum near 1.129; the proof uses 1.75 rather than treating that grid maximum as a certificate.

**Q/K multiplication.** For $Z=QK^T/\sqrt{d_h}$, Frobenius radii satisfy

$$
r_Z\le\frac{\|Q\|_2r_K+\|K\|_2r_Q+r_Qr_K}{\sqrt{d_h}}.
\tag{9}
$$

Frobenius norms can replace the two spectral norms to obtain looser certified constants. The bilinear $r_Qr_K$ term must remain. RoPE is norm-preserving. Causal masking restricts each softmax to its allowed coordinates.

**Softmax and value mixing.** A row-wise softmax has Euclidean Lipschitz constant at most 1/2, since its Jacobian is a categorical covariance with spectral norm at most 1/2. Thus $r_A\le r_Z/2$. For $Y=AV$,

$$
r_Y\le\|A\|_2r_V+\|V\|_2r_A+r_Ar_V.
\tag{10}
$$

These rules account for changed routing rather than assuming the observed attention pattern remains valid. They can be conservative, especially for long contexts. Keeping full source-position dependencies is essential: reducing errors independently per output position before accounting for attention is not a valid certificate.

### Converting a logit bound to KL

For $p=\mathrm{softmax}(z)$ and $q=\mathrm{softmax}(z+\delta)$,

$$
D_{\rm KL}(p\Vert q)=\log\mathbb E_p[e^\delta]-\mathbb E_p[\delta].
\tag{11}
$$

The Hessian of log-sum-exp is the same categorical covariance, bounded by $\tfrac12 I$. Its Bregman remainder therefore gives

$$
D_{\rm KL}(p\Vert q)\le\frac14\|\delta-c\mathbf1\|_2^2
\quad\text{for any scalar }c.
\tag{12}
$$

An alternative bound is

$$
D_{\rm KL}(p\Vert q)\le
\frac{(\max_i\delta_i-\min_i\delta_i)^2}{8}.
\tag{13}
$$

To see (13), define $\psi(t)=\log\mathbb E_p e^{t\delta}$. Then $\psi''(t)$ is the variance of a variable whose range is at most $R=\max\delta-\min\delta$, hence at most $R^2/4$. Integrating the second derivative with weight $1-t$ gives $R^2/8$. Both bounds are finite, and both respect the exact invariance to a common logit shift.

For the final unembedding rows $u_i^T$, a hidden-state radius $r$ implies

$$
\max_i|\delta_i-c^T e|\le r\max_i\|u_i-c\|_2.
$$

Any fixed vector $c$ is allowed, for example the mean unembedding row. Equation (13) then yields $\mathrm{KL}\le\tfrac12r^2\max_i\|u_i-c\|_2^2$. This avoids a vocabulary-sized Frobenius penalty and does not incorrectly penalize common logit shifts.

Combining Proposition 2 with (12) or (13) proves that a support whose certificate is at most $\epsilon$ satisfies the desired KL constraint. It does not establish minimality or competitiveness of its cardinality.

## 7. Which cheap objectives actually imply sparsity?

### 7.1 The existing penalty already favors concentration at fixed mass

The implementation uses a smoothed concave penalty, $\sum_c[(s_c+\eta)^p-\eta^p]$, with $0<p<1$, on nonnegative attribution magnitudes. For two positive scores with fixed sum, moving mass from the smaller to the larger decreases this penalty. Its second derivative is $p(p-1)(s+\eta)^{p-2}<0$. Therefore the penalty itself does **not** prefer spreading a fixed attribution budget thinly. It prefers concentration.[^implementation]

That does not mean the measured absolute attribution mass is fixed. Exact matrix reconstruction fixes the **signed** sum $\sum_c (g^TU_c)(R_c^Th)=g^TWh$ for fixed $g,h$; it does not fix the sum of absolute values. Nor does it make the concentrated scores equal the cardinality of an output-sufficient set. The distinction between signed conservation, absolute mass, and actual sufficient-set count is central.

The added soft count $\sum_c s_c/(s_c+\tau)$ is also increasing and concave, and approaches the number of nonzero scores as $\tau\to0$ for fixed scores. It changes the price of small and large scores and introduces a numerical scale. It does not introduce joint output fidelity. This is the same broad sparsity-surrogate idea used in ordinary sparse coding; the important difference is what the other part of the objective reconstructs. An SAE's reconstruction term evaluates a decoded representation. Here the current exact-sum constraint reconstructs the **unpruned matrix**, while the sparse scores describe neither the selected matrix's output nor its downstream execution.

Literal activity L0 has an additional issue. For nonzero linear readers $R_c$ and a continuous full-dimensional activation distribution, $R_c^Th\ne0$ almost surely. The zero set is a measure-zero hyperplane. Thus literal linear-code activity is dense unless readers are globally zero or the activation distribution has special lower-dimensional structure. The finite empirical token distribution need not satisfy this assumption, but a smooth activity count still cannot be casually identified with sufficient-set L0.

### 7.2 A real cardinality guarantee from magnitude concentration

Let $b_c=\|U_c\|_2|R_c^Th|$, and $B_p=(\sum_c b_c^p)^{1/p}$, $0<p<1$. Order $b_{(1)}\ge b_{(2)}\ge\cdots$. Since $i b_{(i)}^p\le B_p^p$,

$$
\sum_{i>k}b_{(i)}
\le B_p\sum_{i>k}i^{-1/p}
\le \frac{p}{1-p} B_p k^{-(1-p)/p}.
\tag{14}
$$

The last step bounds the decreasing sum by its integral from $k$ to infinity. Retaining the largest k writes therefore gives a constructive local-defect bound. For $p=1/2$, the bound is $B_{1/2}/k$. This is a standard best-k-term tail argument, derived here explicitly so its assumptions are visible.

If valid finite propagation constants $L_v$ give an endpoint radius bounded by $\sum_v L_v\|d_v\|$, then (14) gives an actual count–error certificate. The constants must cover the selected computation and its nonlinear regions. Intact-model Jacobian norms alone do not satisfy this requirement.

There is a useful further calculation. Write $s=(1-p)/p$ and $a_v=L_v pB_{p,v}/(1-p)$. With a total continuous budget K, the best allocation of this bound solves

$$
\min_{k_v>0,\,\sum_v k_v=K}\sum_v a_vk_v^{-s}
=\frac{\left(\sum_v a_v^{1/(s+1)}\right)^{s+1}}{K^s}.
\tag{15}
$$

Lagrange multipliers give $k_v\propto a_v^{1/(s+1)}$; convexity of $k^{-s}$ proves global optimality of this continuous allocation. Integer rounding upward preserves the error bound at an overhead of less than one occurrence per nonempty allocation site. Counts are capped at their dictionary sizes; fully retained sites have zero actual defect and can be removed from the residual allocation problem.

Because $1/(s+1)=p$, the quantity inside (15) is proportional to

$$
\sum_v L_v^p \sum_c b_{vc}^p.
\tag{16}
$$

Thus the most direct mask-free magnitude certificate leads back to a **weighted classical concave write penalty**. It supplies a clear conditional interpretation, but not a new way around cancellation or downstream amplification. The previously tested write objective is already close to its local unweighted part. Given that negative experiment and the likely looseness of global propagation constants, this is not the recommended next training run.

### 7.3 Averaging over random deletions can favor the wrong factorization

An attractive alternative is to integrate a deletion loss analytically, avoiding sampled masked forwards. Its simplest version has an instructive failure. Let $z_c$ be component writes, and let D be a uniformly random omitted set of exactly d out of C components. With $q=d/C$,

$$
\mathbb E_D\left\|\sum_{c\in D}z_c\right\|^2
=q^2\left\|\sum_cz_c\right\|^2
+\frac{q(1-q)}{C-1}
\left[C\sum_c\|z_c\|^2-\left\|\sum_cz_c\right\|^2\right].
\tag{17}
$$

**Proof.** Expand the square, use $\Pr(c\in D)=q$ and $\Pr(c,c'\in D)=d(d-1)/(C(C-1))$, and collect the diagonal and cross terms. □

For fixed reconstructed sum and $0<d<C$, minimizing (17) minimizes $\sum_c\|z_c\|^2$. Without other restrictions this is minimized by equal writes $z_c=(\sum_j z_j)/C$: exactly a spread-out representation. Rank-one and shared-input constraints can limit attainment, but the direction of this local term is explicit. Independent Bernoulli omissions give the same qualitative result through $q^2\|\sum z\|^2+q(1-q)\sum\|z_c\|^2$.

This is why a cheap analytic random-dropout loss is not an automatic solution. It rewards surviving typical indiscriminate deletion, whereas the goal is to find a particularly good small support. A support-dependent deletion distribution changes the optimization problem and introduces selection again.

### 7.4 Fully data-independent matrix penalties also change the target

For any exact rank-one decomposition,

$$
\|W\|_*\le\sum_c\|U_c\|\|R_c\|.
\tag{18}
$$

The nuclear norm is a norm, so the triangle inequality and $\|ur^T\|_*=\|u\|\|r\|$ prove (18). An SVD decomposition attains equality. This objective describes globally economical matrices. It does not imply that the particular coefficients $R_c^Th$ concentrate for language-model activations, or that low-budget subsets preserve a particular prediction.

Some dependence on input geometry is unavoidable if the intended sparsity is conditional on the input. The meaningful aim is reduced dependence on labels, gradient probes, or a learned selector, plus held-out fidelity—not a distribution-free guarantee inferred from a matrix norm alone.

## 8. A concrete objective, with its constraint boundary exposed

### 8.1 The certified optimization problem

Let $\mathcal B_\theta(x,S)$ be the KL upper bound from Sections 5–6, using factors $\theta=(U,R)$, intact activations, the proposed support S, and validated finite propagation bounds. The direct certified problem is

$$
\min_{\theta,\,S(\cdot)}\;
\mathbb E_x\left[\frac{|S(x)|}{T}\right]
\quad\text{subject to}\quad
UR^T=W,\qquad \mathcal B_\theta(x,S(x))\le\epsilon.
\tag{19}
$$

The constraint has the exact intended implication: every certified support is a sufficient unweighted subset of the original pieces on that input. A practical fixed-budget training form is

$$
\mathcal L_K(\theta)=\mathbb E_x\,
\mathcal B_\theta(x,S_K(x;\theta)),\qquad |S_K|\le KT,
\tag{20}
$$

followed by the existing exact-sum projection. A low value of the **valid** bound gives low actual KL. A low value of a convenient surrogate for that bound need not.

The inexpensive core surrogate is

$$
\mathcal L_{\rm defect}
=\sum_v w_v\left\|W_lh_v-U_l\big(m_v\odot R_l^Th_v\big)\right\|_2^2,
\tag{21}
$$

with fixed positive weights and a cardinality budget. It uses the full signed vector residual, rather than one signed or unsigned score per component. This is precisely how it differs from adding another L0 proxy to the current loss. Equation (21) alone is not certified for nonzero errors; the selected amplification and nonlinear bounds cannot be dropped from the theorem.

Both (19) and (21) explicitly use subset membership. No masked target forward, CI network, adversarial inner loop, or Gaussian probe is necessary to evaluate the local defect. Nevertheless, this is **algebraic masking inside a reconstruction loss**. Under the stated strict no-masking constraint, it is a comparison candidate, not an approved replacement training method. Giving it a different name would not change that fact. No training run with it has been launched.

### 8.2 Finding supports remains an algorithmic gap

For a fixed intact activation and dictionary, the change in local squared defect from adding an omitted write z is

$$
\|d-z\|^2-\|d\|^2=\|z\|^2-2d^Tz.
\tag{22}
$$

For swapping a selected write $z_i$ for an omitted $z_j$, it is

$$
2d^T(z_i-z_j)+\|z_i-z_j\|^2.
\tag{23}
$$

These are exact signed joint updates, computable from matrix products and candidate pairs. They support a bounded, deterministic local selector with accepted moves that monotonically lower its **local** defect. They do not guarantee a globally optimal support or decreasing endpoint KL. Large mutually cancelling groups can be inaccessible to single additions or swaps, and changing support can increase the amplification certificate.

One bounded exchange pass is a reasonable diagnostic implementation. An unbounded greedy pass over tens of thousands of candidates is not plausibly within GIM's budget. Differentiating through a fixed selected support is valid within a region where that support remains unchanged; a top-k boundary creates discontinuities. Straight-through gradients would be another heuristic, not a consequence of Proposition 2.

The certificate also cannot establish the existence of a low-count **locally reconstructing** support from the VPD conversion witness. That witness preserves outputs, and its measured internal errors are large. A loss that prizes local equality may actively prefer the wrong solution. This is a reason to test objective alignment before training.

### 8.3 Arithmetic and memory cost

For a matrix with input width $d_i$, output width $d_o$, C components and T cached positions, compare dense factor-gradient arithmetic in multiply-accumulates. Intact inputs and backward signals are detached, as in the current factor-training setup; backpropagation through the frozen target is counted separately.

| Operation family | Current attribution loss | Joint local defect, fixed support |
|---|---:|---:|
| Forward reader projection | $Td_iC$ | $Td_iC$ |
| Forward output-side projection or synthesis | $Td_oC$ | $Td_oC$ |
| Reader-factor gradient | $Td_iC$ | $Td_iC$ |
| Writer-factor gradient | $Td_oC$ | $Td_oC$ |
| Gradient through synthesis to activations | included in elementwise product | $Td_oC$ |
| Total factor arithmetic | $2TC(d_i+d_o)$ | $TC(2d_i+3d_o)$ |

The local joint loss adds one output-side factor projection, while potentially removing the frozen target backward. It requires no C-by-C Gram matrix: cancellation is handled by ordinary synthesis. This makes the **core term** plausibly comparable in cost, especially with sparse gather/synthesis kernels at small supports. It does not prove the whole method costs no more than GIM.

A real comparison must include intact forward capture, support search, selected operator bounds, nonlinear certificates, exact-sum projection, optimizer steps, precision, and memory traffic. Differentiating the full certificate adds work; bounds involving pairwise selected columns cost $O(k_v^2(d_i+d_o))$ per occurrence. The full-sequence attention certificate can dominate even if component synthesis is cheap. A learned selector would add both training compute and description complexity and is outside this candidate.

For the earlier IFR prototype, reader-conditioned selection took roughly 0.65 seconds per single context, compared with about 0.013 seconds for GIM's cache-plus-score stages in that particular timing setup. Those are not whole-training-step times, but they already rule out casually calling that prototype GIM-cost.[^ifr]

### 8.4 What would make the certificate useful rather than merely true?

The main opportunity is to retain the error directions that real downstream computation removes. RMS normalization is nearly insensitive to radial changes; softmax is exactly insensitive to row-wise constant shifts; final logits are exactly insensitive to a common scalar shift. Euclidean balls discard these distinctions. Using the actual quotient geometry can tighten a certificate without inventing an attribution budget.

There are limits. For RMS, epsilon prevents exact radial invariance, and negative rescaling changes direction. For attention, a constant shift must be constant over each query's allowed keys, not over an arbitrary flattened vector. A tangent-space null direction need not remain a null direction along a large deletion. A valid refinement needs a finite remainder or an exact invariance identity.

Layerwise convex pruning and certified neural-network bounds offer relevant techniques. Net-Trim enforces layer input/output consistency for ReLU networks while relearning sparse weights; its result is not a theorem about keeping fixed rank-one components in this GELU/attention model.[^nettrim] CROWN uses validated linear or quadratic activation bounds to certify perturbations; adapting that machinery can tighten regions, but it does not solve support selection or establish the desired cost.[^crown]

The research priority is therefore **tight joint-error geometry**, not a new scalar attribution name. Under the strict no-subset-dependent-loss rule, no practical construction found here yet combines that geometry, GIM cost, and a demonstrated lower L0 floor. That is an unresolved result of the investigation, not a proof that such a construction is impossible.

## 9. What the surrounding literature does and does not provide

| Line of work | Useful idea | Missing assumption or mismatch here |
|---|---|---|
| Approximate Carathéodory | Constructive sparse approximation in vector spaces | Coefficients are reoptimized; our selected coefficients must stay one |
| Weak submodularity | Greedy bounds under restricted curvature | Refit coefficients and suitable curvature; our fixed-coefficient fidelity can be nonmonotone |
| Net-Trim | Local reconstruction plus propagated error guarantees | ReLU structure and learned sparse weights; not fixed-component transformer execution |
| SiPP | Sensitivity tails, sign-aware error analysis, allocation of pruning budgets | Distributional regularity and propagation/sign-complexity terms; not an automatic low-count guarantee |
| Compression-based generalization | Stability and interlayer error amplification determine compressibility | Reparameterization and architecture assumptions; generalization bounds are not per-token KL sufficiency |
| CROWN | Valid finite nonlinear bounds rather than an intact Jacobian | Tightness, attention products, support construction, and total compute remain separate |
| DeepLIFT and IFR | Signed contributions and finite-reference propagation | Attribution completeness or flow conservation does not certify physical deletion of a coalition |

SiPP is particularly relevant, and should not be dismissed as requiring reweighting in every variant. Its deterministic variant retains the original highest-sensitivity weights unchanged. Its random variant reweights samples. Its analysis explicitly introduces sign complexity and layer conditioning to propagate errors. The deterministic guarantee depends on the omitted sensitivity tail being small; it does not make a diffuse positive representation sparse for free.[^sipp]

Compression-based generalization bounds likewise identify stability quantities such as layer and interlayer cushions. Their existence is an argument for measuring amplification, not permission to import a noise-based compression construction into this task. No Gaussian experiment follows from this comparison.[^compression]

The literature supports a fairly sharp conclusion: mathematical sparsification guarantees exist, but their useful cardinality depends on compressibility, coefficient freedom, conditioning, or a support oracle. None of the reviewed results supplies all of the missing properties for these fixed-coefficient components on the 67M transformer. The explicit witness in our own factor class is stronger evidence of attainable representational capacity than a generic approximation theorem would be.

## 10. Failure modes and required evidence

| Failure mode | Why it invalidates a claim | Required check |
|---|---|---|
| Cancelling groups | Large individual effects coexist with joint dispensability | Joint signed residuals and actual cumulative deletion |
| Redundant backups | Every individual removal is harmless, joint removal fails | Conditional removals; do not treat singleton ranking as an oracle |
| Changed downstream response | Clean local errors can be amplified or compensated | Selected-network propagation, finite execution checks |
| Saturation or normalization credit | Modified gradients describe participation instead of dispensability | Actual nonlinear identities and finite bounds |
| Loose certificate | A true bound can reject every useful small support | Bound/actual-KL ratio on known good VPD supports |
| Local reconstruction mismatch | Good endpoint supports can have large internal defects | Compare the converted VPD witness with GIM under the proposed objective |
| Support-search failure | Bad discovered supports do not establish a bad factor floor | Include native VPD as a known support and state search limits |
| Rescaling or grouping | Reweighted atoms or multi-piece routes undercount original L0 | Keep binary coefficients and count primitive occurrences |
| Uncharged Delta | A dense remainder preserves function outside the counted pieces | Primary curves exclude Delta; label residual-retained diagnostics |
| Source/output confusion | A component at one source can affect many output positions | Store isolated source-position effects and count the causal prefix |
| Numerical floor | Tiny ablation scores can rank accumulation noise | Batch-matched references, FP64 KL, explicit zero-effect floors |
| Selection overfit | Reusing evaluation contexts rewards a tuned selector | Fresh contexts and fixed scoring rules for final comparisons |
| Compute mismatch | A cheap local formula hides an expensive selector or certificate | Whole-step GPU time and peak memory at equal context/batch/precision |
| Objective improves without lower floor | Proxy/count reductions need not improve small-budget KL | Predeclared KL-versus-count frontier, especially around VPD's native budget |

The single-source experiment is intended to isolate the failure of local gradient ranking from finite individual effects. If it substantially improves GIM, ranking was a material limitation. If it fails to recover native VPD supports, the ordering itself is known to be insufficient. If GIM still trails VPD under the same ordering, that demonstrates a gap under this selector, not a proof that no better GIM coalition exists.

## 11. Experiment submitted and reproducibility

The revised source-position jobs were submitted to Explorer on September 15, 2026:

| Checkpoint | Initial job | Training provenance |
|---|---:|---|
| Published VPD | 10357593 | Published 400,000-update decomposition |
| VPD5k | 10357594 | Matched 5,000-update checkpoint |
| GIM-1B user label | 10357595 | Original CE-attribution par-auto, 30,518 updates / 1B training positions |
| GIM5k | 10357596 | GIM backward rules, 5,000 updates |

All use the same frozen 67M target, validation context 440, and all 512 source positions. Every component is removed individually at one source position; all potentially affected output KLs are retained. There are 22,020,096 interventions per par-auto checkpoint and 19,922,944 per VPD checkpoint. The one-context scope makes this an exhaustive diagnostic, not a population-level performance estimate.

The main static ranking uses sequence-average individual KL and removes occurrences globally from least to greatest impact. A second uses final-output KL. Retained budgets are sampled at 0/32/64/128/200/256/512/1024/2048/4096/8192/16384/32768/all **average occurrences per source position**. A third curve enforces equal per-source quotas. These are cumulative deletion curves at the listed checkpoints; there is not an extra full forward after every one of the twenty million individual deletions. The exact protocol is in [individual_source67.md](../runpod_compare/individual_source67.md), and implementation in [individual_source67.py](../runpod_compare/individual_source67.py).

The superseded context-wide submissions 10357292–10357295 failed a strict FP32 equivalence preflight before scoring. The revised version caches the intact prefix at the actual intervention batch shape and checks both logit agreement and KL noise floors. It validates causal zero effects and last-layer pointwise shortcuts. All four revised jobs subsequently started and passed preflight. The largest cached/full-forward KL discrepancy was below 2.3e-15 across the checks, and the zero-ablation KL floors were below 1.6e-15. Early-source throughput was about 440–490 interventions/second. Extrapolating that rate gives roughly 12–14 GPU-hours per family, before queue gaps; later source positions require fewer output logits and should be cheaper. Final curves will only be available after exhaustive scoring. Jobs checkpoint by source/matrix and chain two-hour H200 allocations. The IFR follow-up remains stopped.

The independent arithmetic audit is [sufficiency_math_audit.py](sufficiency_math_audit.py), with results in [sufficiency_math_audit.json](sufficiency_math_audit.json). It verifies the explicit ranking counterexample, indistinguishable singleton observations, nonmonotonicity, positive-atom cardinality obstruction, downstream amplification example, the fixed-cardinality deletion moment, 10,000 KL-bound cases, 3,200 exhaustive two-layer subset cases, 200,001 tanh-GELU derivative samples, 5,000 finite RMS-bound cases, 1,000 causal-attention perturbation cases with Q/K/V all changed, and 2,000 tail/allocation cases. These are algebraic checks, not experiments on a substitute language model. Numerical testing supports the derivations and does not replace their proofs.

## 12. Decision

The evidence supports **both a selector problem and an objective-alignment problem**. Existing searches find better GIM supports, so its usual ranking is leaving performance on the table. The exact VPD conversion supplies a sparse witness inside our representation, and ordinary GIM updates move away from that witness while improving the GIM objective. Neither observation establishes the optimal floor of the trained GIM dictionaries.

The strongest proved result here is that a jointly small synthesis defect, together with controlled **selected-network** amplification, certifies a sufficient sparse subset. It explains why individual L0 pressure, signs alone, higher-order intact estimates, and unconditioned random deletion each miss part of the problem. It also identifies the precise distinction from a classical sparsity penalty: fidelity must constrain the joint retained computation.

The practical candidate is not ready for a blind 5k-step run. The next research gate is to measure whether its joint-error geometry accepts the converted VPD witness at the known low budget, whether an inexpensive selector can recover that witness, and whether the complete calculation fits the measured GIM step cost. If local-defect geometry rejects the witness, continuing to optimize it would repeat the current objective mismatch. If a certificate is only useful with costly nonlinear region propagation, it fails the compute requirement.

No reviewed theorem or completed experiment proves that a new strictly mask-free, GIM-cost method will beat VPD's L0. The report therefore separates a valid sufficiency theorem, an explicit but subset-conditioned candidate, and the still-open training/discovery problem. The pending exact source-position experiment can locate more of the empirical gap; it cannot alone settle that open problem.

## 13. Experimental follow-up: objective acceptance screen

The follow-up implements [36 objectives and diagnostics](../runpod_compare/objective67.md): 24 losses using only intact traces, and 12 losses that receive a candidate support. The intact-trace grid includes true versus frozen normalization derivatives, exact finite single-component RMS effects, and ordinary write/gradient controls. Each receives an independent projected one-step descent challenge from the converted VPD witness. The conditional grid progresses from joint matrix defects to selected sublayer and whole-block defects. Separate discovery and confirmation contexts test whether favorable rankings transfer.

There is a mathematical distinction between the block variants. If each **raw selected block output** equals its intact output at the intact block input, the same induction as Proposition 1 guarantees exact execution for a chain of those blocks. This permits compensation inside a block that matrix-by-matrix equality excludes. Equality only **after a following normalization** is weaker and does not provide that guarantee: an altered residual-stream scale can interact with the next residual addition even when the normalized branch input agrees. Such quotient losses are explicitly treated as diagnostics.

A separate [mixture analysis](../runpod_compare/objective67_analyze.py) checks 342 one- or two-term nonnegative combinations of the conditional metrics. Discovery acceptance requires a mean per-context VPD/GIM loss ratio below one and preference for VPD in at least 75% of discovery contexts; confirmation is reported separately. If no combination qualifies, the best pairwise ordering is labeled unaccepted. This exercise is not itself a strictly mask-free loss or a method for discovering native VPD supports. Runtime, checkpoint identities, projected displacements, and actual KL outcomes are recorded by the experiment; submission alone is not a favorable result.

## Sources

External sources were checked against primary papers or author-hosted research pages. The VPD comparison also uses the full local paper transcription, including its appendix. Experimental claims refer to repository artifacts rather than external papers. Equations and propositions without an external citation are derived in this report; no novelty claim is made for standard inequalities or layerwise error propagation.

[^vpd]: Goodfire, [Interpreting language model parameters](https://www.goodfire.com/research/interpreting-lm-parameters), VPD paper and appendix; local full transcription: [paper.md](../out/vpd_runtime_reference/paper.md). See the necessary-set definition, cancellation discussion, residual component, and stochastic/adversarial reconstruction objectives.
[^gap]: [Gap audit and conclusions](../gap_findings_20260915.md); [exact VPD conversion](../out/gap67_bridge_20260915/comparison.json); [200-step GIM descent from the conversion](../out/gap67_descent_20260915/comparison.json). The bridge/descent experiments use the same evaluated contexts; the IFR slice is separate.
[^write]: [Write-norm experiment](../runpod_compare/write_5000.md); [GIM plus soft-count experiment](../runpod_compare/gim_count67.md). The count comparison improves some large-budget points while worsening the critical small-budget region.
[^impossibility]: [Impossibility theorems for feature attribution](https://pmc.ncbi.nlm.nih.gov/articles/PMC10786278/). The cited limitation concerns specified attribution axioms and model classes, not every method that uses derivatives.
[^coax]: Gong, Zeng, Yuen and Lim, [Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits](https://arxiv.org/abs/2607.01940), 2026 preprint. Conditional intervention evidence is distinct from a provable cheap training objective.
[^weak]: Elenberg, Khanna, Dimakis and Negahban, [Restricted Strong Convexity Implies Weak Submodularity](https://arxiv.org/abs/1612.00804); [author-hosted full text](https://eelenberg.github.io/weak-submodular-preprint.pdf). See the support-constrained coefficient optimization defining the set function.
[^caratheodory]: Mirrokni, Paes Leme, Vladu and Wong, [Tight Bounds for Approximate Carathéodory and Beyond](https://proceedings.mlr.press/v70/mirrokni17a.html), ICML 2017. The output is a sparse convex combination, with chosen coefficients.
[^natarajan]: B. K. Natarajan, [Sparse Approximate Solutions to Linear Systems](https://doi.org/10.1137/S0097539792240406), SIAM Journal on Computing 24(2), 1995. This is a broad sparse-approximation hardness result, not a reduction for the particular checkpoint evaluated here.
[^gim]: Edin et al., [Correcting Gradient-Based Circuit Localization via Interaction-Aware Backpropagation](https://arxiv.org/html/2505.17630v4), version 4, June 1, 2026. See Section 4's treatment of softmax redundancy, normalization, and interaction credit. Its circuit-localization results should not be confused with our separate use of its backward rules to train factor decompositions.
[^deeplift]: Shrikumar, Greenside and Kundaje, [Learning Important Features Through Propagating Activation Differences](https://proceedings.mlr.press/v70/shrikumar17a.html), ICML 2017. See signed finite-reference contributions and the summation-to-delta property.
[^implementation]: [parauto_method.py](../parauto_method.py), `lp_concentration`; [gim_count67_experiment.py](../runpod_compare/gim_count67_experiment.py), soft-count objective. Exact reconstruction is a constraint on all factors together, not on the selected endpoint.
[^ifr]: [IFR implementation and protocol](../runpod_compare/ifr67.md); [IFR comparison](../out/ifr67_20260915/comparison.json). Related primary method: Ferrando and Voita, [Information Flow Routes: Automatically Interpreting Language Models at Scale](https://aclanthology.org/2024.emnlp-main.965/), EMNLP 2024. The repository prototype is a bounded adaptation, not a complete reproduction of every routing mechanism in that paper.
[^nettrim]: Aghasi, Abdi, Nguyen and Romberg, [Net-Trim: Convex Pruning of Deep Neural Networks with Performance Guarantee](https://proceedings.neurips.cc/paper/2017/hash/3fab5890d8113d0b5a4178201dc842ad-Abstract.html), NeurIPS 2017. Layerwise sparse weight retraining with ReLU consistency constraints.
[^crown]: Zhang, Weng, Chen, Hsieh and Daniel, [Efficient Neural Network Robustness Certification with General Activation Functions](https://arxiv.org/abs/1811.00866), NeurIPS 2018. Valid linear/quadratic relaxations provide finite activation bounds.
[^sipp]: Baykal, Liebenwein, Gilitschenski, Feldman and Rus, [SiPPing Neural Networks: Sensitivity-informed Provable Pruning of Neural Networks](https://arxiv.org/pdf/1910.05422), version 2, 2021. Sections 4–5 distinguish deterministic retention, randomized reweighting, sensitivity tails, sign complexity, and propagated error.
[^compression]: Arora, Ge, Neyshabur and Zhang, [Stronger Generalization Bounds for Deep Nets via a Compression Approach](https://proceedings.mlr.press/v80/arora18b.html), ICML 2018. Stability and interlayer propagation conditions matter; a compression/generalization guarantee is not automatically a sparse fixed-coefficient explanation guarantee.
