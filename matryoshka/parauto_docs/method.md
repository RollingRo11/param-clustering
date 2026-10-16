# Parameter autoencoder: method

## What this does in one paragraph

Take a trained network and freeze it. Pick its weight matrices. Rewrite each matrix as a sum of many tiny pieces, where every piece reads one direction of the matrix's input and writes one direction of its output. The pieces are required to add back up to the original matrix exactly — always, not approximately. Then train the pieces so that, on any given token, only a few of them are doing anything. "Doing anything" means: this piece picks up a lot of the input **and** what it writes actually moves the loss. Nothing is ever deleted, masked, or ablated during training.

That last sentence is the whole difference from the APD / SPD / VPD line, and it is what the rest of this document is about.

---

## 1. Setup

### 1.1 The object we decompose

The target is a frozen model. We do not train it, and we never change it.

We take its internal weight matrices — for the 67M four-layer Pile model, the 24 matrices $W_{Q}, W_{K}, W_{V}, W_{O}, W_{\text{up}}, W_{\text{down}}$ in each of the four layers.

Each matrix is just a linear map. It takes the vector coming in, $h$, and produces the vector going out:

$$y = W h.$$

### 1.2 What a component is

We write each matrix as a sum of $C$ **rank-one** pieces:

$$W \;=\; \sum_{c=1}^{C} u_c\, r_c^{\top}, \qquad r_c \in \mathbb{R}^{d_{\text{in}}},\quad u_c \in \mathbb{R}^{d_{\text{out}}}.$$

Quick reminder on what a rank-one piece does. Apply $u_c r_c^{\top}$ to an input $h$:

$$\left(u_c r_c^{\top}\right) h \;=\; \underbrace{(r_c^{\top} h)}_{\text{a single number}} \; u_c .$$

So a piece has exactly two parts:

- $r_c$ is the **read** vector. It measures one number off the input: how much $h$ points along $r_c$.
- $u_c$ is the **write** vector. It writes that number into the output along one fixed direction.

One reader, one writer, one number in between. That is a component.

Both $r_c$ and $u_c$ are free parameters. Neither is tied to a row, a column, a neuron, or a singular vector of $W$. They live in the matrix's original input and output spaces, at full dimension.

We use more pieces than the matrix's rank ($C > \min(d_{\text{in}}, d_{\text{out}})$): 1,600 total on the toy, 43,008 total on the 67M model. Because the set is overcomplete, there are infinitely many ways to write $W$ as such a sum. Training picks one of them.

### 1.3 The hard constraint: the pieces must add up to $W$

At every single training step:

$$\sum_{c} u_c r_c^{\top} \;=\; W \quad \text{(exactly, to float roundoff)}.$$

This is enforced structurally, not with a loss term. There is:

- no faithfulness loss and no coefficient balancing it against anything,
- no leftover "delta" or residual component,
- no backbone or catch-all piece.

If the pieces are ever off the constraint, we snap them back. See §4.

This matters more than it looks. It means every quality number we report is uncontaminated: at zero components removed, the reconstruction is the model, full stop. There is no error floor we have to explain away, and there is nowhere for weight mass to hide.

---

## 2. The score: read $\times$ write

This is the core of the method.

We need a number saying "how active is piece $c$ on token $x$." Take a scalar loss $L$ — for the language model, the ground-truth next-token cross entropy. Cache two things per token per matrix:

- $h(x)$, the input arriving at the matrix,
- $g(x) = \frac{\partial L}{\partial y}$, the gradient of the loss with respect to the matrix's output. (This is one ordinary backward pass. It is the standard "how much does the loss care about each output coordinate.")

Then define, for each piece:

$$\rho_c(x) = r_c^{\top} h(x) \qquad \text{(how much the piece reads)}$$
$$\omega_c(x) = g(x)^{\top} u_c \qquad \text{(how much its write moves the loss)}$$
$${\;a_c(x) \;=\; \left|\rho_c(x)\right|\cdot\left|\omega_c(x)\right|\;}$$

**In words:** how strongly this piece reads the input, times how strongly its write affects the output.

**What it equals.** Delete piece $c$ and the matrix output changes by $-\rho_c u_c$. To first order the loss changes by $-\rho_c \omega_c$. So $a_c$ is exactly the size of this piece's first-order contribution to the loss on this token. It is not a proxy; it is the derivative.

**Why both halves are needed.** A piece that soaks up a lot of the input but writes into a direction the loss is flat along scores zero. A piece that writes into a direction the loss cares about a great deal, but is never read on this token, also scores zero. A piece only counts when both are true.

### 2.1 The invariance that makes this work

The exact-sum constraint gives us something useful for free. Sum the *signed* scores over all pieces:

$$\sum_c \rho_c \,\omega_c \;=\; \sum_c (r_c^{\top}h)(g^{\top}u_c) \;=\; g^{\top}\left(\sum_c u_c r_c^{\top}\right) h \;=\; g^{\top} W h .$$

The right-hand side has no $r_c$ or $u_c$ in it. **The total signed effect is fixed by the frozen matrix and cannot be changed by training.** Training can only decide how to spread that fixed total across pieces.

This is why the objective below is well-posed and cannot collapse to "make everything small."

### 2.2 The toy's alternative sensor

On the toy we also use a vector-valued sensor instead of a scalar loss. There $g$ is replaced by the full output Jacobian $J(x)$, and

$$a_c(x) \;=\; \left|r_c^{\top}h(x)\right| \cdot \frac{\lVert J(x)\, u_c\rVert_2}{\sqrt{n_{\text{out}}}} ,$$

which is the RMS change in the model's output vector if you deleted piece $c$. Same shape of quantity: read amount $\times$ write effect. The scalar cross-entropy version is the default for the language model; both are implemented and both can be used for ranking at evaluation time.

---

## 3. The loss: concentration

We want few active pieces per token. Sum a concave power of the scores:

$$\mathcal{L}_W \;=\; \sum_{c}\left[\left(a_c + \epsilon\right)^{p} - \epsilon^{p}\right], \qquad \epsilon = 10^{-6}.$$

$p$ is annealed from $1$ down to $0.5$ over the first 80% of training.

Reminder on what $p$ does:

- At $p = 1$ this is just the sum of absolute scores. Combined with §2.1, the signed sum is already fixed, so the only way to shrink the *absolute* sum is to stop pieces from cancelling each other out. **$p = 1$ removes cancellation.**
- At $p < 1$ the function is concave, so one score of size $2$ costs less than two scores of size $1$. The cheapest configuration is a few large scores and a lot of near-zero ones. **$p < 1$ concentrates.**

Starting at $p = 1$ and annealing down keeps gradients sane early, when the pieces are still random and every score is mid-sized. The $\epsilon$ makes the term zero at zero with a finite derivative there, instead of infinite.

**Per-matrix normalization.** The 24 matrices have wildly different scales. Each matrix's loss is divided by its own value at initialization, measured on a fixed 256-position reference set drawn from training data and evaluated at the current $p$. The normalized losses are then **summed** (not averaged) and one backward pass and one Adam step updates every piece of every matrix at once. Summing preserves what each matrix's own gradient would have been.

---

## 4. Optimization: projected Adam

One step is:

1. Project the current factors onto the constraint set (so the loss is evaluated on a feasible decomposition, and gradients flow through the projection).
2. Compute scores and the summed normalized loss; backward; Adam step. Adam, $\beta = (0.9, 0.99)$, learning rate cosine-decayed $10^{-3} \to 10^{-5}$.
3. Project again, in place.

The projection has a closed form. Stack the pieces into $R = [r_1 \cdots r_C]$ and $U = [u_1 \cdots u_C]$, so the constraint is $U R^{\top} = W$. For a wide matrix, hold $U$ and correct the reads:

$$R \;\leftarrow\; R + \left(W - U R^{\top}\right)^{\top}\left(U U^{\top}\right)^{-1} U .$$

For a tall matrix, hold $R$ and correct the writes the same way. In words: take whatever is still missing from $W$, and push it back through the held factor by least squares. This is the smallest correction to one factor, not a joint nearest-pair projection to the constraint set.

The solve is small ($d_{\text{out}}\times d_{\text{out}}$ or $d_{\text{in}}\times d_{\text{in}}$) and done in FP64; stored factors are FP32. Initialization is seeded random factors, immediately projected.

There is no cancellation cap, no feasibility warmup, no line search, and no update rollback. Every Adam update is applied and then projected. The fixed normalization set only sets loss scales; it never accepts or rejects an update.

---

## 5. What makes this different from APD / SPD / VPD

This section is the point of the method.

### 5.1 What they do

The APD $\to$ SPD $\to$ VPD line also decomposes matrices into rank-one pieces $u_c r_c^{\top}$. The pieces look the same. **The definition of "active" is completely different.**

There, activity is a learned **causal importance** gate $\gamma_c(x) \in [0,1]$, produced by a separate network trained alongside the pieces — in the published configuration this repo pins, an 8-block, width-2048 transformer that reads the token and emits a gate for every one of the 38,912 components. Its training objective has four parts:

- **Faithfulness**: push $\sum_c u_c r_c^{\top}$ toward $W$ (a loss with a large coefficient, plus a trainable delta component to absorb what is left).
- **Stochastic subset reconstruction**: sample random subsets of components, mask them by their gates, and require the model's output to survive (KL to the intact target).
- **Persistent PGD reconstruction**: adversarially search, per batch and per position, for the mask that breaks the output the most, and require the output to survive that too.
- **Importance minimality**: push all gates toward zero ($\ell_p$, $p$ annealed $2 \to 0.4$).

Read those together and the gate's meaning falls out:

> $\gamma_c(x)$ is **how much of this piece must remain so the output stays the same, even while other pieces are partially removed too.**

### 5.2 Side by side

| | **Ours** | **APD / SPD / VPD** |
|---|---|---|
| Activity of a piece | $\lvert$read$\rvert \times \lvert$write effect$\rvert$ — a derivative of the frozen model | A learned gate, trained under ablation |
| Where it comes from | Measured. Two matmuls on cached $h$ and $g$ | Predicted by a separate transformer |
| Is anything removed during training? | No | Yes — random subsets and adversarial masks, every step |
| Reconstruction | Exact by projection, every step | A loss term, plus a delta component |
| Extra trained parameters | None | The full causal-importance network |
| Extra forward passes per step | None | One per mask sample, plus every PGD inner step |

### 5.3 Sculpting versus etching

**We sculpt.** We build pieces up. Training pushes each piece toward having a strong read and a strong write, and pushes the redundant ones toward reading and writing nothing. It is additive: the material is added where the model uses it.

**They etch.** They carve pieces out by repeatedly removing them and seeing what breaks. A piece earns its existence by being painful to delete. It is subtractive: the material is removed wherever the model tolerates its removal.

### 5.4 Cost

Ours, per matrix per step: two matmuls ($hR$ and $gU$), an elementwise product, a concave sum, and one small linear solve. That is it. No second network, no extra forwards, no inner optimization loop.

Measured on the paired H200 pilot (**20 updates, throughput only**):

| Method | Seconds / update | Peak allocated GPU memory |
|---|---:|---:|
| Par-auto | 0.447 | 41.75 GB |
| VPD | 2.203 | 138.46 GB |

These are throughput estimates from a short pilot. **They are not evidence that the two decompositions are of equal quality.** The matched 1B-position run that would test that is described in [`runpod_compare/README.md`](runpod_compare/README.md).

---

## 6. The bet

Two claims, both currently **untested**. Stating them plainly so they can be shot down.

**Bet 1 — similar quality, a fraction of the price.** The read $\times$ write score is a first-order quantity available from one backward pass. If it ranks components about as well as a gate trained against adversarial ablation, then the entire causal-importance apparatus is buying very little for roughly 5x the wall-clock and 3x the memory. The falsifier is the matched-budget KLKeep curve: same target, same tokens, same optimizer updates, each method ranking by its own native sensor.

**Bet 2 — ablation is data-hungry in a way that actively hides mechanisms.** Consider a real mechanism in the frozen weights that the training tokens happen never to exercise. Under an ablation objective, removing it never hurts any output, so no reconstruction term ever pushes back — while the minimality term pushes its gate to zero for free. The method concludes the mechanism does not exist, and masks it. Worse, because faithfulness is a soft loss with a delta component, the corresponding weight mass has somewhere to go.

Our objective has no such pressure. The read and write vectors are geometry of the frozen weights; exactness is structural, so every part of $W$ is inside the pieces at all times, whether or not any training token lit it up. An unexercised mechanism is not punished — it simply does not get sharpened, and it is still there to be found by its score on a token that does exercise it.

The falsifier: hold out a slice of data that carries a specific mechanism, train both methods without it, then check on that held-out slice whether the mechanism is still recoverable from each decomposition.

---

## 7. Implementation notes

### 7.1 Toy versus language model

**Toy** — the verified 10,000-step residual model: 100 input features, two layers, residual width 1,000, MLP width 25. 400 components per matrix, four matrices, batch 2,048. Its 400 components exceed the matrix ranks and the MLP width, but not the residual width.

**67M Pile model** — all 24 internal matrices. Components per layer: Q/K/V/O $=1024/1024/1024/1024$, MLP up/down $=3072/3584$; 43,008 total. Only Q/K differ from VPD's budget — VPD's 512 factors need a residual delta to close the gap, whereas our factors alone must sum to the full-rank 768-wide matrix, so 512 is not available to us. Batch is 32,768 positions (64 contexts of 512). Four contexts at a time pass through the frozen FP32 model, and one backward of their summed next-token cross entropy supplies every matrix's cotangent at once. Caches are BF16, consumed once and freed.

### 7.2 Exact structured scoring (toy)

On the toy, computing $\lVert J u_c \rVert$ naively means materializing a full output Jacobian per token. We avoid that. Write $A_l = W_{\text{in},l}$, $B_l = W_{\text{out},l}$, $F = W_U^{\top}$, and $D_l(x)$ for the cached ReLU mask. The four Jacobians are

$$J_{\text{in},0} = F(I + B_1 D_1 A_1) B_0 D_0, \qquad J_{\text{out},0} = F(I + B_1 D_1 A_1),$$
$$J_{\text{in},1} = F B_1 D_1, \qquad J_{\text{out},1} = F.$$

The last is shared by every token. Input norms use 25-row QR factors instead of 100-row Jacobians. For the first output matrix, a complete orthonormal output basis separates the range of $F B_1$ from its complement; only the 25-dimensional parallel term depends on the token's ReLU mask, and the perpendicular squared norm is computed once per update and broadcast. Working in sums of squares avoids the catastrophic cancellation you get from subtracting expanded quadratic terms near zero. QR setup is FP64; scoring is ordinary eager PyTorch. No custom kernel, no compiler, no learned gate. The cache is 181 MiB. Dense Jacobians remain as a validation reference, and tests compare scores and gradients, inactive ReLUs, null writes, projections, and joint versus sequential optimizer updates.

### 7.3 Precision

Default scoring is FP32. `--precision bf16` autocasts the score matmuls (no loss scaling); the historical `--precision fp16` adds dynamic gradient scaling and records overflow-skipped updates separately from scheduled steps. In every case, master parameters, Adam state, loss reductions, norms, initialization and normalization scores stay FP32, and projections stay FP64 (FP32 in the streaming language-model path). Evaluation always uses FP32 scores and forwards.

---

## 8. Verification

Score concentration is a first-order statement, so concentration alone does **not** prove that a set of components is jointly removable. Everything is checked with real masked forwards.

- Thresholds are chosen on calibration tokens, then checked with actual masked forwards on fresh tokens, including restored supersets.
- A reconstruction pass means output MSE $\le 10^{-4}$; only events whose full MLP contribution itself exceeds that tolerance are counted as meaningful. A pass rate depends on the threshold and the token subset chosen.
- KLKeep freezes each method's own native per-position global ranking on the intact target, then does a real nonlinear forward at every keep fraction with binary masks. These are cumulative-prefix tests; they do not cover arbitrary subsets. VPD's all-kept reconstruction error stays in its curve.
- The SPD figure-10 comparison follows their equations 11–12: pick one input component per feature and layer by its largest signed neuron contribution, then compare its whole 25-neuron profile (5,000 points; correlation, relative RMSE and slope). Selecting 100 distinct components does not count every active component, and this is not the same quantity as the pass rate above.

Results are in [`results.md`](results.md); the whole-model setup is in [`pile_scalar.md`](pile_scalar.md).
