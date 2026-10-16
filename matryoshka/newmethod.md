# Matryoshka par-auto

*Written 23–26 September 2026; updated 7 October 2026 to the settings of the 13B-token KISSKI run. Code: `matryoshka/matryoshka67.py` (one training step) and `matryoshka13b.py` (the 13B run), branch `worktree-matryoshka-parauto`.*

## In one paragraph

Split every weight matrix into many rank-one pieces that add back up to the matrix exactly. For each token, rank the pieces with a cheap score computed from the pieces themselves. Pick a random budget $k$, keep only the top-$k$ pieces, run the model on just those pieces, and train the pieces so that this cut-down model still gives the same output as the real one. Training at every budget at once is the "Matryoshka" trick from MAttr. Small budgets are drawn more often than large ones. Two penalties are added: a sparsity penalty that asks each token's scores to sit on a few pieces, and a mass penalty that stops pieces from growing large and cancelling each other. There is no gate network and no adversarial masks.

---

## 1. The ideal method

### Pieces

Each of the 24 weight matrices (Q, K, V, O, MLP up, MLP down in each of 4 layers) is written as an exact sum of rank-one pieces:

$$W_m = \sum_{c=1}^{C_m} u_c\, r_c^\top$$

- $W_m$ is weight matrix number $m$ ($m = 1, \dots, 24$). It maps an input vector of size $d_{\text{in}}$ to an output vector of size $d_{\text{out}}$.
- $C_m$ is the number of pieces in that matrix, and $c$ indexes the pieces.
- $r_c$ (size $d_{\text{in}}$) is the piece's **read** direction and $u_c$ (size $d_{\text{out}}$) is its **write** direction. $u_c r_c^\top$ is a rank-one matrix.
- Stacking the directions as columns gives $R = [r_1, \dots, r_{C_m}]$ and $U = [u_1, \dots, u_{C_m}]$, so $W_m = U R^\top$.

On input $h$ a piece outputs $(r_c^\top h)\, u_c$: it reads how much of $h$ lies along $r_c$ and writes that amount along $u_c$. The matrix's output is the sum over its pieces, $W_m h = \sum_c (r_c^\top h)\, u_c$. There are $N = 43{,}008$ pieces across all 24 matrices.

### Score

For a token $x$ (a position in a sequence, with its context), each piece gets a score: how much it reads, times how much its write matters to the model's output.

$$a_c(x) = \lvert r_c^\top h(x) \rvert \cdot \sqrt{\mathbb{E}_{y \sim P(\cdot \mid x)} \Big[ \big( u_c^\top g_y(x) \big)^2 \Big]}$$

- $h(x)$ is the input this piece's matrix sees at token $x$, so $\lvert r_c^\top h(x) \rvert$ is how much the piece reads.
- $P(\cdot \mid x)$ is the model's own predicted distribution over next tokens, and $y$ is a next token sampled from it.
- $g_y(x)$ is the gradient of $\log P(y \mid x)$ with respect to this matrix's output at token $x$. So $u_c^\top g_y(x)$ is how much nudging the output along $u_c$ changes the log-probability of $y$.
- The square root of the average over $y$ is the **Fisher norm** of the write direction. It is the norm that KL uses, so $a_c(x)$ is, to first order, how much removing piece $c$ alone would change the output distribution.

The score has no parameters of its own: it is computed from the pieces and the model.

### Budget

The budget $k$ is how many pieces each token keeps. It is drawn at random, with small budgets more likely:

$$\Pr(k) \propto k^{-1.5}, \qquad k \in \{32, \dots, N - 1\}$$

The median draw is about 121 pieces. 45% of draws are below 100 pieces, 75% are below 1% of all pieces ($k < 430$), and 94% are below 10% ($k < 4{,}301$).

### Masked model

For each token, keep the set $S_k(x)$ of its $k$ highest-scoring pieces, ranked across all 24 matrices together. The keep mask is

$$z_c(x) = \begin{cases} 1 & \text{if } c \in S_k(x) \\ 0 & \text{otherwise} \end{cases}$$

and every matrix's output is replaced by the sum of its kept pieces only:

$$\hat{o}(x) = \sum_c z_c(x)\, \big(r_c^\top h(x)\big)\, u_c$$

Here $h(x)$ is the masked model's own input to the matrix, so errors in early layers carry forward. Everything else in the model (embeddings, norms, attention softmax, MLP nonlinearity) stays the same. Write $P_{S_k}(\cdot \mid x)$ for the masked model's next-token distribution.

### Getting a gradient to the ranking

Top-$k$ is a step function, so it has no useful gradient. In the backward pass it is replaced by the **sigmoid top-$k$** mask (Wijk et al. 2025):

$$\alpha_c(x) = \frac{a_c(x)}{a_c(x) + \theta_x}$$

- $\alpha_c(x)$ is a soft version of $z_c(x)$, between 0 and 1. Pieces with scores far above $\theta_x$ get $\alpha \approx 1$, far below get $\alpha \approx 0$.
- $\theta_x$ is a threshold chosen separately for each token so that $\sum_c \alpha_c(x) = k$. It depends on every piece's score.
- This is the same as $\alpha_c = \sigma(\log a_c - \log \theta_x)$, where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid.

The forward pass uses the hard mask $z$; the backward pass uses the gradient of $\alpha$. In one line: the mask is $z_c + \alpha_c - \operatorname{sg}(\alpha_c)$, where $\operatorname{sg}$ (stop-gradient) means "use this value but pass no gradient through it". The gradient tells each piece near the cut-off whether ranking higher or lower would lower the KL, including the effect through $\theta_x$.

### Sparsity penalty (Lp)

The old par-auto penalty, on the same scores that rank the pieces. For matrix $m$ and token $x$:

$$\mathrm{Lp}_m(x) = \sum_{c \in m} \Big( \big(a_c(x) + \varepsilon\big)^p - \varepsilon^p \Big)$$

- The sum runs over the pieces of matrix $m$.
- $\varepsilon = 10^{-6}$ only keeps the gradient finite when a score is 0. Without it the term is $\sum_c a_c^p$.
- $p$ is between 0.5 and 1. With $p < 1$, the same total score costs less when it sits on a few pieces than when it is spread over many (for example $1^{0.5} = 1$, but $0.25^{0.5} \times 4 = 2$). So the penalty pushes each token to use few pieces.

$p$ starts at 1 and falls to 0.5 over the first 80% of training:

$$p(t) = 1 - 0.5 \cdot \min\!\Big(1, \frac{t}{0.8\, T}\Big)$$

where $t$ is the training step and $T = 100{,}000$ is the total number of steps. The full penalty divides each matrix's term by its value for the starting pieces, so every matrix counts equally:

$$\mathcal{L}_{\text{Lp}} = \frac{1}{24} \sum_{m=1}^{24} \frac{\mathbb{E}_x \big[ \mathrm{Lp}_m(x) \big]}{Z_m}$$

$Z_m$ is the average of $\mathrm{Lp}_m$ for the starting pieces over a fixed set of 256 positions, recomputed at the current $p$.

### Mass penalty

A piece's size is $\lVert u_c r_c^\top \rVert_F = \lVert u_c \rVert \, \lVert r_c \rVert$, where $\lVert \cdot \rVert_F$ (the Frobenius norm) is the square root of the sum of a matrix's squared entries. A matrix's **mass** is the total squared size of its pieces, as a share of the matrix's own squared size:

$$\mathrm{mass}_m = \frac{\sum_{c \in m} \lVert u_c \rVert^2 \lVert r_c \rVert^2}{\lVert W_m \rVert_F^2}$$

Reminder of why this is a useful number. Expanding the squared size of the sum gives

$$\lVert W_m \rVert_F^2 = \sum_{c} \lVert u_c \rVert^2 \lVert r_c \rVert^2 + \sum_{c \neq d} (u_c^\top u_d)(r_c^\top r_d)$$

The first sum is the pieces' own sizes; the second is the cross terms between pairs of pieces. So:

- $\mathrm{mass}_m = 1$ when the cross terms add up to zero, for example when the pieces are at right angles to each other.
- $\mathrm{mass}_m > 1$ when some pieces are large and cancel each other (negative cross terms).
- $\mathrm{mass}_m < 1$ when pieces overlap and point the same way (positive cross terms).

The penalty pulls every matrix toward mass 1:

$$\mathcal{L}_{\text{mass}} = \frac{1}{24} \sum_{m=1}^{24} \big( \mathrm{mass}_m - 1 \big)^2$$

### Objective

KL divergence measures how different two distributions are; it is 0 when they are the same:

$$\mathrm{KL}(P \,\Vert\, Q) = \sum_{v} P(v) \log \frac{P(v)}{Q(v)}$$

where $v$ runs over the vocabulary. The full objective is

$$\min_{\{u_c,\, r_c\}} \; \mathbb{E}_x \, \mathbb{E}_{k} \Big[ \mathrm{KL}\big( P(\cdot \mid x) \,\Vert\, P_{S_k}(\cdot \mid x) \big) \Big] + \lambda \, \mathcal{L}_{\text{Lp}} + \mu \, \mathcal{L}_{\text{mass}} \quad \text{subject to} \quad \sum_{c} u_c r_c^\top = W_m \text{ for every } m$$

- $P(\cdot \mid x)$ is the real model's next-token distribution and $P_{S_k}(\cdot \mid x)$ is the masked model's.
- $\mathbb{E}_x$ averages over tokens in the training text; $\mathbb{E}_k$ averages over the budget distribution above.
- $\lambda = 1$ weights the sparsity penalty and $\mu = 10$ weights the mass penalty.

The KL term is the same quantity the KLKeep evaluation measures. The two penalties shape the pieces.

### Why each part is there

- **Every budget at once.** One ranking has to work at every budget, so the kept sets are nested: the top 43 pieces are inside the top 430.
- **More small budgets.** Small budgets are the short explanations we want. Drawing $k$ evenly on a log scale (the first runs) put only about a third of draws below 1% kept; the $k^{-1.5}$ draw puts three quarters there.
- **Sparsity penalty.** With the masked KL alone, pieces grew large, and keeping most pieces could give a worse output than keeping fewer.
- **Mass penalty.** Even with the sparsity penalty and the new budgets, some matrices had large pieces that cancelled each other. Dropping only some of them breaks the cancellation. The penalty pulls every matrix back to mass 1.
- **Hard mask in the forward pass.** With the soft mask $\alpha$ in the forward pass too (MAttr's default), the budget got spread thinly over small fractions of thousands of pieces. The hard mask keeps exactly $k$ whole pieces.

---

## 2. In practice (the 13B run)

One training step is 256 sequences × 512 tokens = 131,072 tokens, split over 4 GPUs with 64 sequences each. The data is VPD's exact token stream from its published run: our step $t$ is VPD's steps $4t$ to $4t + 3$. $T = 100{,}000$ steps is 13.1B tokens, the same as VPD.

Each step, on each GPU:

1. **Cache the score inputs** from one pass through the real model. Sample $K = 4$ next tokens $y^{(1)}, \dots, y^{(4)}$ at every position from the model's own prediction, and run one backward pass for each, of the log-probability of all the sampled tokens in the sequence (so a token's gradient also counts its effect on later tokens). The expectation in the score becomes an average over the 4 samples: $a_c(x) \approx \lvert r_c^\top h(x) \rvert \cdot \sqrt{\tfrac{1}{4} \sum_{j=1}^{4} \big( u_c^\top g^{(j)}(x) \big)^2}$, where $g^{(j)}(x)$ is the gradient at the matrix output for the $j$-th set of samples.
2. **Project the pieces onto the exact sum** (step 7's formula), inside the gradient graph, so the gradient also passes through the projection.
3. **Draw 64 budgets, one per sequence, stratified.** Draw $v_i \sim \mathrm{Uniform}(0, 1)$ and set $u_i = (i + v_i)/64$ for $i = 0, \dots, 63$, so there is exactly one $u_i$ in each 1/64 slice of $[0, 1]$. Turn each into a budget by inverting the distribution's cumulative probability: $k_i = \operatorname{round}\Big( \big( 32^{-1/2} + u_i \, (43{,}007^{-1/2} - 32^{-1/2}) \big)^{-2} \Big)$. Shuffle the 64 budgets across the sequences. Every step then covers the whole range of budgets, and each sequence's $k$ still follows $\Pr(k) \propto k^{-1.5}$. Each GPU draws its own 64.
4. **For each sequence** (16 at a time):
    - score all 43,008 pieces for each of its 512 tokens;
    - **forward pass:** keep exactly the top-$k$ pieces per token (the same hard mask KLKeep uses) and run the masked model;
    - **backward pass:** the gradient reaches $u_c$ and $r_c$ through the kept pieces' outputs and through the soft mask $\alpha$, with $\theta_x$ found per token by bisection (40 halvings) so the $\alpha$ sum to $k$;
    - **loss:** the mean KL over the sequence's tokens, plus $\lambda \cdot \mathcal{L}_{\text{Lp}}$ on the same tokens' scores.
5. **Mass penalty** $\mu \cdot \mathcal{L}_{\text{mass}}$ on the projected pieces, once per step.
6. **Average the gradients** over the 4 GPUs and take one Adam step (β = 0.9, 0.99). The learning rate follows a cosine from $10^{-3}$ down to $10^{-5}$: $\eta(t) = 10^{-5} + \tfrac{1}{2}(10^{-3} - 10^{-5})\big(1 + \cos(\pi t / (T - 1))\big)$.
7. **Project back to the exact sum.** Adam moves $U$ and $R$ freely, so $U R^\top$ drifts away from $W$. Hold one factor fixed and give the other the smallest correction that makes the sum exact again. For a wide matrix ($d_{\text{out}} \le d_{\text{in}}$), hold $U$ and set $R \leftarrow R + (W - U R^\top)^\top (U U^\top)^{-1} U$. For a tall matrix, hold $R$ and set $U \leftarrow U + (W - U R^\top)(R R^\top)^{-1} R$. Multiplying out shows $U R^\top = W$ afterwards in both cases. This runs in float64, because in float32 the earlier 13B run slowly drifted. The run stops if any matrix's error $\lVert U R^\top - W \rVert_F / \lVert W \rVert_F$ goes above $10^{-5}$.

Sizes per layer: Q, K, V and O have 1,024 pieces each, MLP up 3,072, MLP down 3,584. The pieces start from the same seed-0 projected split as the old Fisher baseline (`fisher_k4_5000`).

In the code but off: VPD's adversarial loss (`MATRYOSHKA_ADV_COEFF`) and the GIM score (`MATRYOSHKA_SENSOR=gim`).

---

## 3. What changed relative to MAttr (Arora et al., arXiv 2609.25518)

| | MAttr | This method |
|---|---|---|
| What is ranked | heads, neurons, SAE latents, or weight rows | rank-one pieces that sum exactly to each matrix |
| Where scores come from | one free learned number per unit, shared by the whole dataset | computed per token from the pieces (read × write) |
| What is trained | only the scores; the model is fixed | the pieces themselves ($u_c$ and $r_c$) |
| What "dropping" means | interchange: swap in the value from a second input | zero: the piece is simply left out |
| Loss | logit difference on pairs of inputs | KL to the real model on ordinary text |
| Budget $k$ | one per step; uniform by default, log-uniform as an option | one per sequence (256 per step), $\Pr(k) \propto k^{-1.5}$, stratified |
| Mask in the forward pass | sigmoid top-$k$ by default | hard top-$k$ (their "+hard" ablation) |
| Sparsity penalty | none | Lp on the per-token scores ($\lambda = 1$) |
| Weight penalty | none (the model is fixed) | mass penalty ($\mu = 10$) |

**Kept from old par-auto:** exact-sum pieces, the projection step, the read × write score, the Lp penalty.

**New relative to old par-auto:** the masked-forward KL at random budgets, and the mass penalty.

---

## Symbols at a glance

| Symbol | Meaning |
|---|---|
| $m$ | index of a weight matrix, 1 to 24 |
| $W_m$ | weight matrix $m$ |
| $c$ | index of a piece |
| $u_c$, $r_c$ | write and read directions of piece $c$ |
| $U$, $R$ | all of a matrix's write / read directions as columns; $W_m = U R^\top$ |
| $N$ | total number of pieces, 43,008 |
| $x$ | a token in its context |
| $h(x)$ | the input a matrix sees at token $x$ |
| $y$ | a next token sampled from the model |
| $g_y(x)$ | gradient of $\log P(y \mid x)$ at a matrix's output |
| $a_c(x)$ | score of piece $c$ at token $x$ (read × write) |
| $k$ | budget: how many pieces each token keeps |
| $S_k(x)$ | the $k$ highest-scoring pieces at token $x$ |
| $z_c(x)$ | hard keep mask, 1 if kept, else 0 |
| $\alpha_c(x)$ | soft keep mask (sigmoid top-$k$), used for gradients |
| $\theta_x$ | per-token threshold that makes the $\alpha$ sum to $k$ |
| $P$, $P_{S_k}$ | next-token distributions of the real and masked model |
| $p$ | exponent in the sparsity penalty, 1 → 0.5 |
| $\varepsilon$ | $10^{-6}$, keeps the Lp gradient finite at 0 |
| $\lambda$, $\mu$ | weights of the sparsity and mass penalties, 1 and 10 |
| $t$, $T$ | training step, total steps (100,000) |
