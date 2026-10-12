# CINDER: Curvature-Induced Network Decomposition with Exclusive Routing

*Research proposal, 7 September 2026*

## Executive summary

CINDER is a proposed parameter-decomposition method built around a different
question from ParFact, APD, SPD, VPD, NPEFF, or L3D:

> Which parameters must move together because the effect of one parameter
> changes when another parameter is changed?

The current ParFact family clusters parameters that have similar first-order
usage across prediction events. CINDER instead estimates the model's
**event-conditional parameter-interaction geometry**. It introduces a conceptual
multiplicative gate on every additive parameter atom, measures mixed curvature
between those gates using matrix-free derivative probes, turns the recovered
interactions into a sparse graph, and partitions that graph into bounded,
hard-owned components.

The slogan is: **cluster what bends together, not merely what lights up
together**.

The defining properties are:

1. Every decomposed parameter belongs to exactly one component. There are no
   soft ownership fractions and no duplicated machinery.
2. Components sum exactly to the original parameters.
3. The primary discovery signal is second-order and finite-radius interaction,
   not co-activation, gradient magnitude, or reconstruction of an attribution
   matrix.
4. The decomposition is not trained with random component ablations, output
   reconstruction losses, a causal-importance network, or any VPD loss.
5. Strongly interacting parameters are explicitly encouraged to remain on the
   same side of a component boundary.
6. Parameters that genuinely serve several mechanisms become bounded
   **interface components** in a dependency graph. They are not copied into
   every component and are not hidden in one unlimited background component.
7. If the model does not contain a clean modular partition at a requested
   resolution, CINDER reports high boundary interaction rather than fabricating
   sparse-looking components.

My literature search found work using Hessians for input-feature interactions,
loss-landscape analysis, pruning, and optimization, but not this combination of
parameter-gate curvature tomography, hard additive ownership, balanced graph
partitioning, and explicit interface components for post-hoc neural parameter
decomposition. That is a provisional novelty assessment, not a proof that no
closely related unpublished method exists.

## 1. What the current failures are telling us

The recent results are unusually diagnostic.

The diversity-penalized ParFact run did not collapse into one giant component.
Its largest candidate component held only about 0.7-1.2% of total weight mass,
and its capacity and duplicate-signature penalties were active. Nevertheless,
mean KL below 0.1 required retaining:

- 1,014 of 1,024 components (99.0%);
- 2,028 of 2,048 components (99.0%);
- 3,994 of 4,096 components (97.5%).

The entropy regularizer also did not produce genuinely sparse event usage. The
effective component counts were approximately 826, 1,592, and 2,865 at the
three resolutions. More importantly, changing from scalar parameters to a
complete SVD atom basis barely changed the qualitative KLKeep curves.

Together, these observations argue against three simple explanations:

- The main problem is not just one mega-component taking all the mass.
- The main problem is not just the SVD coordinate system.
- The main problem is not just insufficient diversity between usage signatures.

The deeper mismatch is between the fitted object and the desired property.
ParFact fits structure in an event-by-atom matrix of local first-order
attributions. KLKeep asks whether a set of finite parameter interventions
preserves a nonlinear computation. These are different objects.

For an event, components can have distinct attribution signatures while still
depending on the same downstream or upstream machinery. A soft membership
matrix makes this worse: the same scalar or atom can be fractionally present in
many nominal components. A per-event usage penalty can make the *readout* look
sparse without making the underlying parameter pieces causally independent.

The missing quantity is the interaction term: how much the effect of component
`c` changes when component `d` is also removed. None of the current ParFact
objectives directly minimizes that boundary interaction.

## 2. Design requirements

CINDER is designed against the observed failure modes rather than by adding
another term to co-factorization.

### 2.1 Exact and exclusive parameter accounting

For the decomposed parameter vector `theta`, CINDER must produce components
`C_1, ..., C_C` satisfying

```text
theta = C_1 + ... + C_C
```

and every scalar parameter must occur in exactly one component. A parameter may
not be split fractionally among candidates. There is no unexplained residual in
the primary version.

### 2.2 Nonlinearity must be a discovery signal

Nonlinearity should not appear only at evaluation time as a surprise. The
discovery signal must measure how parameter effects change under other
parameter changes.

### 2.3 No ablation-trained behavioral reconstruction

CINDER may use small, deterministic, nonzero perturbations to measure
derivatives. It may not optimize components through random masks, adversarial
masks, masked-model KL, or cross-entropy reconstruction. Actual ablations are
reserved for frozen post-hoc validation.

### 2.4 No forced sparse story

Event sparsity is an outcome to measure, not the primary loss. If a model uses
many coupled modules for most predictions, the method should reveal that. It
should not manufacture low entropy while leaving interaction boundaries
unchanged.

### 2.5 No mega-component escape hatch

Graph volume and weight mass receive hard upper capacities. Shared machinery is
represented by several bounded interface components and an explicit dependency
graph, not one unlimited background bucket.

## 3. Parameter gates and exact atoms

Let the parameters being decomposed be written as disjoint additive atoms

```text
theta = sum_j b_j.
```

Introduce one conceptual gate `m_j` per atom:

```text
theta(m) = theta_fixed + sum_j m_j b_j.
```

The trained model is `m = 1`; setting `m_j = 0` is exactly the ablation of atom
`j`. Gates are coordinates for measurement, not learned model parameters.

### 3.1 Scalar atoms for the Pile-67M target

The implemented full-model version uses one native scalar parameter per atom:

```text
b_j = theta_j e_j.
```

For the same 24 attention and MLP matrices decomposed by Goodfire, this gives
28,311,552 atoms. Each scalar receives one integer owner. It is never bundled
into a neuron, head, SVD direction, centroid, or learned low-rank unit before
partitioning. Embeddings, the tied LM head, and RMSNorm parameters remain fixed
only to preserve the scope of the VPD comparison.

This refinement matters conceptually. A structural unit is already a theory
about which parameters belong together; using it as the atom can bake the
answer into the experiment. Scalar atoms let nonlinear interaction evidence
make that decision. The price is that no dense ownership matrix or dense
interaction graph is possible. CINDER therefore sketches every scalar's
curvature row and partitions the sketches without ever allocating `P x C`.

Structural MLP/QK/OV atoms remain a useful cheap baseline, but they are not the
atoms used by the Pile-67M implementation described below.

## 4. Use an output map, not a VPD-style loss

A single predicted-token log-probability is a narrow view of a prediction. It
can make unrelated computations look identical whenever they support the same
scalar.

For prediction event `i`, freeze a set of high-probability output tokens under
the full model and form logit-margin contrasts. The implemented scalar pilot
uses the top prediction against each of the next four alternatives. A
higher-fidelity version can freeze the top 31 outputs, add one `other`
coordinate equal to the log-sum-exp of all remaining logits, center away a
common shift, and multiply by a fixed orthogonal contrast matrix:

```text
y_i(m) = Q_i^T Center(Top31PlusOtherLogits_i(m)).
```

`Q_i` can consist of deterministic Hadamard contrasts. Four contrasts are used
in the pilot; all 31 centered dimensions can be used on toys. The token set and
contrasts are fixed using the unmodified model, so there is no discontinuous
top-k operation inside differentiation.

This is a vector-valued description of what the model computes locally. It is
not KL, cross-entropy, or a target-label loss. No decomposition parameters are
optimized to reproduce it.

## 5. The core object: gate interaction curvature

For output contrast `o`, define the gate gradient and mixed gate Hessian

```text
a_i,o,j   = d y_i,o / d m_j
H_i,o,jk = d^2 y_i,o / (d m_j d m_k), evaluated near m = 1.
```

For scalar atoms,

```text
H_gate,i,o,jk = theta_j theta_k H_weight,i,o,jk.
```

For general additive atoms it is the bilinear form

```text
H_gate,i,o,jk = b_j^T H_weight,i,o b_k.
```

The interpretation is direct: `H_i,o,jk` measures how changing atom `k`
changes the functional effect of atom `j` on event `i`. Large mixed curvature
means that treating the atoms as independent components is locally wrong.

### 5.1 Why this quantity is tied to KLKeep-style ablation

After atoms have been assigned to components, let `g_c` gate component `c`.
For removing a set `R`, a second-order expansion of an output coordinate is

```text
Delta y_i(R)
  approximately
  - sum_(c in R) a_i,c
  + 1/2 sum_(c in R) H_i,cc
  + sum_(c<d; c,d in R) H_i,cd.
```

The last line is precisely why individually sensible components can fail when
many are removed together. If cross-component `H_i,cd` terms are small, the
effects of component removals approximately compose. CINDER's partitioning
objective tries to put large mixed terms *inside* components so they do not
cross ablation boundaries.

A usage-entropy loss does not constrain this term. That is the central
technical difference.

## 6. Finite-radius gradient transport

A Hessian at exactly `m = 1` is still local. CINDER therefore measures how
gradients are transported across small but finite gate displacements.

For a balanced signed probe `r_q` and radius `rho < 1`, evaluate

```text
T_i,o,q(rho) =
  [grad_m y_i,o(1 + rho r_q) - grad_m y_i,o(1 - rho r_q)] / (2 rho).
```

By the fundamental theorem of calculus,

```text
T_i,o,q(rho)
  = (1 / (2 rho)) integral_(-rho)^rho H_i,o(1 + t r_q) r_q dt.
```

Thus `T` is an integrated Hessian-vector response along a symmetric local
path. It reduces to an ordinary Hessian-vector product as `rho -> 0`, while a
larger radius captures curvature drift and some higher-order behavior.

Recommended initial radii are a local HVP and one finite radius selected from a
small calibration sweep, likely around 0.03-0.10. The radius should be low
enough that most full-model top-token identities remain stable. Components are
never zeroed during discovery.

The probes should use deterministic balanced signs rather than sampled
component masks. They attenuate some atom gates and amplify others by the same
small amount. They are measurement directions, not training examples.

True automatic derivatives should be used here rather than GIM. GIM modifies
the backward pass and need not define the gradient of any scalar field, whereas
CINDER relies on a coherent, approximately symmetric Hessian.

## 7. Curvature tomography without a dense Hessian

For 28M scalar atoms, an event-specific Hessian is impossible to materialize.
CINDER only needs matrix-vector products and compressed interaction
neighborhoods.

### 7.1 Two-axis sketches

Use a small family of event sign hashes `xi_b(i)` and parameter probe vectors
`r_q`. Stream the following accumulator for every atom `j`:

```text
Y_b,q,o,rho,j =
  (1 / sqrt(N)) sum_i xi_b(i) T_i,o,q(rho)_j.
```

This is a randomized sketch of the full tensor
`event x output-contrast x atom x partner-atom`. The event signs prevent common
high-frequency behavior from erasing rare event-conditional interactions. The
parameter probes compress each Hessian row without constructing the row.

For a candidate pair `(j, k)`, decode a bucketed mixed interaction by

```text
Hhat_b,o,rho,jk = (1 / Q) sum_q Y_b,q,o,rho,j r_q,k.
```

Then define a non-cancelling interaction strength

```text
w_jk = RMS_(b,o,rho)(Hhat_b,o,rho,jk) / sqrt(s_j s_k + epsilon),
```

where `s_j` and `s_k` are robust per-atom sensitivity scales. RMS over event
hashes preserves interactions whose sign changes by context; a plain dataset
mean would cancel them.

The same event sketches of first derivatives are retained for interpretation,
candidate-neighbor retrieval, and later event routing, but first-order
co-usage is not the primary edge weight.

### 7.2 Recovering only a sparse graph

There is no reason to store every pair. Each atom keeps its strongest 32-64
candidate edges. Candidates come from three sources:

1. heavy-hitter recovery from repeated CountSketches of curvature rows;
2. approximate neighbors in transported-curvature signatures;
3. architecture-adjacent and first-order co-used atoms, included as a guard
   against sketch misses.

Edges must survive two independent sketch seeds or a held-out event shard.
This stability filter is important: noisy Hessian estimates otherwise create
spurious bridges between communities.

Matrix-free Hessian-vector products have been practical since Pearlmutter's
method, which computes `Hv` at approximately gradient cost without storing
`H`. Finite-radius transport can alternatively use two ordinary backward
passes, avoiding higher-order autograd at twice the first-order cost.

## 8. Hard graph partitioning, not soft factorization

Let `G = (V, E, w)` be the sparse interaction graph. CINDER finds a hard
partition `z_j in {1, ..., C}` minimizing cross-boundary interaction:

```text
min_z  sum_((j,k) in E) w_jk * 1[z_j != z_k],
```

with normalized-cut scaling and two hard capacity constraints:

- no component may own more than twice the uniform share of decomposed
  squared weight mass;
- no component may own more than twice the uniform share of interaction-graph
  volume.

Unlike the current soft capacity penalty, these are constraints, not small loss
terms. They make a mega-component infeasible. Normalized cut discourages the
opposite degeneracy of peeling off many tiny components.

A multilevel partitioner first contracts unambiguously strong edges, partitions
the contracted graph, and then refines atom assignments by local boundary-gain
swaps. The resolutions should be nested:

```text
C=1024 communities -> split each weak boundary -> C=2048 -> C=4096.
```

This produces a hierarchy rather than three unrelated fits and lets us ask at
which resolution each mechanism naturally separates.

No atom has a probability distribution over owners. After partitioning,

```text
C_c = sum_(j: z_j = c) b_j,
```

so the components are disjoint and sum exactly to the target parameters.

## 9. Interface components: representing shared machinery honestly

A flat partition is sometimes the wrong ontology. A normalization pathway,
generic attention transport, or residual-writing unit may genuinely interact
with several otherwise distinct computations. Assigning such an atom to one
leaf component is arbitrary; copying it into every leaf recreates the current
sharing problem; moving everything into one background component creates a
mega-component.

CINDER detects these atoms using **boundary entropy**. For atom `j`, form the
distribution of its incident interaction weight over provisional communities.
If that distribution has high entropy and substantial total weight, `j` is a
junction rather than a leaf member.

Junction atoms are exclusively repartitioned into bounded interface
components according to their neighboring-community signatures. The output is
therefore:

- disjoint leaf components;
- disjoint interface components;
- a dependency graph saying which leaves use which interfaces.

An event explanation begins with selected leaf components and then takes the
required interface closure. Every interface retained in that closure counts in
the component and mass budget. Nothing is silently always on, and no parameter
is duplicated.

If an interface is genuinely global, it will be selected frequently. The mass
caps force global infrastructure to be split into several coherent interface
components rather than one enormous residual. This gives an honest distinction
between monosemantic leaves and polysemantic shared infrastructure.

## 10. Event routing after the partition is frozen

CINDER does not train a causal-importance network. It derives event scores from
the frozen output map.

For each event, aggregate gate gradients into components and estimate the
diagonal component curvature with a few component-space Hutchinson probes. For
component `c`, use the second-order removal score

```text
score_i,c = norm_o(-a_i,o,c + 0.5 H_i,o,cc).
```

The initial leaf ranking uses this score. Interface closure is then added using
the dependency graph. A greedy version can rank a leaf by score divided by the
new mass added by its not-yet-retained closure.

This router is still an approximation, but it is materially better matched to
the decomposition than the current `gradient dot component` score:

- it is multi-output;
- it includes self-curvature;
- cross-component curvature was explicitly minimized during partitioning;
- shared dependencies are included rather than assumed away.

Actual KLKeep, single-component ablation, and selectivity are measured only
after all graph and routing choices are frozen.

## 11. Why this could fix the observed failure

### 11.1 It attacks the cross-term directly

Current methods can reduce event-usage entropy while leaving the mixed
component Hessian untouched. CINDER makes that mixed term the object being cut.

### 11.2 Hard ownership eliminates diffuse copies

No SVD atom, unit, or scalar can appear at 0.1% strength in hundreds of
components. Keeping more components cannot gradually reconstruct every weight
through fractional accumulation; a parameter is either present through its one
owner or absent.

### 11.3 Multi-output contrasts reduce scalar aliasing

Two atoms supporting the same predicted-token log-probability can have very
different effects on the rest of the output distribution. Their centered-logit
transport signatures can distinguish them.

### 11.4 Finite-radius transport detects curvature drift

The difference between local HVP and finite-radius transport is itself a
diagnostic. Atoms whose interaction signatures change strongly with radius are
unsafe to separate using a purely local approximation and receive stronger
weld edges.

### 11.5 Interfaces model reuse instead of denying it

If several mechanisms share infrastructure, the correct representation may be
a sparse dependency graph rather than mutually independent flat bags of
weights. CINDER keeps the additive decomposition exact while making that reuse
explicit.

### 11.6 It can say "no"

If cross-boundary interaction remains high under the mass caps, then the model
does not support `C` clean components in the chosen atomization and event
distribution. This is more informative than obtaining attractive entropy
statistics with a flat KLKeep curve.

## 12. How CINDER differs from nearby methods

| Method | Primary fitted object | Nonlinearity used in discovery? | Ownership | Key difference from CINDER |
|---|---|---:|---|---|
| Current ParFact | Event-by-atom attribution allocation | Mostly no | Soft/fractional | CINDER partitions mixed gate curvature, not attribution co-variation. |
| Diversity ParFact | Same allocation plus entropy/capacity/signature losses | Mostly no | Soft/fractional | Its penalties change usage statistics, not cross-component interaction energy. |
| Scalar ownership | First-order scalar-attribution fingerprints | No | Sparse/soft plus background | CINDER's primary signal is how effects change under other parameter changes. |
| Direction/ICA decomposition | Sparse activation directions | Only at evaluation | Exact fixed directions | CINDER is not basis sparsification and permits cross-layer interaction communities. |
| Unit-ablation/effect clustering | Profiles of individual finite ablation effects | Yes, one unit at a time | Hard units | Single-unit effects omit mixed interactions; CINDER estimates off-diagonal coupling without zero ablations. |
| `zcofac67` | Signed unit attributions, optionally a diagonal HVP correction | Limited | One owner per atom | CINDER uses off-diagonal, event-conditional, multi-radius curvature as the primary graph. |
| NPEFF | Per-example Fisher factors | Local output sensitivity | Learned directions | It discovers processing strategies, not a hard exhaustive additive partition. |
| L3D | Sparse reconstruction of local loss gradients | Local | Learned low-rank directions | It reconstructs gradients; CINDER cuts interaction boundaries between fixed additive atoms. |
| APD | Minimal-description parameter components using attribution | Primarily local attribution | Learned additive components | CINDER has no component-training objective and no simplicity/usage loss. |
| SPD/VPD | Behavior under stochastic/adversarial component masks | Yes | Learned rank-one subcomponents | CINDER never trains on masked-model reconstruction and never learns a causal-importance network. |
| Integrated Hessians | Pairwise interactions between input features | Yes | Not a parameter decomposition | CINDER applies transported mixed derivatives to parameter gates and constructs exact additive components. |
| H-Sets | Hessian-guided interacting input-feature sets for images | Yes | Input feature sets | It does not perform exhaustive parameter ownership, tensor sketching, or a component dependency graph. |

The closest conceptual precedents are Integrated Hessians and the 2026 H-Sets
work, because both treat mixed derivatives as interaction evidence. Their
objects are input features and explanations of individual predictions. CINDER's
object is the parameter-gated model itself, and its output is an exhaustive,
hard, cross-layer additive parameter decomposition.

## 13. Implemented scalar pilot on the 67M target

### 13.1 Pilot configuration

- Scope: the same 24 matrices as VPD.
- Atoms: all 28,311,552 individual scalar parameters in that scope.
- Prediction positions: 65,536 Pile positions stratified across the first 100M
  tokens and held in an in-memory event bank.
- Output map: four frozen top-token margin contrasts per position.
- Event sketches: four independent deterministic sign hashes, differentiated
  together with batched VJPs.
- Parameter probes: eight dense Rademacher gate directions.
- Radius: symmetric finite transport at `rho = 0.02`.
- Scalar feature width: 32 transported-curvature coordinates.
- Outputs: nested 1,024, 2,048, and 4,096 hard partitions.
- Discovery ablations: none.

This pilot deliberately favors a complete scalar experiment over a richer
structural-atom pretest. Thirty-two dimensions will not recover a literal
28M-node adjacency matrix. Instead, they preserve each scalar's compressed
interaction-neighborhood fingerprint. The partition therefore groups scalars
whose nonlinear response neighborhoods are similar. Increasing event sketches,
parameter probes, radii, and held-out seed replication is the direct fidelity
upgrade if the pilot is promising.

### 13.2 Storage

The implemented `28.3M x 32` FP16 sketch is about 1.8GB. Ownership is one int16
per scalar, about 57MB, rather than a dense parameter-by-component tensor. The
Pile bank is copied once from the existing uint16 corpus into pinned host RAM
and reused for every plus/minus probe. Thus file I/O is outside the curvature
hot loop; derivative collection, not data delivery, is the bottleneck.

### 13.3 Compute strategy

The pilot uses symmetric finite differences of ordinary gate gradients. Four
event/output sketches share each forward graph through batched VJPs. This is
more memory-hungry than one backward but avoids second-order graph retention,
directly measures a finite radius, and checkpoints after every parameter probe.

After collection, globally whitened curvature fingerprints are recursively
split by a balanced two-seed hyperplane. Each split is chosen near half of a
mixture of scalar count and squared-weight mass, with a hard one-third count
floor. This yields a nested tree and prevents both a scalar-count mega-component
and the usual squared-weight-mass escape hatch. It is the scalable surrogate
for the full normalized-cut graph in this first global scalar run.

## 14. Experimental sequence

### Stage A: exact toy validation

On superposition and compressed-computation toys, construct the full gate
Hessian exactly. Test whether strong mixed edges recover known mechanisms and
whether cut energy predicts KLKeep. This stage can falsify the core idea before
any large run.

Required comparisons:

- first-order co-usage graph;
- exact mixed-curvature graph;
- local HVP sketch;
- finite-radius transport sketch;
- random balanced partition.

### Stage B: Pile-67M global scalar pilot

Build scalar CINDER at `C = 1024, 2048, 4096` and compare against:

- paper-faithful ParFact;
- diversity-penalized ParFact;
- hard signed `zcofac67` ownership;
- clustering of measured single-unit effect profiles;
- VPD as an external reference, not a training signal.

Use exactly the same held-out events and KL direction for fair curves.

### Stage C: scalar fidelity refinement

Repeat the global scalar run with independent event/probe seeds and more sketch
dimensions. Measure partition stability and held-out transported-curvature cut
energy. Promote only stable candidate neighborhoods to explicit sparse edges,
then apply multilevel constrained graph refinement.

### Stage D: interface refinement

On the scalar graph, identify high-boundary-entropy junction parameters and
repartition them into bounded interface components. Compare ordinary flat
KLKeep with dependency-closure KLKeep so shared machinery is counted rather
than silently retained.

## 15. Headline metrics and falsification criteria

### Discovery metrics

- Held-out cross-boundary curvature energy.
- Ratio of within-component to between-component interaction.
- Fraction of edge volume assigned to interface components.
- Component weight-mass and graph-volume distributions.
- Stability of the nested partitions across event samples and sketch seeds.
- Curvature-sketch recovery error against exact small-subgraph Hessians.

### Behavioral metrics

- KLKeep versus retained component count and retained weight mass.
- KLDrop for the highest-ranked components.
- Single-component selectivity on high-score versus matched control events.
- Additivity error: measured joint ablation effect minus the sum of measured
  single effects.
- Routing quality with and without the self-curvature term.
- Number and mass of interface components added by dependency closure.

### Initial go/no-go bar

The method should not be judged successful merely because its graph objective
decreases. On the 67M model it should achieve all of:

1. at least a 5x reduction in held-out boundary curvature versus a random
   capacity-matched partition;
2. materially lower joint-ablation additivity error than first-order hard
   ownership;
3. a large KLKeep movement from the current 97.5-99.0% retention requirement;
4. no component above the hard 2x mass or graph-volume capacities;
5. stable communities across independent sketches.

A reasonable first behavioral target is mean KL below 0.1 while retaining less
than half of the components. The ambitious target is below one quarter. If the
boundary metric improves but KLKeep does not, third- and higher-order effects or
the chosen atomization are the likely bottleneck. If boundary energy itself
cannot be reduced, a flat additive decomposition at that `C` is probably not
supported by the model.

## 16. Important failure modes

### Curvature can still be too local

Finite-radius transport mitigates but does not eliminate this. Radius drift
must be reported, and exact finite ablations remain the final test.

### Interaction graphs can become globally dense

This may be real rather than an estimator failure. Spectral decay, graph
conductance, and heavy-edge stability should be checked before forcing a large
number of communities.

### Hessian estimates can be noisy

Use independent event and probe seeds, median-of-means aggregation, held-out
edge confirmation, and exact pair checks on a small sample.

### Parameter symmetries remain

Gate coordinates are more intervention-relevant than raw Euclidean parameter
coordinates, but transformer rescaling and rotation symmetries do not vanish.
Structural micro-atoms should be compared with scalar and learned-direction
variants on toys.

### The output sketch can omit behavior

Increase the number of logit contrasts, compare top-set sizes, and include a
small number of residual-vocabulary sketches. The decomposition should be
stable to these choices.

### Interfaces can become a disguised background

They therefore have the same hard capacities as leaves, are never silently
retained, and count fully in KLKeep budgets. Report their activation frequency;
a collection of nearly always-on interfaces is evidence that the model's
computation is not very sparse.

## 17. Research basis and novelty audit

This proposal deliberately borrows mathematical tools while changing the
object and scientific objective.

- [APD](https://arxiv.org/abs/2501.14926) introduced additive parameter
  components optimized for faithfulness, sparse use, and simplicity.
- [SPD](https://arxiv.org/abs/2506.20790) replaced APD's brittle attribution
  procedure with stochastic causal masking and directly tested whether pieces
  could be removed.
- [VPD](https://www.goodfire.com/research/interpreting-lm-parameters) added
  adversarial reconstruction, stochastic reconstruction, importance and
  frequency minimality, and parameter-faithfulness losses for the same 67M
  target. CINDER uses none of those losses.
- [L3D](https://arxiv.org/abs/2504.00194) learns sparse low-rank subnetworks
  that reconstruct sample loss gradients. CINDER does not reconstruct a
  gradient field and does not learn parameter directions.
- [NPEFF](https://arxiv.org/abs/2310.04649) factorizes per-example Fisher
  matrices into processing-strategy factors. It does not produce an exhaustive
  hard additive parameter partition.
- [Integrated Hessians](https://arxiv.org/abs/2002.04138) established mixed
  derivatives integrated along a path as a useful way to attribute feature
  interactions. CINDER adapts the interaction idea to parameter gates and uses
  it for graph construction rather than per-input feature explanation.
- [H-Sets](https://arxiv.org/abs/2604.22045)
  is the closest recent conceptual neighbor I found: it uses input Hessians to
  build interacting image-feature sets. It does not address additive model
  parameters, exclusive ownership, matrix-free tensor sketches, or shared
  interface modules.
- [Pearlmutter's HVP method](https://doi.org/10.1162/neco.1994.6.1.147) makes
  the curvature measurement matrix-free.
- Work on neural Hessian structure finds substantial low-rank and Kronecker
  structure in practice, which makes sketching plausible, though not
  guaranteed for this exact tensor
  ([Wu et al., 2020](https://arxiv.org/abs/2010.04261)).
- [Normalized cuts](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf)
  provide the generic graph-partitioning foundation; CINDER's novelty is the
  graph being partitioned and the parameter/dependency interpretation of its
  communities.

The exact proposed combination I could not find in the surveyed literature is:

> event-conditional, multi-output, finite-radius mixed curvature in parameter-
> gate coordinates -> matrix-free tensor sketch -> sparse heavy-interaction
> graph -> hard capacity-constrained additive parameter partition -> bounded
> interface-component dependency graph.

That is the core research claim worth testing.

## 18. Bottom line

ParFact asks which parameters are used together. CINDER asks which parameters
**cannot be separated without cutting a nonlinear dependency**.

That shift is motivated directly by the evidence: usage diversity, basis
changes, and balanced weight mass did not move KLKeep. A harsher entropy
penalty may force sparser signatures, but it still does not target the mixed
terms responsible for joint-ablation failure. CINDER does.

The proposal is riskier and more expensive per event than first-order
co-factorization, but it is also falsifiable at small scale and computationally
distinct from VPD. If it succeeds, it yields exact, disjoint parameter pieces
plus an explicit map of their shared dependencies. If it fails, it tells us
something stronger than another flat KL curve: the target model may not admit a
sparse flat additive decomposition at the requested granularity without direct
causal training or a richer, graph-structured notion of mechanism.
