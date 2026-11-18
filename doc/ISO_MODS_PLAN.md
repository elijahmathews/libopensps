# Isochrone modifications free per model on the batched SSP path (#45)

Design and delivery plan. Written before the implementation; each stage records what was measured
or tested when it lands. Companion documents: `doc/GPU_PLAN.md` (GPU rules), `doc/MIXED_PRECISION.md`
(FP32 kernel rules), `doc/TUNING.md` (hardware-dependent sizes), `src/physics/opensps_ssp_operator.f90`
(the operator S, its module notes).

## 1. Goal and constraints

A batch (`compute_csp_batch`, `opensps_context_compute_csp_batch*`) lets the IMF and the metallicity
vary per model through the template operator (`ssp_ops`). It does not let the isochrone modifications
vary: `fbhb`, `sbss`, `agb`, `pagb`, `redgb`, `delt`, `dell` (and the context's TP-AGB normalisation,
which stays shared). (As written, before the implementation: a batch whose models differ in any of them returned
status 2. With #45 delivered they are free per model on the free-modifications operators; `doc/HOT_PATHS.md`
puts them in the classification of per-model parameters.)

Requirements, from the project owner:

1. **Gradients are part of the feature.** Every parameter made free has a row of `params_bar`, checked
   against finite differences, in the same stages as the forward model.
2. **No cost where nothing varies.** The default case (modifications fixed, the resident basis) and the
   IMF-only case must keep their code, outputs (bit for bit) and speed. Gate: identical outputs and
   `benchmark_csp_batch` timings within noise (about 1 %) before and after, CPU and R9700.
3. **No new hot path.** The number of age-sum kernel families must not grow (section 2).
4. **High test coverage** of every new line: oracle tests, gradient tests, status paths. Patch coverage
   is checked locally (gcov) before each pull request.
5. A100 verification happens once all stages are in; fixes it finds go on top of the stack.

Out of scope: `evtype`, `masscut`, `agb_dust` stay shared by the batch (status 2 if they differ), as does the
context's `tpagb_norm_type`. They are selections of which stars are summed, not factors.

## 2. The hot paths: census and the rule for this work

The SSP stage of `compute_csp_batch` (the age sums that build each model's spectrum from the SSPs) picks one
of five families by a priority chain on what is free in the batch. The other stages (dust, nebular, AGN, IGM,
smoothing, photometry) are per-batch physics switches; the ones that fork on a free parameter (redshift per
model: `apply_igm_z` or `apply_igm`, the per-model observed frame) fork independently of the SSP stage, so they add
routines, not products of them.

| Family (chosen when) | Forward | Reverse |
|---|---|---|
| fixed-Z basis (nothing free) | `age_sums` (101 lines) | `backward_age_sums` (171) |
| free-Z basis (`logzsol`) | `age_sums_z`, `_generic`, `_hip` (259) | `backward_age_sums` |
| free-Z with [alpha/Fe] slabs (`logzsol`, `afe`) | `age_sums_z4` (190) | `backward_age_sums_afe` (109) |
| weights over Z (`ssp_weights_z`) | `age_sums_w` (56) | `backward_age_sums_w` (35) |
| operator (`ssp_ops`: IMF free, optionally Z) | `ssp_batch_age_sums_z` and kernels (about 890) | `ssp_batch_age_sums_adjoint`, `adj_*` (about 1240) |

Each family exists in several backend variants (generic, LLVM flang, nvfortran, HIP). Combinations that are not
implemented return status 2: the operator has no [alpha/Fe] axis and refuses weights over Z, SSP-level nebular
emission, X-ray binaries and the LSF. The concern that free-parameter combinations keep adding kernels is
therefore justified inside the SSP stage; it is not a combinatorial explosion across stages.

The structure that already exists, and that this work keeps:

* **General path:** the operator. `x(p, m) = w_IMF(p, m) w_sfh(age(p), m)`, `V(e, m) = sum_p S(e, p) x(p, m)`,
  the template product. Anything free runs here.
* **Specialization by precomputation:** the resident basis is the operator evaluated once for fixed parameters.
  It is used whenever the IMF and the modifications are fixed.

**Rule:** a newly freeable parameter enters the general path only, as data (per-model coefficients, virtual
columns, per-model operator selection), and never as a new family. Specializations for a parameter that is
fixed across a batch are optimizations of that path (section 6), selected on the host, outside the kernels.

Also found, and fixed first (stage 1): the basis path never compared a model's modifications with the
basis's, so a Fortran caller got the basis's silently (the C API refused them).

## 3. What each parameter does

From `modify_horizontal_branch`, `modify_giant_branch` and `apply_blue_stragglers`:

| Parameter | Effect on the isochrone of each age | Kind |
|---|---|---|
| `agb` | TP-AGB (phase 5) weights x `agb` (x Villaume factor, Padova) | weight scaling of a group of points |
| `redgb` | phases 2-4 weights x `redgb`; phase 5 as well (combined with `agb`) | weight scaling |
| `pagb` | post-AGB (phase 6) weights x `pagb` | weight scaling |
| `fbhb` | above a threshold (MIST 1e-4, Padova 1e-3) and for ages >= `BHB_SBS_TIME`: the HB star j keeps weight (1 - fbhb) (MIST: and is moved to the red-clump temperature, so its template changes), and blue stars i, copies of j at other temperatures, get weight fbhb x the fraction of j (Padova: 10 per HB star) | structural switch + scaling of derived stars |
| `sbss` | for ages >= `BHB_SBS_TIME`, `sbss > 1e-3`: blue stragglers, weight (sbss / 20) x (sum of the HB weights before the modification) | derived stars, weight from a per-age reduction |
| `delt`, `dell` | TP-AGB (phase 5) log T and log L shifted | changes the templates of those stars |

`delt` and `dell` are not linear in a fixed operator (the template coefficients are piecewise-bilinear functions
of the shifted log T and log L, and the library's snapping and dust-shell corners follow them). The others are
linear in the IMF weights with coefficients that are simple functions of the parameter.

## 4. Design

### 4.1 Factors (`fbhb`, `sbss`, `agb`, `pagb`, `redgb`): one operator set per group, combined with coefficients (as built)

The first design of this section (coefficients on groups of points and virtual columns, applied in the point-weight stage)
needed edits to the weight kernels, one of them a HIP kernel. What was built leaves every kernel alone:

* **The stars of an isochrone are partitioned into groups** by which factors scale their weight (`MOD_GRP_*` in
  `opensps_ssp_operator`): plain (no factor), TP-AGB (`agb * redgb`), post-AGB (`pagb`), phases 2-4 (`redgb`), the horizontal-branch
  stars at ages with the HB modification at their own temperature (with and without `redgb`), the original HB stars moved to
  the red-clump temperature (MIST, BaSTI), the blue HB stars, and the stragglers. `ssp_operator_build(ctx, pset, op, group)`
  builds the operator of one group: the group's stars only, from the reference modifications (identity; fbhb 0.5 and sbss 20 for
  the groups of the HB modification and the stragglers) with the reference factors divided out of the entries, and only the native
  points the entries use. The groups partition S: their sum with coefficient 1 is the operator of the reference.
* **The coefficients** `ssp_mod_coefficients(pset, pdva)`: TP-AGB `agb * redgb`; post-AGB `pagb`; phases 2-4 `redgb`; Padova HB
  `1 - ind fbhb` (times `redgb` when the star is in phases 2-4), blue `ind fbhb`; MIST and BaSTI HB `1 - ind`, moved
  HB `ind (1 - fbhb) redgb`, blue `ind fbhb`; stragglers `sbss / 20` above `sbss = 1e-3`; `ind` is `fbhb` above its threshold (1e-4;
  1e-3 for Padova), and the ages below `BHB_SBS_TIME` are in the plain and RGB groups. Their derivatives with respect to the factors
  are `ssp_mod_coefficient_derivatives`.
* **Forward** (`ssp_batch_age_sums_mods`): the unchanged `ssp_batch_age_sums_z` once per group set, its outputs combined per model
  with the coefficients: the young and old age sums, the surviving mass and the ionizing photon rate are linear; L_bol is the
  logarithmic blend, over the two bracketing metallicities, of the linear sum over groups of the groups' `sl` (which
  `ssp_batch_age_sums_z` returns per metallicity pass on request); the remnants, which do not depend on the factors, come from
  the plain group alone. The partition adds no cancellation: every term is a positive weight times a spectrum.
* **Reverse** (`ssp_batch_age_sums_adjoint_mods`): per group the unchanged adjoint with the cotangents as given, except L_bol's,
  and its outputs (SFH weights, IMF parameters, escaping fraction, metallicity weight) weighted by the coefficient; the cotangent
  of each coefficient is the group's forward, recomputed, against the cotangents. L_bol's part is formed once from the groups'
  sums, `d L / d sl(g, z) = L cz_z c_g / L_z`, and enters the group passes through one new input of `adj_lq` (the factor of the point
  weights; the accelerated HIP kernel does not take it and is bypassed in this mode). The gradient rows are the chain rule from
  the coefficients' cotangents.
* **The isochrone arrays** (`NM = 2000` points per age; legacy FSPS stops where the HB modification would overflow them: MIST's
  2nd and 12th metallicities): the operator of such a metallicity is built without the HB modification at those ages
  (`hb_blocked`), and a model that needs it returns status 2.
* **No cost where nothing varies:** the default operator and the resident basis are the code that existed; the group operators
  are a separate build, requested by `opensps_context_build_ssp_ops_free_mods`. Verified: unit and regression suites unchanged and
  identical outputs on those paths.

Checked against: the group operators combined with their coefficients against `generate_ssp_grid` with the factors on (spectra
to 4e-15, MIST and Padova, factors at and below their thresholds); a 12-model batch against each model alone through the resident
basis built with its factors (spectra 3e-15, magnitudes 7e-15, mass and L_bol 1e-15); the five gradient rows against central finite
differences with full physics, with a free Kroupa IMF, and with a free metallicity.

### 4.2 `delt` and `dell`: one small operator per distinct pair

For a model with shifts `(delt, dell)` the TP-AGB stars use other templates. The operator of the TP-AGB points
only is a small sub-operator per distinct pair of values in the batch, built on the host with the existing build
(`ssp_operator_build` restricted to phase 5, about 3 % of the points per the issue), cached by pair, and selected
per model exactly as the per-metallicity operators are (concatenated tables with offsets per variant, one launch
serves all). The phase-5 columns of the main operator are then unused for models with a shift, which is a
coefficient of 0 in the group of 4.1. The L_bol of the shifted stars is exact: `L = 10^(logL + dell)`.

**As built (PR 3).** Each distinct nonzero pair is one more group of the batch: `ssp_operator_build(group = TPAGB,
shift_t, shift_l)` builds the TP-AGB stars at the shifted log T and log L (the shifts enter through the isochrone
modification, as in legacy FSPS), the set is cached per pair (`ssp_mods_prepare_pairs`), and the models at the pair take
the TP-AGB coefficient `c_2` in that group's slot instead of the shared TP-AGB group (the forward and the reverse
orchestrators are those of 4.1 with `ng = MOD_NGRP + n_ext` groups). Batches with more than 8 distinct pairs (4 when the
set covers every metallicity: one pair's operators for MIST's 12 metallicities are ~1 GB at the regression data's size,
measured) return status 2; a pair no batch uses is evicted when a new pair needs the slot.

Gradients: not an analytic derivative of the template coefficients but central differences of the group's outputs, taken
in the reverse pass: for each pair in the batch the TP-AGB group is built at the pair shifted by `+-TP_FD_STEP` in `delt`
and in `dell` (four sets, one at a time, freed after use: the memory of one set), the cotangent of every model is formed
against that group's outputs (spectra, surviving mass, ionizing photons, L_bol sums) exactly as for the factors, and
`d L / d delt = c * (cot(+) - cot(-)) / (2 TP_FD_STEP)`. **The step is 5e-4 dex (1e-3 with FP32 kernels)**, not
small: the TP-AGB spectra switch templates at thresholds, so the loss is a staircase in `delt` (measured with MIST, one
model, by scanning the loss at 2e-6 dex: steps of 1e-7 to 1e-4 in L, 50 to 150 of them in 2e-3 dex, one every ~2e-5 dex)
and a central difference over less than ~2e-4 dex returns a step (5.07, -7.56, -1.62 at 1e-5 against 2.1, -2.9, -1.4 from
2e-4 to 1e-3). A gradient-based fit sees the response averaged over some tens of steps, which is also what a finite
difference at the scale a fit explores gives. `dell` shifts L only and is smooth (agreement 1e-9 at 1e-5, 2e-7 at 5e-4).
The tests compare the rows with the loss's own central difference at the same step: they differ at O(step) relative
because the adjoint linearizes the loss in the group's outputs (4e-5 to 9e-5 for `delt`, 2e-7 to 4e-7 for `dell` in
FP64; tolerance 1e-3, and 0.2 in FP32 where the rows are 1e-5 to 1e-3 and the finite difference's rounding is a few
percent). Cost: a host build of the pair's set (and of its four shifted sets with gradients) per distinct pair not
cached, so a batch with continuously varying shifts is slow by design and a batch of fits with all-distinct shifts
cannot use this path beyond the cap. The all-metallicity group of the gradient test has the shifts at zero (the pair
sets would take 1 GB each there, on top of its 4.7 GB); they are tested with one metallicity, and the C test builds
the operator at `zmet = 10` (the default `zmet = 1` is MIST's most metal-poor grid, which has no TP-AGB stars, so
neither `agb` nor the shifts change anything there).

### 4.3 Fixed IMF, free modifications: the shared IMF weights

With every model on the same IMF parameters the point weights `w_IMF(p)` are identical across the batch. The
per-model IMF integrals are about 85 % of the per-model weight cost, so computing them once and broadcasting
makes "modifications free, IMF fixed" cost about the fold only, not the free-IMF price (about 40 us per model on
the R9700 in FP32, against about 12 for the basis). Selected on the host by comparing the models' IMF parameters;
the same specialization applies to a free-IMF batch that happens to share its IMF. Gradient: the IMF rows are the
sum over the models' common weights, as the reverse pass does when models share an IMF.

**As built (PR 4), the forward pass only.** `ssp_batch_age_sums_z` finds the models of a chunk with the same IMF parameters
(all of `pm`: the parameters and the limits) and metallicity (`chunk_unique_imf`, the last 16 distinct ones compared), has
`imf_pieces_multi` (new argument `part`: 1 = the weights, 2 = the sums over the ages) compute the weights of the distinct
ones into a scratch sized by their number, copies them to the columns of all the models (`share_expand`) and runs the
sums over the ages for every model. The columns are copies of one computation, so the numbers are the same bit for bit
(`check_share_imf`: spectra, magnitudes, mass and L_bol, with all models on one IMF, with groups sharing one and with every
model on its own, and with the metallicity free per model). It is chosen per batch: when no two models share an IMF
(`imf_shared_models`) the path is the one it always was, and batches smaller than `SHARE_IMF_MIN_BATCH` (default 1024,
`doc/TUNING.md` M6) take it too. The reverse pass is not changed: it computes the weights of each model (its IMF cotangents
are per model and sum correctly over models that share an IMF). Measured on the R9700: 3 - 13% per model from N = 1000
(`doc/GPU_RESULTS.md`); on the CPU ~1%, as the weights are a small part of its cost. `ssp_batch_share_imf` switches it off
for tests and benchmarks (`benchmark_csp_batch`: configs `_imfs`, `OPENSPS_BENCH_NOSHARE=1`).

## 5. C API and bindings

* Names `fbhb`, `sbss`, `agb`, `pagb`, `redgb`, `delt`, `dell` are accepted per model when the context's
  operators were built free-modifications (`is_ssp_sensitive_float_param` and the IMF exception extend to them);
  otherwise status 2 as before.
* `params_bar` rows are added in stages (`BGRAD_FBHB`, `BGRAD_SBSS`, `BGRAD_AGB`, `BGRAD_PAGB`, `BGRAD_REDGB`; then
  `BGRAD_DELT`, `BGRAD_DELL`): `OPENSPS_NBGRAD` 37 to 42 to 44. Stage 1 first adds `opensps_context_compute_csp_batch_grad_rows`
  (#98): a caller that gives its row count gets status 3, not an overrun, when the library has more rows.
* The Python binding is a separate repository: it must use the rows entry point and `opensps_get_nbgrad()`.

## 6. Delivery: four pull requests, stacked

Each is a squash-mergeable change with green CI; the first is against `main`, each other against the previous
branch.

1. **Groundwork and this document.** `doc/ISO_MODS_PLAN.md`; the Fortran-level guard on the basis path; the
   rows entry point (#98), changelog and C API text; tests of both.
2. **Free factors.** The free-modifications operator (build, groups, virtual columns, coefficient table), the
   forward kernels' extension, the reverse reduction and rows for `fbhb`, `sbss`, `agb`, `pagb`, `redgb`; the C API
   names; oracle tests per parameter against `generate_ssp_grid` (MIST and Padova, ages on both sides of
   `BHB_SBS_TIME`, `fbhb` at its threshold), finite-difference gradient tests, status paths, the no-cost gate.
3. **`delt`, `dell`.** Sub-operators per distinct pair, selection per model, the derivative of the template
   coefficients, rows 43-44; the same test set.
4. **Shared IMF weights and the closing measurements.** The host-selected broadcast of `w_IMF`; benchmarks of the
   free-modifications configurations on the CPU and the R9700 (the A100 after the stack); the numbers in
   `doc/GPU_RESULTS.md`; the roadmap and manual text.

Each stage's description records the size of any change, its cause, and the tests; FP32 kernels follow
`doc/MIXED_PRECISION.md` (long sums compensated, grid-only factors in FP64, tolerances in `KP`), device code
follows `CLAUDE.md` ("Device code"), and every hardware-dependent size goes through `src/core/opensps_tuning.f90`.
The suites pass in FP64 and FP32 before each pull request; GPU runs of this stack wait for the owner's go-ahead
on a machine whose GPU is free.

## 7. Verification plan

* **Oracles:** per modification, `generate_ssp_grid` for each model (the host, legacy-parity-tested path) against
  the batch with the modification free: spectra, mass, L_bol, ionizing photons; FP64 to rounding, FP32 to
  `KP` tolerances. Each group on its own and all together; the threshold edges.
* **Gradients:** per row against central finite differences of the batch (away from the documented kinks), FP64
  tight, FP32 loose, and the cotangent reductions against a dense reference on a small operator.
* **Unchanged paths:** `unit` and `regression` suites bit-identical; legacy parity unchanged;
  `benchmark_csp_batch` A/B on the default configurations.
* **Coverage:** each stage's new lines are covered by the tests above; the status and error paths (unsupported
  combinations, the pair limit, bad names) have their own tests; patch coverage is measured locally (gcov) before the
  pull request.
