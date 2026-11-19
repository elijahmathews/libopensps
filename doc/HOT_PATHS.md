# Hot paths: which parameters can be free in a batch, and what each costs

(New to the code? `doc/HOT_PATH_TOUR.md` walks through one batch call, stage by stage, in plain terms.)

A batch (`compute_csp_batch`, `opensps_context_compute_csp_batch*`) lets any member of `PARAMS_CORE` differ from
model to model, within the exclusions of this document. The aim is that the cost of a batch is the sum of what its free
parameters need, not a property of the combination, and that a new free parameter is a new instance of an existing
mechanism, not a new hot path.

## The classes

Every per-model parameter is registered in `src/core/opensps_batch_registry.f90` with a class and a stage (the table is
checked against `PARAMS_CORE` by the unit test `unit_batch_registry_covers_params`, so a new parameter cannot be
forgotten). The class says how the parameter enters the model, which fixes the mechanism and what the cost grows with:

| Class | Enters as | Mechanism | Cost grows with | Fields (today) |
|---|---|---|---|---|
| **C** closed form | read by uniform per-model kernels | the CSP stage: SFH weights, attenuation tables, dust emission, gas, AGN, redshift | nothing | SFH (`tau`, `tage`, `const`, `fburst`, `tburst`, `sf_*`, `sfh`), dust (`dust1`-`dust3`, `dust_index`, ...), `duste_*`, `gas_*`, `frac_obrun`, `fagn`, `agn_tau`, `zred`, `igm_factor` |
| **L** linear mixture | the coefficient of a fixed set of sources | the batch is a sum over sources: metallicity (2 slabs), [alpha/Fe] (4 slabs), the modification groups (`ssp_opset_build_mods`) | the number of sources K | `logzsol`, `afe`, `fbhb`, `sbss`, `agb`, `pagb`, `redgb` |
| **W** weights | per-point weights of the SSP | computed once per distinct value of the parameters (shared IMF weights, `SHARE_IMF_MIN_BATCH`) | the distinct values | `imf1`-`imf3`, `vdmc`, `mdave` |
| **G** grouped discrete | selects a precomputed structure | one structure per distinct value, gathered per model (the TP-AGB pairs, the smoothing kernel) | the distinct values | `delt`, `dell`, `sigma_smooth`, `min_wave_smooth`, `max_wave_smooth` |
| **S** shared | must be equal across the batch, or is not a per-model quantity | status 2 if it differs | | `masscut`, `agb_dust`, `evtype`, `zmet`, `pmetals`, `wgp1`-`wgp3`, `fcstar`, `frac_xrb`, the file names and the output switches |

The SSP stage has two sources behind these classes: the **resident basis** (the SSPs for one IMF and one set of
modifications, built once; metallicity and [alpha/Fe] are blended per model from several resident bases) and the
**operator** (`ssp_ops`; the SSP of each model from its IMF weights and the operators S; the modifications as a sum over
groups). Which one a batch takes follows from which fields are free:

| Free at the SSP level | Source | K |
|---|---|---|
| nothing | resident basis | 1 |
| `logzsol` | resident bases, 2 slabs | 2 |
| `logzsol`, `afe` | resident bases, 4 slabs | 4 |
| IMF (W) | operator | 1 (2 with `logzsol`) |
| `fbhb`..`redgb` (L), `delt`/`dell` (G) | operator, modification groups | 9 + distinct shift pairs |

## Seeing the path

`compute_csp_batch(..., info=info)` returns a `batch_info_t` (`opensps_batch_registry`): the fields that differ among
the models, the number of distinct values of each (up to 16), whether a shared field differs, the SSP source and K.
`batch_info_text(info)` formats it; with `OPENSPS_DEBUG_PATH=1` every batch prints that line to standard error. The
classification costs a pass over the models and is made only on request, so the default path pays nothing.

## Checking that a combination works, and that a refactor changes nothing

* `test_batch_equivalence` (regression suite): each field of the registry that can be free here (`registry_range`), alone,
  then seeded random subsets of 2-6 fields and all of them together; each model of the batch against a batch of that model
  alone (on a resident basis built for its SSP parameters). A batch with only closed-form fields free runs the resident basis
  both ways, so it catches a kernel chosen by the free set (the uniform and per-model twins); one with an SSP-level field
  free runs the operator path against the resident basis. Metallicity and [alpha/Fe] free per model need the bases of every
  metallicity and are covered by `test_csp_batch`.
* Bit-identical baseline for a refactor: `benchmark_csp_batch` with `OPENSPS_BENCH_DUMP_FWD=<file>` writes the magnitudes and
  spectra of its last batch; `tools/bench_matrix.py --dump DIR` does it for every free set and distribution of the matrix, and
  `tools/bench_matrix.py --dump-grad DIR` writes the gradients (params_bar) of every set likewise, and
  `tools/compare_builds/compare_grad_dump.py DIR_A DIR_B` compares two runs bit for bit; `tools/compare_forward_dump.py DIR_A DIR_B` compares two forward runs (`--tol 0`, the default, needs identical bits, which the same
  build on the same device gives).

## Guardrails

`tools/check_hotpath_structure.py` (the `unit_hotpath_structure` test) holds the number of preprocessor conditionals per
directory of `src/` and of bare `stop` statements in the library at a committed baseline
(`tools/hotpath_structure_baseline.json`): they may fall, never grow without the baseline saying so in the same change. The
rules for new parameters, kernel variants and library routines are in `AGENTS.md` and `CONTRIBUTING.md`.

## Not yet free per model (each its own piece of work)

* (`sigma_smooth` and its range are per model since the smoothing PR: the models are grouped by their distinct (sigma, range)
  triples, one kernel per group on the fast device operator (at most 16 distinct), a column at a time with each model's own
  values on the host modes. With nebular lines in the spectrum the lines' profile table is built once per batch from one
  sigma, so a batch whose models differ in sigma there is status 2.)
* (`imf_lower_limit` and `imf_upper_limit` are per model since the IMF-limits PR, class W with the other IMF parameters: the
  IMF weights of the operator path already took the limits per IMF (`IMF_P_LOWER_LIMIT`, `IMF_P_UPPER_LIMIT`, and the lower edge
  of the first bin with the lower limit unless the isochrones fix it), so a model's own limits are two more entries of its
  parameter vector. They are context settings, and the parameter set carries them (`prepare_pset`); without operators the
  resident basis has the context's, so other limits are status 2, as with a user IMF file, which has its own.)
* The operator path with `afe`, `ssp_weights_z`, `add_neb_emission = 2`, `add_xrb_emission = 1` or `smooth_lsf`; cue with
  a free IMF; `agb_dust`, `masscut`, `evtype` (they select stars: a rebuild per distinct value).

## Where the cost goes: the free-parameter matrix

`tools/bench_matrix.py` runs `benchmark_csp_batch` with each parameter alone free on top of the baseline (`full`: dust,
dust emission, nebular emission, IGM, AGN; `tau`, `tage`, `duste_umin`, `gas_logu` always vary), then pairs and all together,
under three distributions. Microseconds per model at N = 10^4, magnitudes only; ratio to the baseline in parentheses
(2026-11-18, MIST + MILES; the CPU is FP64 with 8 OpenMP threads, the R9700 FP32 kernels through the ROCm container):

| Free (alone) | CPU (baseline 322.8) | R9700 (baseline 8.41) |
|---|---|---|
| any of 21 closed-form fields (`dust1`-`dust3`, `dust_index`, `dust1_index`, `dust_tesc`, `frac_nodust`, `duste_gamma`, `duste_qpah`, `gas_logz`, `frac_obrun`, `fagn`, `agn_tau`, `zred`, `igm_factor`) | x0.99 - x1.06 | x1.02 - x1.09 |
| `logzsol` (resident bases) | x1.04 | x1.23 |
| `imf1`, `imf2`, `imf3` (operator path) | x3.7 | x2.7 |
| `mdave` | x3.7 | x4.1 |
| `vdmc` (the van Dokkum IMF) | x4.8 | **x12.3** |
| `fbhb`, `sbss`, `agb`, `pagb`, `redgb` (group operators) | x4.3 | x3.5 - 3.7 |
| `delt` or `dell` (four distinct pairs) | x6.7 | x5.4 |
| all of dust, gas, AGN, `zred`, `logzsol`, IMF, `agb`, `pagb`, `redgb` together | x6.6 | x6.6 |

* **The CSP stage is not fragmented by free-parameter set:** the closed-form fields cost nothing beyond the baseline, in any
  number (all together adds only the SSP-level steps).
* **The cost steps are the classes.** The operator path (a free IMF, a factor or a shift) is x2.7 - 3.7; the group operators
  (class L) add the groups; the shifts (class G) cost with their distinct pairs: with every model on one pair, `delt` is x0.64
  (CPU) and x0.66 (R9700) of the uniform spread, one group instead of four.
* **Interactions.** Metallicity free on the operator path costs more than the sum of its parts (cost(A+B) / (cost(A) + cost(B) -
  baseline)): with an IMF field x1.43 (CPU) and x1.51 (R9700), with a factor x1.57 and x1.58, with a shift x1.78 and x1.81:
  the operator runs once per bracketing metallicity (K = 2). On the resident basis the metallicity is a blend (x1.04, x1.23).
  IMF + factors (x0.61 CPU, x0.76 R9700) and factors + shifts (x0.69, x0.68) are sub-additive: they share the group operators'
  entry cost.
* **Distribution.** Clustering `logzsol` or the IMF parameters into four tight groups, or putting every model at one value,
  does not change the cost (CPU: all fields x1.0; R9700, distributions interleaved per set so the clocks' drift does not
  show up as a difference, noise floor 5%: `logzsol`, `imf1`, `fbhb` and `logzsol` + `imf1` within it). The exception is
  class G above. (A first R9700 run with the distributions in long blocks showed x0.6 - 1.4 differences for every field,
  baseline included: a drift of the clocks between the blocks; the driver now interleaves them.)
* **Where to look next.** `vdmc` x12.3 (the van Dokkum IMF's weights are the single most expensive field on the R9700, x2.5
  over the Kroupa ones: the per-bin integrals of the lognormal); metallicity on the operator path (x1.5 over additive); the
  shifts (a group per distinct pair).

The TSV and the reports of these runs are not in the repository (they depend on the machine); `tools/bench_matrix.py EXE --tsv
out.tsv --report` reproduces them, `--dump DIR` writes forward dumps for `tools/compare_forward_dump.py`.
