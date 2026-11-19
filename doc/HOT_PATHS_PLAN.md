# Unifying the hot paths: plan and status

What `doc/HOT_PATHS.md` describes (the classes of per-model parameters, the cost of each, the matrix benchmark) is the model;
this file is the work plan for making the code match it: one driver, one interface for where the SSP sums come from, one body
per kernel family. The purpose is maintainability (one path to read, test and tune); speed parity is the bar, a speedup a bonus.

## Rules for every step

* Outputs bit-identical: the forward dumps (`tools/bench_matrix.py --dump`, `tools/compare_forward_dump.py`) and the gradient
  dumps (`--dump-grad`, `tools/compare_builds/compare_grad_dump.py`) of the matrix before and after, on the CPU (FP64) and on the
  R9700 (FP32 kernels). A baseline for a comparison is built from the parent commit, in the same way as the new one.
* No slowdown in the matrix (parity within noise, median ratio near 1 on CPU and R9700).
* The preprocessor conditionals of `src/` (and its `.inc` files) fall or stay: `tools/check_hotpath_structure.py`.
* Host FP64 and FP32 suites, the R9700 device suite, and an nvfortran compile in the NVHPC container
  (`-Daccel=omp_gpu -Dgpu_arch=cc80`); the A100 cannot be run here, so a step that touches device kernels is added to the list
  for the ICDS session.
* A step that fails its gate is dropped, not forced. One pull request per step, stacked on the previous one.

## Done (stack #112 - #128)

* Registry, `batch_info`, the matrix benchmark, equivalence tests, forward dumps, value classes (#117 - #120).
* Per-model `sigma_smooth` (#121), IMF mass limits (#126), library `stop` to a status (#124), the structure ratchet (#122).
* One body for the mass/L_bol/weight loops (#125); the host age sums for K = 1, 2, 4 slabs from one skeleton and a per-model
  descriptor (#127); the two- and four-slab device age sums from one include template (`age_sums_slabs.inc`, #128).
* Not unified on purpose: the K = 1 register tile (`age_sums`) and `age_sums_w` are matrix products (a gemm in BLAS builds).

## Where the code stands (survey)

* `compute_csp_batch` is 1148 lines with 38 conditionals; its SSP section is steered by about 70 uses of the flags `zmode`, `zvar`,
  `host_m`, `zw`, `imf`.
* Three SSP sources: the resident basis (K slabs), the weights over metallicities, the operators (`opensps_ssp_batch`).
* The adjoint repeats the forward structure: `backward_age_sums` (K = 1, 2) and `backward_age_sums_afe` (K = 4) are 171 + 109 lines.
* The nebular code is about 1500 lines in three variants of the Q source (from the IMF, from the resident basis, cue).
* Conditionals per routine, most first: `compute_csp_batch` 38, `nebular_kernels` 18, `apply_igm_z` 17, `backward_nebular` 17,
  `cue_nebular_backward` 16, `igm_cap_pixels` 13, `backward_attenuation` 13.

## The steps

**Step 0, the safety nets.** The gradient dumps of the matrix (`bench_matrix.py --dump-grad`, directory mode and exact
comparison in `compare_grad_dump.py`), free [alpha/Fe] (`afe`, `afe+logzsol`) and the IMF limits in the matrix, so that the four-slab
kernels and the gradients have dumps. Status: in progress.

**Step 1, the adjoint on the slab template.** `backward_age_sums` and `backward_age_sums_afe` on the descriptor of
`age_sums_slabs.inc`, with one include template for the adjoint device and host branches (about 280 lines to about 150).
Gate: the gradient dumps bit-identical, `test_csp_batch_grad` and `c_batch_grad`, the matrix.

**Step 2, one "SSP source" interface** (the biggest change to how the code reads). A small type that says where the SSP sums come
from (resident basis with K slabs, weights over metallicities, operators), built by the planner (`batch_info` already reports it);
the SSP section of `compute_csp_batch` calls one entry point and the flags become its fields. The operator path keeps its own
implementation (its physics differs: IMF weights, group operators) behind the same interface.

**Step 3, the descriptor in the other blends.** The nebular Q and ionizing-flux blends, the L_bol blend and the mass blend repeat the
cascade `(1 - da) ((1 - dz) a + dz b) + ...`; with the descriptor they become one helper. Only the exact-match sites (the
derivative forms stay).

**Step 4, the driver in stages.** Split `compute_csp_batch` into stage routines (SFH weights, SSP sums, nebular, attenuation, emission,
AGN, IGM, smoothing, photometry) with a shared workspace type. Mechanical, long; the risk is the device rules (mapping derived
types, temporaries): check on the R9700 after every cut.

**Step 5, backend consolidation, one kernel family at a time.** By conditional count: IGM (`apply_igm_z`, `igm_cap_pixels`), nebular,
attenuation; the include-template pattern where it fits. The vendor branches that exist because a compiler needs a different loop
structure stay.

Not proposed: merging the operator path with the resident-basis path (different physics; the interface of Step 2 is the right level
of sharing); a language change (nothing in the plan needs one).

Order: Steps 0 and 1 first (safe, clear payoff), then 2; re-evaluate before 3 - 5.

## How to check a step (recipes)

* Forward dumps on the CPU: build the benchmark of the parent and of the new commit, `bench_matrix.py EXE --sizes 1000 --reps 1
  --dists uniform --dump DIR`, then `compare_forward_dump.py DIR_A DIR_B` (move the TSV out of the dump directory). The benchmark
  assigns values to models by field index: a change of the registry's field order changes the dumps, so rebuild the baseline.
* The same on the R9700: `tools/containers/run.sh localhost/opensps-rocm:7.2.4` with `/src` the tree to build (a `git archive` of
  the parent for the baseline) and one build directory; `OMP_TARGET_OFFLOAD=MANDATORY`, `ROCR_VISIBLE_DEVICES` the R9700.
* nvfortran: the NVHPC container, `FC=nvfortran CC=nvc meson setup ... -Daccel=omp_gpu -Dgpu_arch=cc80 -Dkernel_precision=single`,
  compile only.
* Memory: one build or suite at a time under `systemd-run --user` with a `MemoryMax` (the desktop has 32 GB).
