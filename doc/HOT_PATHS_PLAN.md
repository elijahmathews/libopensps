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

**Step 0, the safety nets (done, #129).** The gradient dumps of the matrix (`bench_matrix.py --dump-grad`, directory mode and exact
comparison in `compare_grad_dump.py`), free [alpha/Fe] (`afe`, `afe+logzsol`) and the IMF limits in the matrix, so that the four-slab
kernels and the gradients have dumps.

**Step 1, the adjoint on the slab template (done, #130).** `backward_age_sums` and `backward_age_sums_afe` became one template
(`age_sums_adj_slabs.inc`, K = 1, 2, 4 on the descriptor of `age_sums_slabs.inc`) of per-slab dot products and an FP64
combination: the adjoint of a linear blend is the dot of each slab with the cotangents, and w_bar and the dz and da derivatives are
combinations of those dots with the weights and their derivatives. The CPU K = 2 path already worked this way (and passed its FP32
tests); now host and device, and K = 1, 2, 4, share it. Not bit-identical for K = 2, 4 on the device and K = 4 on the host (the sums
are formed differently): the host FP64 gradient dumps equal the old ones bit for bit for K = 1 and K = 2 and to 1e-12 for K = 4; on the
R9700 (FP32) the changed rows differ from the old by 2e-8 - 1e-7 of their scale and their error against the FP64 reference stays at
1e-7 (the [alpha/Fe] row 2e-7 against 1 - 1.5e-7 before). The saving is smaller than the forward side's (about 280 lines of the two
routines to about 200 of the template and its wrapper), since the per-pixel difference forms, the CPU dot-product branch and the K-specific
loops are gone but the combination code remains.

**Step 2, one "SSP source" interface (two slices done, #131 and #132; the rest not started).** A small type that says where the SSP sums
come from (resident basis with K slabs, weights over metallicities, operators), built by the planner (`batch_info` already reports it);
the driver calls the source and the flags become its fields. The operator path keeps its own implementation (its physics differs: IMF
weights, group operators) behind the same interface. Slice 1: `ssp_source_t`, `ssp_source_select` (the kind, the number of slabs and the
path label, formerly an inline chain in the driver) and `ssp_source_age_sums` (the dispatcher over the resident-basis kernels). Slice 2:
`ssp_source_describe` (the per-model slab descriptor `zlo`, `zhi`, `dz`, `da`, `nzs`, and the refusal of a horizontal-branch modification that
is not in the operators) and `ssp_source_scalars` (the mass, L_bol and weights block, with its host and device loops). What is left: the
flags `zmode`, `zw`, `imf`, `zimf`, `naf` are still locals of the driver (the nebular calls, the backward pass, the checks), and the stages
after the SSP sums still take them as arguments; passing the source to them is what Step 4 (stage routines) does.

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

* **The whole chain in one go:** `~/scratch/forward-opt/run_verify_chain.sh NAME` (host FP64 + FP32 suites, the R9700 device suite
  rebuilt from scratch, the nvfortran compile; one stage after the other, under a memory cap) after the forward and gradient dumps.

* **Force a rebuild of the tree under test.** Ninja goes by modification times: in a build directory shared between a baseline and a
  new tree, a tree whose files are older than the objects of the previous build is not rebuilt, and the comparison is of a build
  with itself (it happened once: the first R9700 dumps of stage 2 were that, and were redone). The runner scripts `touch` every source
  of the tree first, and the container script fails unless `csp_batch.f90` was compiled.
* **Compile the nvfortran device build for every device kernel change.** nvfortran rejects what flang accepts (a `BLOCK` construct inside
  a parallel region; private arrays are fine there and a callee with local arrays is fine for both).

* Forward dumps on the CPU: build the benchmark of the parent and of the new commit, `bench_matrix.py EXE --sizes 1000 --reps 1
  --dists uniform --dump DIR`, then `compare_forward_dump.py DIR_A DIR_B` (move the TSV out of the dump directory). The benchmark
  assigns values to models by field index: a change of the registry's field order changes the dumps, so rebuild the baseline.
* The same on the R9700: `tools/containers/run.sh localhost/opensps-rocm:7.2.4` with `/src` the tree to build (a `git archive` of
  the parent for the baseline) and one build directory; `OMP_TARGET_OFFLOAD=MANDATORY`, `ROCR_VISIBLE_DEVICES` the R9700.
* nvfortran: the NVHPC container, `FC=nvfortran CC=nvc meson setup ... -Daccel=omp_gpu -Dgpu_arch=cc80 -Dkernel_precision=single`,
  compile only.
* Memory: one build or suite at a time under `systemd-run --user` with a `MemoryMax` (the desktop has 32 GB).
