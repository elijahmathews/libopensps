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

**Step 3, the descriptor in the other blends (done, small).** `slab_blend_wp` and `slab_blend_kp`: the cascade `(1 - da) ((1 - dz) a + dz b) +
da (...)` / `(1 - dz) a + dz b` / `a` over the metallicity slabs and [alpha/Fe] planes, for the ionizing-photon table and the spectra of
the nebular continuum (forward and backward) and the interpolated mass and L_bol of the SSP source. Only the sites that were exact
matches (five, plus the two of `model_sums`); the derivative forms and the sites with other structure stay as they are. The saving is
a few lines: the point is one definition of the blend.

**Step 4, the driver in stages (three parts done, #134, #135 and #137).** `compute_csp_batch` was 1037 lines; it is 775 now (759 of code and a 16-line stage index). Part 1 (#134): the
host-only blocks with a clear input and output are routines: `batch_check_shapes`, `batch_check_models`, `batch_imf_vectors`,
`batch_chain_derivatives`. Part 2 (#135): `batch_model_indices` and `batch_sfh_weights`. Part 3 (#137): the reverse pass's post-processing,
`batch_mass_cotangent` and `batch_sfh_gradients`, and a stage index at the top of the driver (a comment that lists the stages with their
routines; `doc/HOT_PATH_TOUR.md` is its long form). What is not done: the nebular and attenuation calls, smoothing and photometry are still
inline, as is the call of `batch_backward` with its 30 arguments; splitting those needs a workspace type for the packed arrays (`ipk`,
`rpk`, `lpk`, `rout`) and the arrays the stages share, and the device rules about mapping derived types make each cut something to check
on the R9700.

**The nebular code, surveyed (no step).** The family of about 2000 lines (`nebular_setup`, `add_nebular_batch`, `nebular_kernels`,
`nebular_continuum_tiled`, `cue_nebular`, `cue_nebular_backward`, `backward_nebular`, `neb_run_block`, the line routines) is two
physical models (the CLOUDY grids; cue, Li et al. 2024, a neural-network emulation of CLOUDY), each with a forward and a reverse pass,
plus the line profiles and the dust absorption of the lines. They share the interface (`add_nebular_batch`) and the line handling, not
code; merging them would merge two algorithms. `doc/HOT_PATH_TOUR.md` has a section that maps them for a reader.

**Step 5, backend consolidation (analysed, mostly not pursued).** The conditional count by kind (961 after step 2): about 586 are `OMP_TARGET`
switches, of which 153 are the host arm (`#elif defined(_OPENMP)`) of one directive switch repeated at every parallel loop (`!$omp target
teams distribute parallel do` on the device, `!$omp parallel do` on the host, nothing without OpenMP); about 50 are LLVM flang specifics,
79 HIP, BLAS or precision selection. The directive switches are not copies of kernels, so no include template removes them; doing so
would need macros that emit OpenMP directives (not portable across gfortran, flang, nvfortran and ifx) or OpenMP 5 `metadirective`
(not supported by the compilers in use). The kernel families that did have a copy per backend (the age sums, forward and adjoint) are
done; the remaining flang/nvfortran/HIP branches are tuned variants of different algorithms (tiles, products, native HIP kernels), which
is where the code differs because the hardware wants it to. Not pursued unless a specific family shows a copy of the same algorithm.

Not proposed: merging the operator path with the resident-basis path (different physics; the interface of Step 2 is the right level
of sharing); a language change (nothing in the plan needs one).

Order: Steps 0 and 1 first (safe, clear payoff), then 2; re-evaluate before 3 - 5.

## How to check a step (recipes)

* **CPU bit identity needs `-fno-tree-vectorize`.** The CPU library calls glibc's vector math routines (`_ZGVbN2v_exp`, `_log`, `_log10`,
  `_pow`, libmvec: `nm -D libopensps.so | grep _ZGV`): GCC vectorizes loops containing `exp`, `log` and `pow` with them, and libmvec differs
  from the scalar libm by about an ulp. Which loops vectorize depends on the code around them, so a refactor that cannot change arithmetic
  (step 4: moving shape checks into a routine) still changes about 7% of the models of a batch by 1e-16 relative (up to 7e-16 in a pixel, 1e-14 in
  a gradient dump) at the default flags, and not at all with `-O1` or `-fno-tree-vectorize`. So: compare two CPU builds of a refactor
  with `-Dfortran_args=-fno-tree-vectorize` (both built the same way: `~/scratch/forward-opt/novec_full.sh` does it for a baseline tree and
  the worktree, forward and gradient dumps), and expect 1e-15 differences at the default flags. (The device kernels have no libmvec; their
  dumps stayed bit-identical through every step.)

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
