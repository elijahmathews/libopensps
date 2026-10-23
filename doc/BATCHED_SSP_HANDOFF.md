# Batched SSP: hand-over notes

Written 2026-10-23 for whoever continues the batched SSP (ROADMAP priority 3) after the two
agents of that day (a laptop agent: CPU compilers, LLVM flang, AMD Radeon 780M; an A100 agent:
nvfortran offload). The discussion, with every measurement, is GitHub issue #32 ("Batched SSP:
coordination"). **Delete this file when the work has merged**; what should stay is in
`doc/GPU_PLAN.md` ("Rules learned so far"), `doc/GPU_RESULTS.md` (2026-10-23 entry),
`doc/ROADMAP.md` (priority 3) and `tools/legacy_parity/README.md`.

## What it is

`compute_csp_batch` gets the SSP-level parameters free per model without an SSP grid per model:
SSP age sums = effective templates . V, V = S . (w_IMF w_sfh) per model, S the sparse
IMF-independent operator of `src/physics/opensps_ssp_operator.f90` (built once per metallicity),
the IMF weights of every isochrone point computed per model, then the shared batch. Code:
`src/physics/opensps_ssp_batch.f90` (kernels, `ssp_opset_t`), `opensps_ssp_operator.f90` (the
operator and its batch form), `opensps_imf.f90` (`imf_bin_weights`), the plumbing in
`opensps_csp_batch.f90` (optional `ssp_ops`, an `ssp_opset_t`; on `v2` also `ssp_op`). Tests: `tests/test_csp_batch.f90`
(`check_imf_per_model`, `check_imf_unsupported`, `check_imf_z`) and `tests/physics/
test_opensps_imf.f90`; benchmark: `tests/benchmark_csp_batch.f90` (configs `*_imf`, `*_z_imf`).

## Branches (origin), oldest first

| Branch | Content | State |
|---|---|---|
| `feat/ssp-batch` | the first draft (d4d524b on 6e7ee94, not an ancestor of `main`) | superseded |
| `feat/ssp-batch-v2` @ 9ee300a | the draft rebased on `main` (PR #31) + fixes: effective templates restored, Gauss-Legendre for narrow lognormal bins, status 2 for modifications / per-model metallicity, oracle tests for every IMF type, the A100 N = 1 fix and piece / segment kernels, the zin = 0 test fix | **verified**, PR candidate 1 (IMF free, one metallicity) |
| `feat/ssp-batch-v2-a100-perf`, `-a100-kernels` | steps of the above (ancestors of v2) | merged into v2 |
| `feat/ssp-batch-v2-780m-weights` @ a31e56e | `exp(e ln m2)` instead of a power in the device weights kernel (`imf_bin_weights_ln`) | optional, not in v2 (accuracy 4.6e-11 -> 6.9e-11 FP64 Salpeter, FP32 N = 1000 -17% on the 780M; the A100 agent was asked to measure) |
| `feat/ssp-batch-v2-a100-zvar` @ fdd08cd (+ 9f11157) | IMF and metallicity free per model: `ssp_ops`, `ssp_opset_t`, multi-metallicity kernels, batched cuBLAS product, `check_imf_z`, `*_z_imf` benchmark; 9f11157 adds the one code path | superseded by `zvar2` (not rebased: the laptop agent had pulled it) |
| `feat/ssp-batch-v2-a100-zvar-flang` @ bc9b391 | one commit on fdd08cd: host loops for the multi kernels, `ssp_ops` on every build | superseded by `zvar2` (its host loops are in the unified kernels); 362/362 on CPU compilers and the 780M at that commit |
| `feat/ssp-batch-v2-a100-zvar2` @ 2468916 | zvar rebased on v2 9ee300a, **one code path** (`ssp_op`, `ssp_batch_age_sums`, `imf_*_chunk`, `fold_chunk` / `fold_segments` deleted; one metallicity = a set of one operator, one pass, products straight into `yo`), host loops in the multi kernels, one second-context block in the test | A100 verified (offload FP64, FP32, FP64 + cuBLAS 38 / 38, FP32 + cuBLAS `test_csp_batch` 362 / 362; gfortran and nvfortran host threads verified before the rebase); laptop agent verified the tip 2468916 (2026-10-23): unit + regression 39 / 39 on gfortran FP64 and FP32, gfortran and nvfortran 26.1 host threads (362 / 362 at 1, 2, 4, 8, 16 threads), flang CPU, ifx, and on the 780M with flang offload in FP64 and FP32 (suites 39 / 39, `test_csp_batch` 362 / 362): **no flang-specific change needed**; still to do: the alternating timing on the A100 against `main`, sanitizers on the final tip, the +5% of the one-metallicity case, the PR |

`zvar2` is the one to continue from. The A100 agent measured `feat/ssp-batch-v2-780m-weights`
(a31e56e) on the A100: 6% faster (FP32 N = 10^4 16.6 -> 15.5 us/model), but the FP64 oracle fails
by a hair in the cuBLAS build (Kroupa spectrum 1.05e-10 to 1.15e-10 against the 1e-10 tolerance;
3 of 350 checks), because `exp(e ln m)` has ~|e ln m| ulp of error and `phi - plo` over
narrow bins amplifies it by the inverse relative width, as with the erf: do not merge it as is;
the proposed form is `val = c plo expm1(e dl) / d`, `plo = exp(e ln m1)`,
`dl = log1p((m2 - m1) / m1)` with `ln m1` and `dl` computed once per call. When the weights
variant is final (in v2), rebase `zvar2` and add its per-point arrays to `ssp_opset_t`
(`ssp_opset_build` concatenates the per-point arrays with the offsets `pt_off`).

## Verified (matrix to repeat at any new tip)

CPU, laptop (all 39 unit + regression, `test_csp_batch` 350 / 362 checks): gfortran 16 FP64 and
FP32; OpenMP host threads with gfortran and with nvfortran 26.1 at 1, 2, 4, 8 and 16 threads
(a pointer not private in a host-thread loop once failed CI only at 4+ threads); ifx; flang CPU.
780M (flang offload, `OMP_TARGET_OFFLOAD=MANDATORY`, `--num-processes 1`): FP64 and FP32. A100
(nvfortran 25.1 offload, FP64, FP32, with and without `-Dcublas=true`). Sanitizer on the new
kernels: memcheck and racecheck clean.

Recipes (laptop; `unset HSA_OVERRIDE_GFX_VERSION` in every shell; data from `opensps-data`):

    export OPENSPS_DATA_HOME=~/opensps-data FSPS_DATA_HOME=~/opensps-data SPS_HOME=~/opensps-data
    meson setup build --buildtype=release -Ddata_home=$OPENSPS_DATA_HOME           # gfortran FP64
    meson setup b-cpu32 --buildtype=release -Dkernel_precision=single -Ddata_home=...
    meson setup b-omp --buildtype=release -Denable_openmp=true -Ddata_home=...      # then OMP_NUM_THREADS=1,2,4,8,16
    FC=nvfortran CC=nvc meson setup b-nvomp --buildtype=release -Denable_openmp=true ...
    (source /opt/intel/oneapi/setvars.sh; FC=ifx CC=icx meson setup b-ifx ...)
    FC=flang CC=clang meson setup b-flang-cpu ...
    D=--rocm-device-lib-path=/usr/lib64/rocm/llvm/lib/clang/20/lib/amdgcn/bitcode
    FC=flang CC=clang meson setup b-amd -Daccel=omp_gpu -Dgpu_arch=gfx1103 \
        -Dfortran_args="$D" -Dc_args="$D" \
        -Dfortran_link_args="$D -Xoffload-linker --library-path=/usr/lib64/amdgcn-amd-amdhsa" \
        -Dc_link_args="$D -Xoffload-linker --library-path=/usr/lib64/amdgcn-amd-amdhsa"  # + -Dkernel_precision=single
    OMP_TARGET_OFFLOAD=MANDATORY b-amd/test_csp_batch

The 780M also drives the display: ask the user before GPU runs (CLAUDE.md, "Machines").
A100 (ICDS, `module load nvhpc/25.1`; the data root is an unpack of the `opensps-data` release, no C3K,
so the matrix is run without the C3K case; `unset SPS_HOME`; build directories must be named `build*` or
untracked; never edit sources while a background build or test of the same worktree runs):

    export FC=nvfortran CC=nvc
    meson setup b_a64 --buildtype=release -Daccel=omp_gpu -Dgpu_arch=cc80 -Ddata_home=$DATA   # FP64 tile
    meson setup b_a32c ... -Dkernel_precision=single -Dcublas=true                              # FP32 + cuBLAS
    OMP_TARGET_OFFLOAD=MANDATORY meson test -C b_a64 --suite unit --suite regression --num-processes 1
    compute-sanitizer --tool racecheck --kernel-name regex=ssp_batch b_a32c/test_csp_batch  # (not regex:)
    nsys profile ./b_a32c/benchmark_csp_batch 10000 0 full_smooth_z_imf  # kernel table: nsys stats -r cuda_gpu_kern_sum

Nsight Compute is blocked on the node; check `nvidia-smi --query-compute-apps` before timing (a shared
GPU slows kernels ~30%). `-Dcublas=true` is 35-40% faster at every size on NVIDIA (recommend for
NVIDIA builds; FP64 + cuBLAS uses the double tensor cores and is as fast as FP32 + cuBLAS). Case matrix against another build:
`tools/compare_builds` (`run_cases.sh BUILD OUT CASES.psv`, `compare_builds.py`); the batched
IMF dump: `tools/compare_builds/imf_batch_dump.f90`, `compare_imf_dump.py`.

Benchmark protocol: `benchmark_csp_batch 10000 0 full_smooth` against `full_smooth_imf` (and
`full_smooth_z` against `full_smooth_z_imf`), alternating runs, >= 6, medians; the 780M varies by
up to 30% run to run (N = 1 especially), the A100 by a few %. Baselines: `doc/GPU_RESULTS.md`.

## A100 numbers (nvfortran 25.1, FP32 + cuBLAS unless noted, us/model, magnitudes only)

Baselines, resident basis: `full_smooth` N = 10^4 7.3 (N = 1: 1090); `full_smooth_z` (metallicity
varying, real bases) 10^4: 7.9, 1000: 9.3, 100: 25.9, 1: 1137.

IMF free per model, one metallicity (`basic_imf`; the basic resident-basis batch is 3.0 at 10^4):
tile kernels at v2 N = 1 / 100 / 1000 / 10^4: 4300 / 60 / 26 / 25; with cuBLAS 1800 / 38 / 17.8 /
16.5 (FP64 + cuBLAS 1500 / 33.4 / 16.7 / 15.8); the draft was 15100 / 128 / 32.6 / 28.0. Where the
time goes at 10^4 (nsys): the segment fold ~26%, the IMF weights ~26%, GEMMs ~14%, the rest of the
batch ~25%; the remaining batch kernels are 0.02 to 0.6 us/model each (nothing to win there).
`generate_ssp_grid` per IMF would cost 134 ms (FP32) to 167 ms (FP64) per model here.

IMF and metallicity free (`full_smooth_z_imf`, 12 metallicities, 1.25 s to build the operators once)
N = 1 / 100 / 1000 / 10^4: 3263 / 98.8 / 41.4 / 34.5 (first version 3656 / 217.7 / 48.9 / 33.9); the
IMF part costs twice the one-metallicity part at large N (two operators per model). One launch for
all metallicities and the strided batched FP32 cuBLAS product (`gemm_columns_batched`, <= 512 columns)
fixed the launch-bound N = 100. The one-metallicity case through the unified kernels: N = 10^4
17.5 vs 16.6 (+5%, open), N = 100 36.0 vs 37.6.

Accuracy against the per-IMF `generate_ssp_grid` oracle: FP64 1e-11 to 6e-11 (Chabrier 2e-11 and van
Dokkum 6e-11 after the Gauss-Legendre fix), FP32 5e-7 (3.5e-7 typical). Against gfortran FP64 the
A100 FP64 matrix passes with 4.7e-10 to 1.4e-9 in every SSP case, and an nvfortran *host* build gives
the same numbers (offload vs host 4e-16): not FMA (`-Mnofma` gives the same), not the IMF code.

Existing paths vs `main`, alternating, median of 6: all ratios 0.92 to 1.03 (N = 1 and 10^4; basic,
`full_smooth`, `full_smooth_z`, FP32; FP64 basic).

## Isochrone modifications per model: the design (not built)

`fbhb`, `sbss`, `agb`, `pagb`, `redgb` and the TP-AGB normalisation are per-point factors on a fixed
structure of S (`ssp_operator.f90`: w_j = f_j w_IMF(j); blue HB stars w_i = f_i w_IMF(j); blue
stragglers w_i = g_i sum over HB points of w_IMF, g_i = sbss / 20), so with the points in groups
(normal, phase 5, phase 6, HB, blue HB, stragglers) the fold takes x(p, m) = c_group(p)(m) w_IMF
w_sfh, the same work as today; the stragglers use one per-age HB sum per model. `delt` and `dell`
change which templates the TP-AGB stars use (a few % of the points; the dust-shell corners of C-rich
stars are slot-keyed: the hard part): rebuild only those rows per model on the device (needs the
templates reachable under the shifts in `etpl`, so n_eff grows), or group models by (delt, dell)
and build one operator set per pair (no new hot-path code; the usual case, they are rarely free).
A full operator rebuild per model costs 0.12 s (as `generate_ssp_grid`): never the batch path.

## Decisions taken (by the user, 2026-10-23)

- Isochrone modifications and per-model metallicity may be deferred with a documented status
  error: done for `ssp_op` (status 2 with `ssp_op_mod_signature`); `ssp_ops` serves metallicity.
- Lognormal narrow bins: change the formula (Gauss-Legendre), not the tolerances.
- Join the metallicity-free work (flang and host builds); design direction (A100 agent, at the
  user's request): one code path, `nz = 1` as the special case, resident-basis path kept for a
  fixed IMF (7.9 vs 34.5 us/model at N = 10^4 with metallicity varying on the A100).

## Open

1. **Legacy parity** for the lognormal bin change: `tools/legacy_parity/run_matrix.sh`, Chabrier
   and van Dokkum; needs legacy FSPS src at 82a8735 and a legacy-layout data tree (the A100 agent
   was checking what exists on ICDS). Record the numbers in `tools/legacy_parity/README.md`.
2. **Public API** (the user's decision, before `include/` or the C header changes): how `ssp_op` /
   `ssp_ops` appear in `compute_csp_batch` and whether the single-operator form becomes a wrapper
   over a set of one. The C API does not expose either yet.
3. **One code path**: done on `zvar2` (old kernels deleted, host loops in the multi kernels); open: the
   +5% at N = 10^4 for one metallicity (permutation indirection and accumulators; A100 16.6 -> 17.5
   us/model), the whole matrix at the new tip on flang / the 780M, ifx and the host thread sweep,
   sanitizers and the alternating timing against `main` on the final commit.
4. **Per-model metallicity timing on the 780M** (`full_smooth_z` vs `full_smooth_z_imf`) and the
   weak spot at N = 100 (launch bound: about 12 metallicity runs of ~8 models each).
5. **Isochrone modifications per model**: per-point group factors in the same tables (A100 design
   note in the issue).
6. 780M: the template product is the largest piece at N = 1 (5 ms of 10), the IMF weights at
   large N (FP64 transcendentals: 332 us of ~860 per model in FP32); `PT_PIECE` = 32 and `SEG_LEN`
   = 256 (`opensps_ssp_operator.f90`) are A100-tuned guesses; FP32 weights for narrow bins would
   change the accuracy rules (ask the user).
7. The ~6000 racecheck hazards in `band_fluxes` (`opensps_batch_photometry.f90:658-674`), on `main`
   too and numerically benign: worth their own issue.
8. The nvfortran-vs-gfortran 5e-10 in every FP64 SSP case (A100 and nvfortran host alike; flang
   matches gfortran to 5e-14; not FMA; dlogL 1.4e-13 on the same cases so it enters before the
   spectrum sums and is amplified ~10^3): not bisected; dumping `buf%weights` and the template weights
   from both compilers for `ssp_salpeter` would show where it enters.
9. The weights change (see Branches: `expm1` form) and, after it, the opset tables.
10. Per-point group factors for the isochrone modifications (design above), and `PT_PIECE` /
   `SEG_LEN` per device.

## Pull request plan

Small, isolated PRs (CONTRIBUTING.md; squash-merged, the title and description become the
commit): (1) `feat(csp-batch): IMF free per model from the SSP operator` from v2 after items 1
and a final run of the matrix at the tip (state the sizes: 3.6x the resident-basis batch at large
N on the 780M, 1.2 to 3.3x on the A100 per the tables); the lognormal fix could be its own
`fix(imf)` PR first (it changes the reference integrals by up to ~3e-9); (2) metallicity free
(zvar) after items 2 to 4. The user opens and merges PRs; end descriptions and commit messages
with the attribution lines of the session.

## Pitfalls met

`git rebase --continue` needs `-c commit.gpgsign=false`; do not push over a branch the other agent
has pulled; `opensps_setup(ctx, 10, ...)` loads one metallicity (use 0 for all); flang cannot map
arrays of derived type (`ssp_ops%ops(iz)` maps only its `etpl` component per element and works);
nvfortran 25.1 rejects `c_f_pointer` in device code and the `present` map modifier
(`OMP_MAP_PRESENT_ALLOC`); a reduction nested in a teams loop behaves differently per compiler;
never `pkill -f` / `pgrep -f` with a pattern that occurs in your own command. More:
`doc/GPU_PLAN.md`.
