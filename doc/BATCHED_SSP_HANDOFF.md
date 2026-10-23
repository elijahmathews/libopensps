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
`opensps_csp_batch.f90` (optional `ssp_op`, `ssp_ops`). Tests: `tests/test_csp_batch.f90`
(`check_imf_per_model`, `check_imf_unsupported`, `check_imf_z`) and `tests/physics/
test_opensps_imf.f90`; benchmark: `tests/benchmark_csp_batch.f90` (configs `*_imf`, `*_z_imf`).

## Branches (origin), oldest first

| Branch | Content | State |
|---|---|---|
| `feat/ssp-batch` | the first draft (d4d524b on 6e7ee94, not an ancestor of `main`) | superseded |
| `feat/ssp-batch-v2` @ 9ee300a | the draft rebased on `main` (PR #31) + fixes: effective templates restored, Gauss-Legendre for narrow lognormal bins, status 2 for modifications / per-model metallicity, oracle tests for every IMF type, the A100 N = 1 fix and piece / segment kernels, the zin = 0 test fix | **verified**, PR candidate 1 (IMF free, one metallicity) |
| `feat/ssp-batch-v2-a100-perf`, `-a100-kernels` | steps of the above (ancestors of v2) | merged into v2 |
| `feat/ssp-batch-v2-780m-weights` @ a31e56e | `exp(e ln m2)` instead of a power in the device weights kernel (`imf_bin_weights_ln`) | optional, not in v2 (accuracy 4.6e-11 -> 6.9e-11 FP64 Salpeter, FP32 N = 1000 -17% on the 780M; the A100 agent was asked to measure) |
| `feat/ssp-batch-v2-a100-zvar` @ fdd08cd | IMF and metallicity free per model: `ssp_ops`, `ssp_opset_t`, multi-metallicity kernels, batched cuBLAS product, `check_imf_z`, `*_z_imf` benchmark | experimental; device only (nvfortran offload verified) |
| `feat/ssp-batch-v2-a100-zvar-flang` @ bc9b391 | one commit on zvar: host loops for the multi kernels, `ssp_ops` on every build | CPU compilers and 780M verified (362/362) |

zvar (and zvar-flang) are based on v2 at 89eaa28; v2 has since gained 9ee300a (a test-only fix
that also exists, in another form, in the zvar branch): rebasing meets that conflict in
`tests/test_csp_batch.f90` and `tests/benchmark_csp_batch.f90`; keep zvar's `check_imf_z`, take
v2's second-context block.

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
A100: `module load nvhpc/25.1`, `-Daccel=omp_gpu -Dgpu_arch=cc80`, `-Dcublas=true` is 35-40%
faster at every size on NVIDIA (recommend for NVIDIA builds). Case matrix against another build:
`tools/compare_builds` (`run_cases.sh BUILD OUT CASES.psv`, `compare_builds.py`); the batched
IMF dump: `tools/compare_builds/imf_batch_dump.f90`, `compare_imf_dump.py`.

Benchmark protocol: `benchmark_csp_batch 10000 0 full_smooth` against `full_smooth_imf` (and
`full_smooth_z` against `full_smooth_z_imf`), alternating runs, >= 6, medians; the 780M varies by
up to 30% run to run (N = 1 especially), the A100 by a few %. Baselines: `doc/GPU_RESULTS.md`.

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
3. **One code path** (A100 agent, in progress on the zvar branch): delete `imf_pieces_chunk`,
   `fold_segments`, `ssp_batch_age_sums` once the one-metallicity case costs the same (A100
   targets: 16.5 / 15.8 us per model at N = 10^4 with cuBLAS, FP32 / FP64) and gives the same
   results; the host paths (`imf_ages_chunk`, `fold_chunk`) stay until the multi kernels have host
   loops everywhere (zvar-flang has them). Re-run the whole matrix above afterwards.
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
   matches gfortran to 5e-14): not bisected; dumping `buf%weights` and the template weights from
   both compilers for `ssp_salpeter` would show where it enters.

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
