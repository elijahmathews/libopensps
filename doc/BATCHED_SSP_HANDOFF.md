# Batched SSP: hand-over notes

Written 2026-10-23 for whoever continues the batched SSP (ROADMAP priority 3) after the two
agents of that day (a laptop agent: CPU compilers, LLVM flang, AMD Radeon 780M; an A100 agent:
nvfortran offload), extended 2026-11-02 by a desktop agent (AMD Radeon AI PRO R9700). The
discussion, with every measurement, is GitHub issue #32 ("Batched SSP: coordination"). **Delete this file when the work has merged**; what should stay is in
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
| `feat/ssp-batch-v2-r9700` @ ddf2a2d | `zvar2` + 11db183 (benchmark `n_min`) + afa456e/92fca37 (IMF weights from precomputed logs, "design A": FP64, no cancellation) + ddf2a2d (`-Dfp32_imf_weights`, "design B", opt-in) | verified 2026-11-02 on gfortran FP64/FP32, flang CPU, host threads, R9700 offload FP64/FP32, with and without the option; **not yet nvfortran** (briefing in #32) |

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

R9700 desktop (Fedora 44; LLVM flang 22.1.8 + Fedora ROCm 7.1.1; needs the packages `flang`,
`libomp-devel`, `rocm-device-libs`, `llvm`, `clang-tools-extra` and `lld`; 32 GB RAM; the R9700
drives the display, so ask the user before GPU runs of code that has not passed on the host):

    export ROCR_VISIBLE_DEVICES=GPU-1788065730744145   # the R9700 by UUID; hides the 610M iGPU (gfx1036)
    D=--rocm-device-lib-path=/usr/lib64/rocm/llvm/lib/clang/20/lib/amdgcn/bitcode
    LA="['$D','-Xoffload-linker','--library-path=/usr/lib64/amdgcn-amd-amdhsa']"  # a meson list: a
                                         # comma-separated string reaches the linker as one argument
    FC=flang CC=clang meson setup b-r9700-64 --buildtype=release -Daccel=omp_gpu -Dgpu_arch=gfx1201 \
        -Dfortran_args=$D -Dc_args=$D -Dfortran_link_args="$LA" -Dc_link_args="$LA"  # + -Dkernel_precision=single
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
        meson test -C b-r9700-64 --suite unit --suite regression --num-processes 1   # OMP_TARGET_OFFLOAD=MANDATORY

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

## A100 verification of `zvar2` @ 2468916 (end of the 2026-10-23 session)

Offload (nvfortran 25.1, `OMP_TARGET_OFFLOAD=MANDATORY`): unit + regression 38 / 38 in FP64, FP32 tile
and FP64 + cuBLAS; `test_csp_batch` 362 / 362 in FP32 + cuBLAS. compute-sanitizer on `test_csp_batch`
restricted to the `ssp_batch` kernels: racecheck 0 hazards, memcheck 0 errors (FP32 + cuBLAS, FP64, FP32
tile). Alternating timing against `main` (ab08100), median of 6, N = 1 and 10^4, FP32, mags / spec:
existing paths within 0.99 to 1.05 of `main` everywhere except two cases: **tile `full_smooth_z`
N = 10^4 +9% / +11% (8.0 -> 8.7 / 15.0 -> 16.7)** and cuBLAS `full_smooth` N = 1 mags +11% (1048 -> 1159,
spec +5%; N = 1 spread is ~5-10%). The first is not the code: with `zin = 10` in the benchmark (empty bases
for the other metallicities, as `main` has it) the `zvar2` tile build measures 7.94 / 14.95, equal to
`main`; real bases for every metallicity cost 8.7 / 16.7 in the tile build (8.5 / 15.4 with cuBLAS), so the
old `*_z` baselines were slightly optimistic. The N = 1 cuBLAS `full_smooth` case was not repeated.
The one-metallicity IMF case through the unified kernels is +5% at N = 10^4 (17.5 vs 16.6 us/model): the
per-launch kernel times show +4.7% in the weights kernel and +3% in the fold (table lookups by
metallicity: `viz`, `perm`, the offsets); accepted for one code path. Host builds on this tip: see the
issue (#32) for the last run; before the rebase gfortran 13.4 serial / OpenMP host threads (FP64, FP32) and
nvfortran 25.1 host threads passed 38 / 38. **Not done by the A100 side:** flang / 780M and ifx on this
tip (the laptop agent is validating), the legacy parity matrix, the CPU case matrix against gfortran on
this tip (done earlier at cb5ff34: PASS), the `a31e56e` weights change settled.

## R9700 verification and numbers (`zvar2` @ 2468916, 2026-11-02)

Desktop: AMD Radeon AI PRO R9700 (gfx1201, 64 CUs, 32 GB GDDR6, PCIe 5 x16), Ryzen 9 9950X;
LLVM flang 22.1.8 + Fedora ROCm 7.1.1, gfortran 16.2. Not available there: nvfortran, ifx, cuBLAS.

Verification (all with the C3K case skipped: no C3K data in the release): unit + regression 38 / 38
(1789 TAP checks) and `test_csp_batch` 362 / 362 on flang CPU, gfortran FP64, gfortran FP32, gfortran
OpenMP host threads (`test_csp_batch` at 1, 2, 4, 8, 16, 32 threads; suites at 16), and **R9700 flang
offload FP64 and FP32** (`OMP_TARGET_OFFLOAD=MANDATORY`, one test at a time); no GPU fault in the
kernel log. `check_imf_z` spectrum, Kroupa / Chabrier: R9700 FP64 1.2e-10 / 1.3e-11, FP32 4.9e-7 /
4.2e-7 (as on the 780M); gfortran FP64 7.7e-15 / 7.1e-15, FP32 4.6e-7 / 4.2e-7.

Timing by the protocol above (6 alternating rounds of the four configs in FP32 and FP64; us/model,
magnitudes only, median [min-max]; the last column with spectra copied to the host):

| FP32 | N = 1 | N = 100 | N = 1000 | N = 10^4 | N = 10^4, spectra |
|---|---|---|---|---|---|
| `full_smooth` | 5821 [5228-6468] | 59.5 [59.0-61.9] | 29.5 [29.3-29.9] | **28.1** [27.9-28.5] | 32.9 |
| `full_smooth_imf` | 7941 [7796-8038] | 163.7 [163.0-166.6] | 113.4 [112.4-113.7] | **105.9** [105.2-106.0] | 110.3 |
| `full_smooth_z` | 5834 [5225-6288] | 66.5 [63.6-67.5] | 30.4 [30.3-30.9] | **28.8** [28.7-28.8] | 33.5 |
| `full_smooth_z_imf` | 9923 [9896-10002] | 564.2 [564.1-566.9] | 194.7 [193.1-196.7] | **175.1** [174.7-176.3] | 179.4 |

| FP64 | N = 1 | N = 100 | N = 1000 | N = 10^4 | N = 10^4, spectra |
|---|---|---|---|---|---|
| `full_smooth` | 5909 [5706-6390] | 64.5 [62.4-66.6] | 32.9 [32.6-34.8] | **33.4** [33.1-34.0] | 41.4 |
| `full_smooth_imf` | 8863 [7935-9049] | 207.5 [205.7-210.9] | 147.9 [147.3-148.3] | **140.2** [139.2-141.6] | 149.4 |
| `full_smooth_z` | 6110 [4441-7127] | 73.9 [70.3-75.1] | 36.3 [36.1-38.5] | **36.9** [36.5-37.3] | 44.1 |
| `full_smooth_z_imf` | 11225 [11204-11293] | 733.5 [726.8-734.6] | 261.1 [260.8-261.5] | **231.3** [230.8-231.7] | 239.3 |

- Cost of the free parameters, N = 1 / 100 / 1000 / 10^4: IMF free / resident basis 1.36 / 2.75 /
  3.85 / 3.76 (FP32; 780M 3.6 at large N); IMF and metallicity free / metallicity free 1.70 / 8.48 /
  6.41 / 6.09 (FP32; 780M ~5.9, A100 4.4). N = 100 is the weak spot here too.
- Against the other GPUs at N = 10^4 (FP32): ~8x faster than the 780M in every config; slower than
  the A100 by 3.3-3.9x (resident basis; the A100 has ~2.5-3x the memory bandwidth) and 5.1x
  (`full_smooth_z_imf`, 34.5 on the A100, cuBLAS).
- FP64 costs only 1.02-1.34x FP32, so the batch is bound by memory traffic and fixed costs, not by
  arithmetic. The IMF weights are FP64 (`WP`) in both builds, so this does not measure their cost:
  that needs per-kernel times (LLVM offload time trace or the ROCm profiler; not done yet).
- Run-to-run spread is < 4% at N >= 100 (N = 1 up to ~25%). Junction temperature 62-86 C, edge
  ~20 C lower; no sign of throttling.

## IMF weights without cancellation (2026-11-02, `feat/ssp-batch-v2-r9700`)

- **Design A (default):** `imf_bin_weights_tab` takes a power-law bin as (c/e) lo^e expm1(e dl)
  from `ssp_opset_t%pt_llo` = ln m1 and `pt_dl` = ln(m2/m1) (`imf_bin_logs`, log1p; once per
  operator set). It's FP64, one exp plus a 13-term series per bin with constant reciprocals,
  and no FP64 divisions per bin: a first version with divisions was 6-8% *slower*.
  - Accuracy ≤ 2.1e-15 per bin against 60-digit references (`tests/data/imf_powerlaw_bins.txt`,
    `tools/reference/imf_powerlaw_bins.py`). The difference form loses 3.6e-2 at relative width
    1e-14, 2e-7 at 1e-9 to 1e-6, and up to 100% within 1e-6 of slope 1.
  - 30% of MIST's bins are narrower than 1e-6.
  - R9700 FP32 / FP64 at N = 10^4: `full_smooth_imf` 104.6 / 141.4 -> 74.5 / 111.7 us,
    `full_smooth_z_imf` 176.4 / 233.0 -> 118.4 / 176.4.
- **Design B (`-Dfp32_imf_weights=true`, FP32 kernels only, off by default):** FP32 exponentials,
  exponent reduced in FP64. A further -8 to -9% (69.1, 106.9 us). Mass and L_bol go to ~3e-8 per
  age (measured on MIST vs quad precision); spectra and magnitudes are unchanged.
- **The oracles:** `test_csp_batch`'s IMF oracles use SSPs from the difference form, whose own
  error is ~7e-10 in spectra; their FP64 tolerance is 1e-9 (`IMF_SPEC_RTOL`). Issue #37 is the
  fix for the single-SSP path; after it the tolerance can return to 1e-10.
- **Not yet:** nvfortran / A100 (brief in #32), ifx, the 780M.

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

1. ~~**Legacy parity** for the lognormal bin change~~: **done 2026-11-02** on the R9700 desktop
   (`main` and v2 9ee300a both PASS, 34 cases without C3K; v2 moves the fork's Chabrier / van
   Dokkum outputs by <= 3.0e-9; numbers in `tools/legacy_parity/README.md`). Legacy source:
   upstream 82a8735 (`git archive 82a8735 src`); legacy data tree: the upstream checkout at
   82a8735 itself (byte-identical to `elijahmathews/fsps` `data/` at `fsps-final` except six
   nebular line tables); corrected tables from `elijahmathews/fsps` `data/nebular/`. The scripts
   needed the data roots as inputs (`DATA_HOME`, `LEGACY_NEBULAR`): branch
   `fix/legacy-parity-data-roots`, its own PR against `main`.
2. **Public API** (the user's decision, before `include/` or the C header changes): how `ssp_op` /
   `ssp_ops` appear in `compute_csp_batch` and whether the single-operator form becomes a wrapper
   over a set of one. The C API does not expose either yet.
3. **One code path**: done on `zvar2` (old kernels deleted, host loops in the multi kernels); open: the
   +5% at N = 10^4 for one metallicity (permutation indirection and accumulators; A100 16.6 -> 17.5
   us/model), the whole matrix at the new tip on flang / the 780M, ifx and the host thread sweep,
   sanitizers and the alternating timing against `main` on the final commit.
4. **Per-model metallicity timing on the 780M** (`full_smooth_z` vs `full_smooth_z_imf`) and the
   weak spot at N = 100 (launch bound: about 12 metallicity runs of ~8 models each). Measured on
   the R9700 (2026-11-02, section above): N = 100 is the weak spot there too (8.5x vs 6.1x at 10^4).
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
never `pkill -f` / `pgrep -f` with a pattern that occurs in your own command. On a 32 GB machine
three suites at 8 processes each ran out of memory (`test_csp_batch` peaks at 2.6 GB with every
metallicity loaded, `benchmark_csp_batch ... *_z_imf` at 2.9 GB) and the OOM kill took the agent's
terminal with it: one suite at a time, <= 4 processes, under `systemd-run --user --scope -p
MemoryMax=...`. A desktop can suspend in the middle of a timing run (the run's times then include the
sleep): run long jobs under `systemd-inhibit --what=sleep`. More:
`doc/GPU_PLAN.md`.
