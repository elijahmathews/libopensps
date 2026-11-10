# Mixed precision: FP32 kernels

The expensive, per-pixel parts of FSPS (the kernels a GPU port would run) can be computed in
single precision without changing the results beyond FP32 rounding. This matters for GPUs:
consumer and workstation cards (AMD Radeon / Radeon Pro, Intel Arc, NVIDIA A40 and GeForce)
execute FP64 at 1/16 to 1/64 of their FP32 rate, so an FP64-only port would leave most of their
throughput unused. This document records the design, the evidence and the rules the kernels
must follow.

## Build option

```
meson setup build -Dkernel_precision=single     # default: double
```

`src/core/opensps_precision.f90` defines two kinds:

- `WP` (always FP64): setup, isochrones, IMF and SFH weights, masses, luminosities, all scalar
  bookkeeping, and the public API (C and Fortran interfaces stay double). The one exception is
  the per-model IMF bin weights of the batched SSP, below.
- `KP` (kernel precision; FP64 by default, FP32 with `-Dkernel_precision=single`): the
  per-pixel arrays and arithmetic of the expensive kernels, and the batched SSP's per-model IMF
  bin weights.

With `KP = WP` the default build is unchanged: SSP spectra are bit-identical and CSP outputs
move by at most 2e-14 (the more accurate absorbed-luminosity integral below).

## What runs at kernel precision

| Kernel | Arrays (KP) | Notes |
|---|---|---|
| Stellar spectra: main library, WMBasic, AGB, post-AGB, WR | template libraries, per-star spectra | bilinear weights computed in WP per star, cast once |
| AGB circumstellar dust screen | `flux_dagb`, spectrum | |
| SSP sum over stars | `ssp_temp_grid`, weights | compensated sum |
| CSP sum over ages | SSP basis, `spec_young/old`, emission-line grids | weights cast once per age |
| Dust attenuation and emission | curve tables, transmission, work spectra, Draine & Li shape | curve from grid-only tables (formed in FP64 per call), power laws via `exp_kp`, absorbed luminosity integrated directly (1 - T in Kahan's form) |
| Nebular emission | line profiles (`gaussnebarr`), spectra, continuum weights | ionizing integral compensated; continuum per age in KP, the CLOUDY log grid interpolated and the exponents reduced in FP64 (`exp_kp`) |
| IGM, AGN torus emission | transmission, spectra | |
| Magnitudes | filter integrals | compensated sum |

Still computed in WP from the FP32 spectrum (element-wise or cheap; candidates for later):
velocity smoothing, spectral indices, the redshift interpolation before the filter integrals,
and the Draine & Li template interpolation.

The main stellar library (`speclib`, the largest array) was already stored in FP32.

## Evidence

All numbers compare `-Dkernel_precision=single` with the default build, case by case.

**Parameter matrix** (`tests/data/param_cases.psv`, 36 cases: IMFs, SFHs, all dust types, dust
emission, nebular emission, IGM, AGN, smoothing; MIST + MILES, Padova + BaSeL, MIST + C3K,
THEMIS, BPASS):

| Output | Max difference |
|---|---|
| Spectrum per pixel, 912 A - 2.5 um (99th percentile) | 3.3e-7 (2.3e-7) |
| Spectrum per pixel, > 2.5 um | 3.0e-7 |
| Magnitudes | 1.6e-7 mag |
| Emission lines | 3.1e-7 |
| Spectral indices | 4.6e-6 (absolute) |
| Stellar mass, L_bol | identical (WP) |

Pixels below 1e-6 of a spectrum's peak are excluded. The FP32 build also passes the legacy
FSPS parity matrix (`tools/legacy_parity/run_matrix.sh`, `BUILD=<fp32 build>`) at the FP64
tolerances, and the full unit and regression suite (31 tests).

**Stress cases** (23 extreme configurations: dust2 up to 6, dust2 = 0.001, Witt & Gordon, hot
dust, birth-cloud fractions, 0.8-5 Myr populations with nebular emission at log U = -1 and -4,
IGM at z = 6 and 10, strong AGN, sigma = 1000 km/s, late bursts, lowest and highest
metallicity, extreme IMF slopes, maximal AGB dust, C3K, BPASS, BaSeL): spectra within 4.8e-7,
lines 3.4e-7, indices 8e-8 relative. The only exception is magnitudes of bands more than 40
mag (a factor 1e16) fainter than the brightest band, behind the z = 10 Lyman break, where the
flux is numerically zero in either precision.

**Nebular continuum and attenuation in KP (2026-11).** The batch's nebular continuum and dust
attenuation, FP64 until then, moved to KP (rule 7, curve tables). Measured with gfortran 16 FP32
against FP64, data release `data-v2`, without C3K:

| | Before | After |
|---|---|---|
| Matrix (34 cases): spectrum per pixel, 912 A - 2.5 um / all | 3.4e-7 / 3.5e-7 | 3.7e-7 / 4.0e-7 |
| Matrix: magnitudes | 1.7e-7 mag | 1.8e-7 mag |
| Stress (25 cases, `tools/compare_builds/stress_cases.psv`): spectrum | 7.7e-7 | 7.7e-7 |
| Stress: dust2 = 6 / 5 Myr behind dust1 = 3 | 2.3e-7 / 2.2e-7 | 7.5e-7 / 6.2e-7 |
| Stress: magnitudes | 3.1e-7 mag | 3.1e-7 mag |

The FP64 build moves by at most 5e-15. The stress table is a reconstruction of the 23 cases
above, which were not kept, plus young populations behind dust; its worst case before and after
(7.7e-7, a late burst of half the mass) does not involve this change.

**Naive FP32 everywhere fails.** Building all of FSPS in FP32 (`FORCE_SINGLE_PRECISION`) gives
errors up to 50% in old far-UV spectra, 0.5 mag in UV bands, and NaNs in the nebular and BPASS
paths. The causes are in the cheap parts (isochrone handling and IMF weights, which difference
nearly equal masses; constants that overflow), which is why they stay in WP.

## What the FP32 results are gated on

The requirement is set by what the output is used for, not by what the kernels happen to reach.

| Quantity | Size |
|---|---|
| Photometric noise in fits (5%, 1%, 0.1% at best) | 54, 11, 1 mmag |
| Known, intentional differences from legacy FSPS (`tools/legacy_parity/README.md`) | 0.2-2 mmag |
| Systematic uncertainty of libraries, isochrones, IMF and dust law | at least 50 mmag |
| **The requirement: band magnitudes and per-pixel fluxes, FP32 against FP64** | **1e-5 (0.011 mmag)** |
| Reached by the kernels (`tools/compare_builds`, matrix cases) | 5.7e-7 per pixel, 5.3e-7 mag (the tripwire) |

An error of 1e-5 is 2e-4 sigma at 5% noise and 0.01 sigma at 0.1%, 20 times below the smallest physics
difference already accepted against legacy, and it is round-off: close to random across pixels and averaged
over the hundreds of pixels of a band, not a coherent shift. The 5e-7 level stays as a tripwire: a change
that moves the measured error past it is justified in the pull request (what, how much, why the physics
allows it), not blocked. What the 1e-5 does **not** cover, and is gated explicitly instead, are the
failures that are not round-off: overflow and underflow (rules 5-7), cancellation (rule 2: the dust
luminosity at small dust2, differences of nearly equal integrals) and the large exponent of the nebular
continuum. Gradients are the other consumer: reverse-mode gradients are fine at 1e-5, but finite
differences of FP32 output with small steps are not (Caveats).

## Rules for FP32 kernels

These came out of the measurements; a GPU port must follow them.

1. **Long sums are compensated or pairwise.** A plain sequential FP32 sum over the ~2000 stars
   of an isochrone loses up to 1.1e-5; compensated (Kahan) summation brings it to 1e-7. On a
   GPU, tree reductions and tiled matrix products give the same effect. Compilers must not
   reassociate floating point (no `-ffast-math`, no nvfortran `-fast`), which would delete the
   compensation.
2. **Never subtract nearly equal integrals.** The dust luminosity was intrinsic minus
   attenuated, which cancels when little light is absorbed (an FP32 error of 1e-7 becomes 1e-5
   for dust2 = 0.01). It is now integrated directly from positive terms, with
   1 - T = -expm1(-tau) evaluated without cancellation.
3. **Never store logarithms in FP32 that are exponentiated later.** An FP32 log of size |x|
   gives 10^x a relative error of ln(10)|x|eps: 5e-6 for the nebular continuum (|x| ~ 40). Keep
   such arrays in WP (they are small) or store linear values.
4. **Grid-only factors are formed in FP64.** Trapezoid weights such as 0.5|nu(i+1) - nu(i)|
   lose three digits if differenced in FP32 (nu / dnu ~ 1e3). They depend only on the grid and
   can be precomputed once in FP64 and stored in FP32.
5. **Physical units keep values in range.** FP32 spans 1.2e-38 to 3.4e38. Spectra per solar
   mass in L_sun/Hz (~1e-13 at the peak) sit well inside; scale factors such as L_sun/h ~ 6e59,
   ionizing photon rates ~ 1e46-1e53 s^-1 and constants like (10 pc)^2 stay in WP scalars.
   Test inputs must use physical scales too (the nebular unit tests used 1 L_sun/Hz).
6. **Floors are tiny(1.0_KP)**, not `SAFE_FLOOR` (which underflows to 0 in FP32). Values below
   1.2e-38 flush to zero on GPUs with flush-to-zero; only pixels and bands many decades below
   the peak are affected.
7. **Exponentials of large arguments: reduce in FP64, exponentiate in KP.** exp(x) with |x| ~ 100
   (nebular continuum: a log grid of ~1e-44 times a weight of ~1e50) loses |x| eps if x is rounded
   to FP32. `exp_kp` (`opensps_csp_batch.f90`) forms x in FP64, splits x = n ln 2 + r there, and
   takes exp(r) in KP times 2**n: ~1 ulp of KP for three FP64 operations instead of an FP64
   exponential. Weights that leave the FP32 range (rule 5) are divided by their largest value
   per model, whose logarithm joins x. The result is still a KP exponential: where the argument
   itself is a large optical depth, e**(-tau) keeps a relative error of ~tau eps (see Caveats).

8. **Share the FP64 part of a kernel among the models that need it.** On consumer and workstation GPUs
   the FP64 operations that rules 3-4 and 7 leave in a per-(pixel, model) kernel are its cost (R9700: the
   nebular continuum, 70 ms of a 190 ms batch at N = 10^4). They depend on the pixel, the CLOUDY cell and the
   age interval, not on the model: the HIP nebular kernel groups the models by (Z, U) cell (up to 8 per
   thread), forms the corner values, their differences, the interval's log step and the range reduction of
   the large exponent once per group in FP64, and does the per-model work in FP32 on small numbers (the
   interpolation offset, <~ 3 in log10; the model's own ln S, reduced once). Measured on 300-pixel
   synthetic tables with deliberately wide structure: 6.3e-7 worst per pixel (the one-thread-per-model kernel:
   2.8e-7); on the matrix cases the same as before. The emission lines use the same kernel, and the IGM
   optical depth is a sum of six wavelength x redshift terms, a table over the pixels below Lyman-alpha and six
   powers per model, instead of ~20 FP64 powers per (pixel, model) (agrees with `igm_tau` to 2e-13).

## What this means for a GPU port

The kernels above are fused multiply-adds, exponentials and reductions over spectra with
6,000-15,000 pixels: compute- or bandwidth-bound work that runs at the FP32 rate.

- Consumer and workstation GPUs: 16-64x more arithmetic throughput in FP32 than in FP64, and
  half the memory traffic. This is the difference between these cards being unusable and being
  competitive with data-centre GPUs for batched model evaluation.
- Data-centre GPUs (A100, H100, which have full-rate FP64): up to 2x (FP32 vs FP64 rate), plus
  half the memory traffic and footprint, i.e. twice as many models resident per batch.
- The SSP sum over stars and the CSP sum over ages are matrix-vector products; batched over
  models they become matrix-matrix products, the shape GPUs execute best.

On the CPU, where this was developed, the FP32 build is slightly faster for SSP generation
(0.133 s vs 0.147 s per grid, 1 core, MIST + MILES) but slower for single CSP calls (3.3 ms vs
1.7 ms), because the compensated band integrals are serial per band. CPU speed is not the goal;
the default stays FP64 there.

SSP generation is computed as a sum over library templates (`accumulate_by_templates` in
`opensps_ssp.f90`): each star is a combination of at most four templates, so the IMF-weighted
template weights are aggregated in FP64 and each used template is added once; AGB stars with a
dust shell, which are bilinear (templates times the shell's transfer function), are aggregated
per dust-grid corner. This is the matrix-vector form (templates x weights); batched over ages
or models it is a dense matrix product.

## IMF weights of the batched SSP

With FP32 kernels, the batched SSP's per-model IMF bin weights (`imf_bin_weights_tab`, the IMF
free per model) take FP32 arithmetic in their power-law bins: per bin, the exponent e ln m is
formed and reduced in FP64 (x = n ln 2 + r, `exp_reduced_kp`) and the exponential and expm1
(`expm1_kp`) are FP32; the weights are stored in WP. The single-context SSP and all IMF weights
of FP64 builds stay FP64. On the R9700 (FP32 kernels, N = 10^4) this takes `full_smooth_imf` from
75.5 to 69.1 us/model (-8%) and `full_smooth_z_imf` from 118.1 to 106.9 (-9%); the A100 gains
nothing measurable.

Accuracy of the FP32 build against FP64 (CPU, `tools/compare_builds/imf_batch_dump`: 64 models
at extreme IMF parameters, MIST + MILES, z = 0.5, nebular, IGM, AGN and dust emission, each of IMF
types 1-4): spectra within 3.4e-7 relative and magnitudes within 2.5e-7 mag (the FP32 kernels'
own error, unchanged by the weights); mass within 4.2e-8 and L_bol within 5.2e-7 (the A100
measured 4e-8 and 7.3e-7). At ordinary IMF parameters the weights are good to a few FP32 ulp per
bin (2.8e-7 worst on MIST) and mass, L_bol, star counts and ionizing photons to ~3e-8 per age
(against quad precision). Mass and L_bol of an FP32 build therefore carry the same ~1e-7 noise
as its spectra; the FP64 build keeps them exact. The finite differences of the closed-form bin
integrals that give the IMF parameters' gradients use a relative step of 1e-3 in FP32 builds
(error ~1e-4, the weights' rounding ~1e-7 over the step) and 1e-6 in FP64 builds (~1e-10). The tests' tolerances follow the
kernel precision (`test_opensps_imf`: 16 ulp of FP32 per bin; `test_csp_batch`: 1e-6 for mass and
L_bol with the IMF per model).

## cue's networks (`nebular_model = 1`)

cue's networks (`src/physics/opensps_cue.f90`) were trained in FP32, and their own error against
CLOUDY is 1-5% (Li et al. 2025). Evaluated in FP32, they reproduce their FP64 evaluation to
~2e-4 dex in the lines: the line networks' PCA and output scales (up to 6 and 20) amplify the
hidden layers' rounding. That is far below the emulator's accuracy, but 5e-4 relative is outside
the 1e-5 requirement above, so by default (`cue_fp64 = 1`) FP32 builds run the hidden layers in
FP64. The weights are FP32 as trained, so FP64 builds and the FP32 builds' default compute the
same thing; the weights are converted exactly as read. The other parts:
- **PCA means and output affine** (the means are ~50-100, the output scales up to 20): FP64.
- **Exponentials:** formed and reduced in FP64, then taken in KP (`exp_kp`).
- **The power-law fit:** FP64.

Measured against cue's FP64 algebra:

| | Lines | Continuum |
|---|---|---|
| FP64 build | 2.5e-13 dex | 1e-13 dex |
| FP32 build, `cue_fp64 = 1` | 3e-13 dex | 1e-13 dex |
| FP32 build, `cue_fp64 = 0` | 2e-4 dex | 3e-6 dex |

`cue_fp64 = 0` is an explicit opt-in for speed (timings in `doc/GPU_RESULTS.md`).

## Caveats

- **Finite differences.** FP32 output carries ~1e-7 relative noise. Fitters that estimate
  gradients by finite differences with small steps (e.g. scipy's default ~1.5e-8) need FP64
  output or larger steps. The FP64 build remains the default.
- **Run-to-run reproducibility.** Parallel (OpenACC, and GPU) reductions sum in an order that
  depends on scheduling, so repeated calls differ by a few ulps of the kernel precision: ~1e-16
  with FP64 kernels, ~1e-7 with FP32. Bitwise-reproducible FP32 output would need fixed-order
  (e.g. blocked, then sequential) reductions.
- **Large optical depths.** A transmission e**(-tau) computed in FP32 carries a relative error of
  ~tau eps (tau itself is rounded at KP). Since the attenuation runs in KP (2026-11), pixels behind
  optical depths of several differ from FP64 by up to ~9e-7 (stress cases: dust2 = 6, a 5 Myr
  population behind dust1 = 3), and GPU and CPU FP32 builds by as much (8.4e-7 measured on the A100,
  6.9e-7 on the R9700); magnitudes stay within
  3e-7 mag. Such pixels are suppressed by e**-5 or more, so this is far below any observational
  uncertainty; forming tau in FP64 would bring back the FP64 power laws the change removed.
- **Coverage.** The evidence is the matrix cases and the stress cases above
  (`tools/compare_builds/stress_cases.psv`); new physics must be checked the same way before it
  runs in an FP32 kernel.
- **Tests.** In FP32 builds, unit-test tolerances that assumed FP64 use `KP_RTOL`
  (100 eps of KP), and the regression suite compares against the FP64 references with
  `FSPS_TEST_RTOL = 1e-4` and an absolute floor of 1e-4 for spectral indices (weak indices near
  zero). CI runs gfortran and nvfortran (OpenACC multicore) with FP32 kernels.

## Reproducing

```
meson setup build_fp64 --buildtype=release
meson setup build_fp32 --buildtype=release -Dkernel_precision=single
# run both over the parameter matrix with build_*/generate_opensps_fixtures and compare the
# OPSF outputs; against legacy FSPS:
BUILD=build_fp32 tools/legacy_parity/run_matrix.sh LEGACY_SRC LEGACY_SPS_HOME out_fp32
```
