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

`src/core/fsps_precision.f90` defines two kinds:

- `WP` (always FP64): setup, isochrones, IMF and SFH weights, masses, luminosities, all scalar
  bookkeeping, and the public API (C and Fortran interfaces stay double).
- `KP` (kernel precision; FP64 by default, FP32 with `-Dkernel_precision=single`): the
  per-pixel arrays and arithmetic of the expensive kernels.

With `KP = WP` the default build is unchanged: SSP spectra are bit-identical and CSP outputs
move by at most 2e-14 (the more accurate absorbed-luminosity integral below).

## What runs at kernel precision

| Kernel | Arrays (KP) | Notes |
|---|---|---|
| Stellar spectra: main library, WMBasic, AGB, post-AGB, WR | template libraries, per-star spectra | bilinear weights computed in WP per star, cast once |
| AGB circumstellar dust screen | `flux_dagb`, spectrum | |
| SSP sum over stars | `ssp_temp_grid`, weights | compensated sum |
| CSP sum over ages | SSP basis, `spec_young/old`, emission-line grids | weights cast once per age |
| Dust attenuation and emission | transmission, work spectra, Draine & Li shape | absorbed luminosity integrated directly |
| Nebular emission | line profiles (`gaussnebarr`), spectra | ionizing integral compensated |
| IGM, AGN torus emission | transmission, spectra | |
| Magnitudes | filter integrals | compensated sum |

Still computed in WP from the FP32 spectrum (element-wise or cheap; candidates for later):
velocity smoothing, spectral indices, the redshift interpolation before the filter integrals,
and the per-pixel attenuation-curve evaluation.

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

**Naive FP32 everywhere fails.** Building all of FSPS in FP32 (`FORCE_SINGLE_PRECISION`) gives
errors up to 50% in old far-UV spectra, 0.5 mag in UV bands, and NaNs in the nebular and BPASS
paths. The causes are in the cheap parts (isochrone handling and IMF weights, which difference
nearly equal masses; constants that overflow), which is why they stay in WP.

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
`fsps_ssp.f90`): each star is a combination of at most four templates, so the IMF-weighted
template weights are aggregated in FP64 and each used template is added once; AGB stars with a
dust shell, which are bilinear (templates times the shell's transfer function), are aggregated
per dust-grid corner. This is the matrix-vector form (templates x weights); batched over ages
or models it is a dense matrix product.

## Caveats

- **Finite differences.** FP32 output carries ~1e-7 relative noise. Fitters that estimate
  gradients by finite differences with small steps (e.g. scipy's default ~1.5e-8) need FP64
  output or larger steps. The FP64 build remains the default.
- **Run-to-run reproducibility.** Parallel (OpenACC, and GPU) reductions sum in an order that
  depends on scheduling, so repeated calls differ by a few ulps of the kernel precision: ~1e-16
  with FP64 kernels, ~1e-7 with FP32. Bitwise-reproducible FP32 output would need fixed-order
  (e.g. blocked, then sequential) reductions.
- **Coverage.** The evidence is the 36 matrix cases and 23 stress cases above; new physics
  must be checked the same way before it runs in an FP32 kernel.
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
