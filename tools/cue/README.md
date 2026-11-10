# cue weights for libopensps

[cue](https://github.com/yi-jia-li/cue) (Li et al. 2025, ApJ 986, 9; arXiv:2405.04598) emulates
CLOUDY nebular emission (138 lines and the nebular continuum). It has neural networks for an
ionizing spectrum described by a 4-part power law, and free gas density, metallicity, N/O and C/O.
libopensps evaluates cue's networks itself (`src/physics/opensps_cue.f90`), on the CPU or the
GPU, when the context option `nebular_model` is 1. The weights are not distributed with
libopensps: convert them from a cue checkout once and point libopensps at the result.

```
git clone https://github.com/yi-jia-li/cue
python3 tools/cue/convert_cue.py cue ~/opensps-cue          # NumPy only
export OPENSPS_CUE_DATA=~/opensps-cue
```

`OPENSPS_CUE_DATA` may name the output directory (a data root holding
`libraries/nebular_emulator/cue`) or the library directory itself. The conversion reads cue's
pickles without TensorFlow, scikit-learn or cue. It writes the 17 networks and PCA bases
unchanged: the networks in FP32 as trained, the PCA bases in FP64 as fitted. Each array is one
raw file, with the manifest of `doc/DATA_FORMAT.md` (kind `nebular_emulator`). cue is MIT
licensed; the manifest records the commit converted.

## Options

- `--reference` also stores reference outputs for the tests of libopensps (`tests/physics/test_opensps_cue.f90`):
  - the networks evaluated in FP64 by cue's algebra (`Speculator.log_spectrum_`, then the PCA's
    `inverse_transform`), for 64 inputs in the training box;
  - if SciPy and dill can be imported, cue's power-law fit (`cue.utils.fit_4loglinear_ionparam`)
    on synthetic ionizing spectra. That fit is both as released (SciPy's SLSQP) and with its loss
    minimized to convergence. On some segments SLSQP stops early (a slope off by up to ~0.1), so
    the converged minimum is the reference.

## What libopensps does with them

For each model of a batch:

1. **The ionizing source** is the model's stars of the nebular ages, the ages the CLOUDY grids
   cover, up to 20 Myr.
   - Their summed spectrum below 911.6 Å is fitted with cue's 4-part power law. It is cue's own
     fit, solved to convergence by Newton's method.
   - The photons they emit, times (1 - frac_obrun), normalize the emission (cue's
     `gas_logqion`).
2. **The networks** take the 7 power-law parameters and the gas parameters:
   - `gas_logu`, `gas_logn` (log n_H / cm^-3), `gas_logz` (log (O/H)/(O/H)_sun);
   - `gas_logno`, `gas_logco` (log N/O and C/O relative to solar).

   They return the lines and the continuum for log Q(U, n_H); scaled by Q / Q(U, n_H), as cue
   does, these become L_sun and L_sun/Hz.
3. **Lines** go to the context's line list. Each cue line goes to the nearest list line within a
   relative 1.5e-3, at vacuum wavelengths. 130 of the 166 FSPS lines receive one; the others
   are zero. N III] 1750, Si III] 1883 and 1892, and S IV 1406 have no FSPS counterpart.
4. **The continuum** is splined to the spectral grid as cue does: not-a-knot cubic,
   extrapolated, none at or below 911.6 Å.
5. **Lines and continuum** are split between the young and old age sums in proportion to their
   absorbed photons, so they attenuate with the stars that ionize them, as the CLOUDY grids do.

## Precision

- **FP64 builds:** everything is FP64, and the networks match cue's FP64 algebra to ~3e-13 dex.
- **FP32 builds:** the networks' hidden layers are FP64 by default (`cue_fp64 = 1`), so FP32
  builds match FP64 builds to the 1e-5 of `doc/MIXED_PRECISION.md`.
- **`cue_fp64 = 0`** runs the hidden layers in FP32, at roughly half the networks' cost. The
  continuum then differs by ~3e-6 dex. The lines differ by up to ~2e-4 dex (5e-4 relative): the
  line networks' output scales (up to 20) amplify the hidden layers' FP32 rounding. That is far
  below cue's own accuracy (1-5%), but outside the 1e-5 requirement, hence opt-in.

## Limits

- **Not with the IMF per model** (`ssp_ops`), and not with gradients: the batch returns status 2.
- **Without the weights** (`OPENSPS_CUE_DATA`), the batch returns status 4.
- **Outside cue's training box** (n_H 10-10^4 cm^-3, log U -4 to -1, log Z_gas -2.2 to 0.5,
  N/O and C/O 0.1-5.4 solar), cue extrapolates. The power-law parameters are clipped to their
  training ranges, as cue clips them.
