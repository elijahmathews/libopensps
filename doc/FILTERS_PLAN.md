# Bandpass (filter) curves

Status: **implemented** (this document began as the design sketch, 2026-11-15; Section 9 records the
decisions it was implemented under). Reference for users: `include/opensps.h` (entry points),
`doc/C_API.md` ("Filters and linear fluxes"), `doc/DATA_FORMAT.md` (the `filters` library kind).

## 1. What there was

One file of 159 stacked filters (`allfilters.dat`, or the standard-format library `filters/allfilters`),
or one alternative legacy-format file with exactly as many filters. The curves were interpolated onto the
model wavelength grid at setup and normalized, and the result (`bands`, `magsun`, `magvega`,
`filter_leff`) lived in the **shared setup cache**, keyed on the file name. A curve partly outside the
grid was truncated and renormalized without a word; calibrations were applied by filter index; only
photon-counting curves could be expressed; nothing identified which curve a result was computed with.

## 2. Principles

1. **The kernels do not change.** Everything happens when filters are set, never in a kernel: every source
   ends as the same `bands(nlambda, nbands)` array, photon form, normalized to integral T/lambda = 1. No FP32,
   device-code or cache-layout consequence.
2. **The core takes arrays, in Angstrom.** No network, VOTable parser or `sedpy` in `libopensps`; those are
   in `tools/` and in the language bindings, which also convert units (m, nm, um, Hz) before the call.
3. **Shape and calibration are separate, explicit inputs**, so a curve from any source means one
   reproducible thing.
4. **The legacy list goes through the same path**, bit-identically (checked on the real data: `bands`,
   `magvega`, `magsun` and `filter_leff` of a default setup have the same SHA-256 as before this change).
5. **Never silently change a bandpass.** Truncation by the grid, curves finer than the grid and dead filters
   are reported and, for user curves, refused by default.

## 3. The filter set (`src/spectra/opensps_filters.f90`)

Per filter: `name`, the curve (`lambda` [A], `response`), `detector`, optional `calibration`
{`lambda_ref`, `exponent`}. The curve is cleaned (repeated wavelengths dropped, the first kept; negative
response clamped to zero; a decreasing curve reversed; NaN/Inf, wavelengths <= 0, non-monotonic
wavelengths and fewer than two distinct points refused), then:

- **Detector.** The kernels evaluate the photon-counting form `<f_nu> = int f_nu T dlambda/lambda /
  int T dlambda/lambda`. An energy counter has `<f_nu> = int f_nu T dnu / int T dnu = int f_nu (T/lambda^2)
  dlambda / int T/lambda^2 dlambda`, the photon form with T' = T/lambda. The loader resamples T onto
  the grid (so the piecewise-linear curve is represented exactly) and then divides the grid values by
  lambda: the same sampling the kernels' quadrature uses. SVO's `DetectorType` 1 (photon counter) is
  detector 0, its 0 (energy counter) is detector 1.
- **Resampling.** `RESAMPLE_LINEAR` is the legacy point interpolation (the legacy list's, bit-identical).
  `RESAMPLE_CONSERVING` (default for user curves) projects the piecewise-linear curve on the hat functions
  of the grid, `T_i = int T h_i dlambda / w_i`, with `w_i` the trapezoid weight of node i, evaluated
  exactly (Simpson's rule on the product of two linear functions). Then `sum_i w_i T_i` equals the integral
  of T over the grid range for any grid, uneven ones included (tested to 1e-10), and the means of a smooth
  spectrum agree between grids of 0.5, 1 and 4 A to 1e-6 .. 2e-5. Point sampling does not: a sharp-edged
  filter on a coarse grid gains or loses edge area.
- **Normalization.** Integral T'/lambda = 1; a zero integral is a dead filter (flagged; before, silent).
- **Calibration.** Division by integral (lambda/lambda_ref)^exponent T'/lambda dlambda (exponent -1 flat
  nu f_nu, -2 beta = 2 blackbody): the manifest's existing field, now available to every source.
- **Zero points and pivot wavelength** as before (Vega and the Sun in every band; `sqrt(int lambda T /
  int T/lambda)`). They are always computed from the data release's Vega and solar spectra, never taken
  from a provider's header (SVO's zero points depend on their Vega model and detector convention; the
  tool keeps them as provenance only). The legacy list keeps its order (zero points from the
  uncalibrated bands: `legacy_order`); a user set calibrates first, so its zero points are those of
  the bandpass actually used.

### 3.1 Report and policy

Each filter reports its **coverage** (fraction of the integral of T/lambda for photon counters,
T/lambda^2 for energy counters, that lies on the model wavelength grid, exact for the piecewise-linear
curve), the number of grid points above half its maximum (`n_fwhm`) and flags: truncated (coverage <
0.999), undersampled (`n_fwhm` < 5), dead.

- 0.999: below any plausible difference between tabulations of one filter, above the truncation of a
  tail that merely reaches past the grid end. 5 points: the trapezoid rule needs several nodes across a
  passband; fewer is reported, not refused, since the conserving resampling still conserves the throughput.
- `grid_policy` `ERROR` (default for user curves) refuses a truncated or dead filter; `WARN` and `ALLOW`
  report only (`ALLOW` is the legacy list's behaviour).

### 3.2 The content hash

SHA-256 (own implementation, `src/math/opensps_sha256.f90`, tested against the FIPS 180-4 vectors including
one million `a`) of: the tag `OPENSPS-FILTER-v1`, detector, calibrated flag (int64), `lambda_ref` and
`exponent` if calibrated, n (int64), then wavelengths and response (binary64), all little-endian, `-0`
as `+0`: the **cleaned** curve, so it is the hash of what is used. Names and provenance are not hashed, so
one curve from SVO, from `sedpy` or from a file hashes alike. The report also gives the hash of the
wavelength grid and the resampling and policy used. `tools/opensps_data/filters_tool.py` has the
reference implementation in Python; a test checks both against each other on a library the tool wrote, and
a library whose stored hash differs from its curve is refused (status 323).

## 4. Entry points

- **Arrays** (`opensps_filterset_create/add/count/free`, `opensps_context_set_filters`): the set is copied into
  the context, which can be done **after setup**, any number of times, with no new setup. The arrays
  (`bands`, `magsun`, `magvega`, `filter_leff`, the report) belong to that context; the shared setup cache
  and every other context are untouched (tested). `nbands`, the band mask and every output follow the new
  set; the batch's per-band tables are rebuilt; a GPU-resident context is taken off the device and put back
  (**not run on a GPU**). A new setup restores the default list. All computation is done on new arrays first,
  so an error leaves the context as it was.
- **Library** (`opensps_context_set_filters_library(handle, name, ...)`): any standard-format
  `filters/<name>`; the manifest gives per filter `name`, `detector` ("photon"/"energy"), `calibration`,
  `hash`. (`allfilters` keeps the legacy list's zero-point order.)
- **Tools and wrappers.** `tools/opensps_data/filters_tool.py pack|svo` writes such a library from SVO
  VOTables (TABLEDATA) or text files, in any wavelength unit, recording SVO's own metadata as
  provenance; `svo` downloads first (not exercised against the live service; tests use no network). The
  Python package (`opensps-python`, its own repository) is to build a set from `sedpy` filters (duck-typed
  on `.wavelength`, `.transmission`, `.name`, `.detector_type`; to be checked against the sedpy version it
  pins), arrays, VOTables and SVO IDs, and to expose the hashes for recording with fit results.

## 5. The Vega system with a user set

With `compute_vega_mags = 1` the legacy convention is anchored on band 1 (V: it stays AB, the others are
shifted by their Vega offsets from it). A user's first band is not V, so a set carries
`vega_ref_band`: **0 (default for arrays and libraries): Vega is magnitude 0 in every band**; k > 0 the legacy
convention anchored on band k; the default list keeps 1. `get_filter_data` returns the AB magnitude of Vega
relative to that anchor. AB magnitudes are the default (`compute_vega_mags = 0`).

## 6. Linear fluxes

`keep_maggies = 1` makes every batch keep its linear AB fluxes, `maggies = f_nu / 3631 Jy = 10^(-0.4 m_AB)`
(distance term and redshift included, independent of the Vega setting), formed from the band flux
directly: no logarithm and back, and 0 where the magnitude is NaN (a band without flux). Read with
`opensps_context_get_batch_maggies`. It is a setting and a getter, not a new argument, so no existing
entry point changes; off by default (nothing is computed or stored). Single CSPs run as batches and are
covered. The gradient entry point is unchanged (cotangents of the magnitudes).

## 7. Tests

`unit_filters` (Fortran; 71 checks): SHA-256 vectors; cleaning; hashes; conservation and grid
independence of the resampling; both detectors against analytic means of f_nu = 1/lambda; coverage,
dead and undersampled filters; calibration; on a context of the small test library: the default list
installed through the new path is **bit-identical** (bands, zero points, pivots, hashes, magnitudes),
a flat f_nu gives AB = 0 in every band of a user set, Vega's own spectrum gives one offset from `magvega` in
every band, the Vega system with and without a reference band, batched and single-spectrum photometry
agree on the new set, errors leave the context as it was, another context and a re-setup are unaffected,
the maggies. `unit_c_filters` (C, through `tests/run_filters_library_test.py`): the whole C API, a library
written by `filters_tool.py` with hashes cross-checked against Python, status codes 3, 301, 320-324,
batch magnitudes with the new bands, maggies = 10^(-0.4 m), masks, Vega. `unit_data_tools`: the tool.

## 8. Decisions (maintainer, 2026-11-15)

1. Filters can be replaced after setup. 2. `conserving` resampling and an `error` grid policy for user
curves. 3. Photon and energy counters from the start. 4. The core takes Angstrom; unit handling is the
wrappers'. 5. Content hash of the filter as used. 6. SVO-specific code under `tools/`. 7. AB magnitudes
are the default; linear fluxes available (Section 6).

## 9. Not done

- A `filter_set` argument of `opensps_context_setup`: a library is selected after setup
  (`opensps_context_set_filters_library`), so no setup signature changed. `alt_filter_file` is unchanged.
- The default-context (global-state) C API has no filter-set entry point.
- Run on the R9700 (gfx1201, LLVM flang, `OMP_TARGET_OFFLOAD=MANDATORY`, FP64 and FP32): `unit_filters` and
  `unit_c_filters` pass with the context resident on the device, so installing a set takes the
  `remove_from_device` / `move_to_device` path, and the full unit and regression suites give the results of the
  unmodified base (the 3 failures against the pinned references). That run found that the linear fluxes were
  read from a device-only array (fixed: `fband` is copied back when `maggies_out` is requested). Not run on
  the A100 (nvfortran), and no GPU-against-CPU comparison of a user filter set over many models.
- `opensps-python` (`FilterSet`, `sedpy`/SVO adapters) is a separate repository.
