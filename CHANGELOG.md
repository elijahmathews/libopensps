# Changelog

Changes of each release of `libopensps` that a user or a packager can see, newest first. The C API
carries the release's version (`OPENSPS_API_VERSION_*`, `opensps_get_api_version`); while the
major version is 0, a minor release may change the API, and each such change is listed here under
"Breaking". Pull request numbers refer to <https://github.com/elijahmathews/libopensps>.

## Unreleased

### Changed

- **A library routine no longer stops the host program.** The 16 `stop` statements of the library (an SSP generated before
  `sps_setup`, a metallicity or IMF type out of range, an unreadable user IMF file, a full isochrone array, a tabular SFH
  that cannot be loaded, BPASS asked for an operator) are statuses now: the Fortran routines take an optional `stat`
  (`OPENSPS_STAT_*` of `opensps_constants`), and the C API returns them (`OPENSPS_ERR_SSP_*`, 501 - 508, with the message in
  `opensps_get_last_error`). The legacy global-state C entry points set the error and return. A Fortran caller that passes no
  `stat` gets the message on standard output and an early return instead of the stop.

### Added

- **IMF mass limits per model**: `imf_lower_limit` and `imf_upper_limit` may differ from model to model in a batch with the
  context's SSP operators (like the other IMF parameters), forward and with gradients of the other parameters. They are now
  members of the parameter set (`params`), kept in step with the context's settings; a batch without operators, or with a
  user IMF file (which has its own limits), refuses other limits per model (status 2).
- **Isochrone modification factors free per model** (#45): `opensps_context_build_ssp_ops_free_mods` builds the operators of
  the modification groups, with which `fbhb`, `sbss`, `agb`, `pagb`, `redgb` and the TP-AGB shifts `delt`, `dell` vary per
  model in a batch (with the IMF and the metallicity), forward and with gradients (rows `OPENSPS_BGRAD_FBHB` ..
  `OPENSPS_BGRAD_DELL`; the shifts' rows are central differences over 5e-4 dex, 1e-3 with FP32 kernels: the TP-AGB spectra
  switch templates at thresholds; at most 8 distinct nonzero pairs per batch, 4 with every metallicity; **`OPENSPS_NBGRAD`
  is now 44**, size `params_bar` from `opensps_get_nbgrad()` or use the `_rows` entry point). The paths without it are unchanged.
  `doc/ISO_MODS_PLAN.md`, `doc/C_API.md`.
- **Shared IMF weights in the batched SSP's forward pass** (#45): models of a batch that share their IMF parameters
  (the operator path, e.g. the isochrone factors free with one IMF) have the point weights computed once per distinct IMF
  and copied; the numbers are the same bit for bit. Batches below `SHARE_IMF_MIN_BATCH` models (default 1024;
  `OPENSPS_TUNE_SHARE_IMF_MIN_BATCH`) take the path as before. 3 - 13% per model on the R9700 from N = 1000.
- **`opensps_context_compute_csp_batch_grad_rows`**: the gradient entry point with the row count of the caller's
  `params_bar`. With fewer rows than `opensps_get_nbgrad()` it returns status 3 and writes nothing, instead of the
  heap overrun that a binding sized for fewer rows gets from `opensps_context_compute_csp_batch_grad` (#98).
  `opensps_batch_options_t` gains `params_bar_rows` (0: the header's `OPENSPS_NBGRAD`), and
  `opensps_context_compute_csp_batch_ext` uses the new entry point. Bindings should move to it.
- **`compute_csp_batch` refuses models whose isochrone modifications differ from the context's** (status 2) when the
  SSPs come from the context's resident basis, which has the context's: the Fortran entry point used to ignore
  them silently (the C API already refused them).

- **`-Dtuning_overrides=NAME=VALUE,...`** builds with other values of the compile-time tuning parameters
  (`SEG_LEN`, `PT_PIECE`, the tile sizes, ...), for sweeps and tuners, not for releases (`doc/TUNING.md`). A name the
  registry does not define is refused at configuration. Every build lists its compile-time parameters
  (`tuning_report`, and `tuning_ct_<NAME>` lines of `benchmark_csp_batch`). No change to the defaults.

- **One module for the hardware tuning parameters** (`src/core/opensps_tuning.f90`, `doc/TUNING.md`). The sizes that
  were literals in ten places are named constants there. Three run-time ones (`SSP_BATCH_CHUNK`, `HOST_SFH_MAX`,
  `PINNED_CHUNK_BYTES`) can be set with `OPENSPS_TUNE_<NAME>`. Defaults are the previous values: no change in
  results or speed unless a variable is set.
- **Device profiles for the tuning parameters** (`OPENSPS_TUNE_PROFILE`, `doc/TUNING.md`): per-architecture values
  measured on the device and cited in the document, chosen from the build's `-Dgpu_arch` (`cc80`, `gfx1201`).
  Built for the R9700, batches below 160 models form the SFH weights on the host (the host path measured
  up to 1.4x faster there; the default is 16, measured on the A100). `OPENSPS_BENCH_SIZES` sets the batch sizes of
  `benchmark_csp_batch`. The register tiles and other kernel sizes that were literals are named
  constants of the registry; `unit_tuning_literals` fails on a literal tuning parameter outside it.
- **Bandpass curves of your own** (`doc/FILTERS_PLAN.md`). A filter set (name, curve in Angstrom, photon- or
  energy-counting detector, optional calibration) is built with `opensps_filterset_*` and installed in a set-up
  context with `opensps_context_set_filters`, at any time and without a new setup; `opensps_context_set_filters_library`
  installs any `filters/<name>` library. Curves are cleaned and checked, resampled onto the model grid so that the
  integral of the response is conserved on any grid (the legacy point interpolation stays for the legacy list),
  normalized and calibrated; a filter that lies beyond the model wavelength range or has no throughput is refused
  by default instead of being truncated and renormalized silently. Each filter has a SHA-256 content hash
  (`opensps_context_get_filter_report`, with coverage, points across the passband and the resampling used), verified against a
  library's manifest. `tools/opensps_data/filters_tool.py` packs SVO Filter Profile Service VOTables or text
  files into such a library. The default list is bit-identical to before. New status codes 320-324.
- **Linear fluxes**: with `keep_maggies = 1`, `opensps_context_get_batch_maggies` returns the batch's AB fluxes in
  maggies (10^(-0.4 m), no logarithm and back). Magnitudes stay AB by default.
- The Vega system of a user filter set can have no reference band (Vega is magnitude 0 in every band);
  the default list keeps the legacy anchor on band 1.

## 0.2.1

A bug-fix release: no C API entry point or signature changes (the one new status value, 5, is returned only where
the call used to give wrong results or crash).

### Fixed

- **`H0`, `om0` and `ol0` set on a context now change the batched magnitudes** (#95). They were accepted but the
  distance table was built once, at setup, from FSPS's defaults (H0 = 72, Omega_m = 0.27): fluxes at a redshift in
  another cosmology were off by 8% at z = 0.1 for WMAP9. The table (and the age of the universe) is rebuilt at
  the next computation after a change. **Results change for code that set these parameters** (they were
  ignored). Invalid values (H0 <= 0, Omega_m < 0, NaN) are now refused with status 104.
- **Setting an SSP-sensitive parameter to the value it already has no longer drops the SSP basis** (#96). A client
  that sets every parameter before every call (Prospector does) rebuilt the basis each time: a single-model call
  took 20 ms instead of 2 ms for `imf_type`. A real change still invalidates.
- **An offload build whose target regions run on the host** (no device, or `OMP_TARGET_OFFLOAD=DISABLED`) no longer
  returns wrong magnitudes (#97). In an AMD build the native HIP band-flux kernel was called with host pointers
  and never wrote its output (the HIP kernels are compiled in whenever `hipcc` is found, with or without
  `-Daccel_hip`); with no device it now takes the OpenMP kernel. What cannot run on the host, `compute_csp_batch`
  refuses with **status 5**: LLVM flang's libomp crashes in `teams distribute parallel do` at one OpenMP thread
  (#87, no workaround here), and the HIP gradients, the IMF per model and cue take device pointers.

## 0.2.0

Data: `data-v3` (the regression references; `FSPS_DATA_RELEASE` pins it for CI) and `data-v4`
(adds the [α/Fe] libraries as optional archives: `tools/fetch_data.py DEST --release data-v4
--with isochrones/mist_afe stellar_spectra/c3k_lr`, 0.45 GB compressed).

### Added

- **[α/Fe] as an SSP axis** (#92). `afe` is a per-model parameter of `compute_csp_batch`, a linear
  blend of the neighbouring grid planes; `opensps_build_afe_basis` (Fortran) and
  `opensps_context_build_afe_basis` (C) build the resident stack of bases over [α/Fe] and
  metallicity. Gradients with respect to `afe` and `logzsol` (row `BGRAD_AFE`). Libraries whose
  (log T, log g) grid is not the default one load (C3K-LR, 80 × 14). The libraries are FSPS v4's
  MIST v2.5 (`mist_afe`) and C3K-LR (`c3k_lr`); `doc/DATA_FORMAT.md` and `doc/C_API.md` describe
  the format, the API and what the axis cannot be combined with yet (status 2: the IMF per model,
  weights over ages and metallicities; nebular emission uses the model's [Fe/H]).
- **cue nebular emission** in the batched CSP, `nebular_model = 1` (#89), with gradients through
  the networks, the ionizing-spectrum fit and the spline (#90; rows `BGRAD_GAS_LOGN`, `_LOGNO`,
  `_LOGCO`). `tools/cue/README.md`.
- `use_lw_tpagb` (default 1: the previous behaviour; legacy FSPS v4 uses 0).
- `opensps_get_build_id` and the `Build:` line of the banner: the commit a development build was
  made from (#78). `opensps_context_set_mag_compute` (which bands to compute) and
  `opensps_context_set_lsf` (see "Breaking").
- `tools/legacy_parity/afe`: a reproducible comparison with legacy FSPS v4's [α/Fe] build.

### Changed

- Faster batches on GPUs (#83, #88): band fluxes with a workgroup per band for small batches,
  nebular emission grouped by grid cell, a factored IGM, a band mask, cheaper parameter setup.
- The regression references are regenerated for the C3K library (`references/sps` v2,
  `references/param` v2, in `data-v3`), see "Fixed".

### Breaking

- **`OPENSPS_NBGRAD` grew from 33 to 37** (rows `gas_logn`, `gas_logno`, `gas_logco`, `afe`). The caller allocates
  `params_bar` (NBGRAD x n) and the library writes every row, with no capacity argument: a binding that sizes the
  buffer with a constant overruns it silently (wrong gradients, heap corruption). Size it with
  `opensps_get_nbgrad()`; the row count may grow in any minor release (#98).
- **FP32 builds only:** the batched IMF bin weights are always formed in FP32 arithmetic (the
  `fp32_imf_weights` build option is gone), so the mass and L_bol of the IMF-free batch are no
  longer FP64-exact there (#84); cue's hidden layers default to the kernel precision
  (`cue_fp64 = 0`; lines differ from the FP64 evaluation by 1.7e-5 to 5e-4 dex, #90).
- **The instrumental line spread function** is supplied through the context
  (`opensps_context_set_lsf`; `opensps_set_ssp_lsf` feeds the same table). `lsf.dat` is no
  longer read, and `smooth_lsf = 1` always convolves in velocity (#84). Legacy FSPS convolves
  with a 99 Å Gaussian when `smooth_velocity = 0`; this is listed in `tools/legacy_parity/README.md`.
- **Results that change:** libraries with filler corners of 1e-70 in the (log T, log g) grid (C3K)
  now treat them as missing, as legacy FSPS does (its `tiny30`): hot-star spectra of C3K change
  (log t < 6.6 up to 15-30%, the post-AGB UV 70%); CSP spectra change by 9e-5, magnitudes by
  6e-5. The other libraries are unchanged. The AGB dust screen acts only above an optical depth
  of 1e-70.

### Fixed

- Small batches in an offload build with one OpenMP thread no longer crash in libomp (#87);
  flang offload builds link again (`isoc_is_mist` is a device routine).
- Band fluxes are reproducible and race-free for small batches on NVIDIA (#81); contexts stay in
  place when the device pool grows (#79); the IMF weights of every model of a chunk are computed
  on HIP, not the first 32 (#86); nightly jobs fetch their data (#82).
- `mist_afe` takes the MIST branches (Wolf-Rayet wind radius, horizontal-branch tags).

### Known limitations

- Isochrone modifications per model (#45), per-device tuning of the batched SSP kernels (#46)
  and the single-call latency on GPUs (#49) are not done.
- LLVM flang offload builds run on the host (`OMP_TARGET_OFFLOAD=DISABLED`, or no device) with
  wrong batched band magnitudes (they vary with the thread count) and, at one thread, crash in
  libomp in the parallel regions that fall back to the host. Use a CPU build (no `-Daccel`) for
  host runs.
- AMD gfx1103 (the 780M) and Intel GPUs: OpenMP offload does not work yet.

## 0.1.0

First release of `libopensps`: FSPS in modern Fortran, built with Meson, with a C API, batched
SSPs and CSPs, reverse-mode gradients, OpenMP target offload and the data releases of
`elijahmathews/opensps-data`.
