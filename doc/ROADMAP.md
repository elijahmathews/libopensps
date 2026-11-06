# Roadmap

The single source of truth for what comes next (updated 2026-10-21). Related documents keep
their own scope: `doc/GPU_PLAN.md` is the runbook for GPU sessions (builds, validation, rules
learned), `doc/GPU_RESULTS.md` records measurements, and `doc/MIXED_PRECISION.md` the FP32
evidence. When a priority below is finished, move its outcome to the relevant document and
delete it here.

## Where things stand

- **SSP generation:** template-sum form with exact IMF integrals for every IMF type, one
  template kernel for CPU and GPU builds; one SSP grid (one metallicity, MIST + MILES) takes
  0.11 s on one laptop core and 0.065 s on an A100.
- **Batched CSP** (`compute_csp_batch`, C entry point `opensps_context_compute_csp_batch`): full
  physics per model (SFH parameters or an SSP-weight matrix, dust, dust emission, nebular gas
  parameters, AGN, metallicity, redshift and IGM), with smoothing, indices and line
  luminosities. A100 (OpenMP target), FP32, N = 10^4, magnitudes only: 7.2 us per model with
  full physics and smoothing, 2.9 us basic. Spectra back on the host add ~1.7 us per model at
  kernel precision.
- **Single calls:** `compute_csp_scenario` runs as a batch by default (priority 2): one
  full-physics call with smoothing takes 1.17 ms on the A100 (OpenMP target FP32; 18 ms before
  the single call was a batch); on one CPU core 6.4 ms (12.3 ms before).
- **Accuracy:** FP64 GPU = CPU to 1.6e-11 per pixel; FP32 kernels reproduce FP64 to ~4e-7
  per pixel and 5e-7 mag. Legacy FSPS parity is checked case by case
  (`tools/legacy_parity`).
- **Portability:** OpenMP target offload on AMD (LLVM flang with ROCm; gfx1201, the R9700: unit and
  regression suites pass with `OMP_TARGET_OFFLOAD=MANDATORY`, FP64 and FP32) and on NVIDIA
  (nvfortran; cc80, the A100: all suites pass). Other GPUs, macOS, arm64 and ifx build in CI only.

## Cutover to libopensps

The code moves to a fresh repository, `elijahmathews/libopensps`, without the history and the
data of this fork. Its first commit is a snapshot: "libopensps: the state of
elijahmathews/fsps at commit <hash>". Everything after it happens in the new repository.

**Cutover commit (this repository):** the last commit on `develop` before the move, tagged
`fsps-final`. It must contain:

- [ ] the standard data format and loaders, so the code runs from fetched data without `data/`
      (priority 1), with the small test library in the repository;
- [ ] the data archives published (per-library archives, release `data-v1`) and the fetch
      script pointing at them; CI fetching and caching them for the regression and
      legacy-parity suites;
- [ ] the open pull requests that are ready merged (GPU performance stack, roadmap, C API
      additions for the bindings); work not ready by then moves to the new repository as
      patches (`git format-patch` / `git am`: the trees are identical apart from `data/`);
- [ ] a green CI run on that commit, including the nightly suites.

**First commit (libopensps):** the cutover tree without `data/` and without large generated
files, the FSPS MIT license and copyright notice unchanged, and an `AUTHORS` / credit section
naming the FSPS authors and the fork's contributors (the history that recorded them is not
carried over). The commit message names the source commit.

**Immediately after, in libopensps:**

1. Rename (one pull request): library `libopensps` (soname), header `opensps.h`, pkg-config
   `opensps`, C prefix `opensps_`, Meson project name, documentation; the bindings switch with
   a one-file change.
2. Repository settings: issues on, branch protection on `main`, CI and coverage tokens,
   description and topics; `CITATION.cff` citing FSPS and libopensps.
3. Tag a first release; point `opensps-python`'s Meson wrap at the tag.

**This repository afterwards:** a final README note pointing to libopensps, then archived
(read-only). The tag `fsps-final` keeps the full history and data reachable.

## Priorities

In order of work, with the reason for the order. Items marked *parallel* need no GPU and can
run in laptop sessions alongside the GPU work.

### 1. Data out of the repository; a standard input format

The repository carries 1.3 GB of data (spectra 0.8 GB, isochrones 0.3 GB, nebular 0.16 GB) and
1.8 GB of history. Goal: any stellar, isochrone, nebular, dust or filter library that follows a
documented format can be used, and a fresh repository started from a clean commit holds only
code, small test data and references (megabytes).

- **Format:** a versioned manifest (TOML or JSON: library kind, axes, units, wavelength grid,
  provenance, checksums) plus plain little-endian binary arrays; HDF5 optional, not required
  by the Fortran core. Arbitrary extra axes from the start ([alpha/Fe], IMF, ...), so new
  grids need no format change.
- **Adapters:** a converter from the legacy FSPS layout (`SPS_HOME` trees) to the new format,
  and a loader that still reads the legacy layout directly during the transition.
- **Distribution:** versioned data archives (e.g. Zenodo or release assets) and a fetch script
  that verifies checksums; `FSPS_DATA_HOME` points at the unpacked data.
- **Tests:** a small test library (coarse subsets, a few MB) in the repository for unit tests
  and CI; the regression references and the legacy-parity matrix, which need the full
  MIST + MILES data, run in a CI job that downloads and caches it.
- **Name:** the fresh repository is `libopensps`. Rename the core with it: library
  `libopensps` (soname), header `opensps.h`, pkg-config `opensps`, C prefix `opensps_`, Meson
  project name; keep the FSPS license, copyright notice and credit in the README and citation
  file.
- **Done when:** the suites pass from fetched data, a new library can be added without code
  changes, and a fresh repository from that commit is below ~50 MB.

Why first: it is independent of the GPU work, mostly I/O, and it defines the inputs that the
batched SSP (priority 3) reads.

### 1b. A stable C API; language bindings in their own repositories (*parallel*)

This repository provides the C API (header, shared library, pkg-config file) as the stable,
versioned interface: semantic versioning of the library, an API version macro and query
function, and a changelog of API changes. Language bindings live in separate repositories that
depend on a pinned version of the C library, each with its own release cycle and packaging.

- **Naming:** each binding follows its ecosystem's convention around the `opensps` name:
  Python `opensps-python` (`pip install opensps`, `import opensps`), Julia `OpenSPS.jl` with
  the binary as `libopensps_jll`, R `opensps-r` (package `opensps`), Rust `opensps-rs` (crates
  `opensps-sys` and `opensps`).
- **Python first** (`opensps-python`): a pip-installable package (meson-python) that builds the
  core as a Meson subproject pinned to a tag, or links an installed `libopensps` found through
  pkg-config; NumPy arrays in and out, the batch entry point with one array per parameter,
  device-resident outputs where available, and a layer compatible with python-fsps
  (`StellarPopulation`) so that Prospector and existing scripts can switch with few changes.
- Julia, R and others later, on the same C API.
- Anything a binding needs from the core (a getter, a version query) is added here, with C
  tests, in its own small pull request.

### 2. One code path (done)

Finished with the removal of OpenACC (2026-10; details in `doc/GPU_RESULTS.md`, "One code
path, steps 1-7"). The criterion of passing on AMD GPUs was dropped (2026-10-22): development
targets NVIDIA with nvfortran, and AMD offload is future work (priority 5). The CSP physics has one implementation, `compute_csp_batch`;
`compute_csp_scenario` runs as a batch (one model per output age). SSP generation has one
template kernel for every build. OpenMP target is the only GPU programming model and OpenMP
host threads the only CPU parallelism. Light-weighted ages were dropped (status 104; could
return as a batch feature); the device likelihood is under Further items.

Carried forward: keep the intermediates the adjoints of priority 4 need (per-stage spectra,
transmissions) addressable rather than overwritten, and avoid hidden non-differentiable
branches. Accepted for now (2026-10-22): OpenMP target trails the former OpenACC build by 11%
in FP64 at large N and 10-18% at N = 1 (host time in the OpenMP runtime); FP32 at large N is
level. A persistently mapped batch workspace is the untested remedy.

### 3. Batched SSP: SSP-level parameters free per model

An SSP spectrum is templates . W (W: template weights x ages). Folding the SFH first gives a
CSP = templates . (W . sfh_weights), so no SSP grid is built per model. IMF parameters touch
only the IMF weights:

1. Once per metallicity and age: the sparse matrix S (templates x isochrone points, ~4 entries
   per point, independent of the IMF), from the weights `accumulate_by_templates` gives each
   star without the IMF factor.
2. Per model: IMF weights of all isochrone points on the device (closed forms;
   `integrate_imf_interval_analytic` is already a device routine), ~2e5 evaluations.
3. W = S . w_IMF (sparse), folded with the SFH weights; then the shared template product and
   the rest of the batch. The ionizing photon rate, AGB dust shells and dust emission stay
   linear in the templates.

First step: for one metallicity, check W against `generate_ssp_grid` for a few IMF settings
before folding it into the batch. IMF *type* is best shared per batch. Isochrone modifications
(`sbss`, `fbhb`, `delt`, `dell`, `agb`, `pagb`) change which templates stars use, so S is
rebuilt per model (the expensive case). Narrow mass bins: evaluate as midpoint value x width
with a correction, so IMF weights work in FP32.

Status (2026-11-02, #32): `compute_csp_batch(..., ssp_ops=ops)` takes the IMF and the
metallicity free per model. `ops` (`ssp_opset_t`, `src/physics/opensps_ssp_batch.f90`) holds
one operator S per metallicity (`ssp_operator_build` per metallicity, then `ssp_opset_build`;
built once, ~0.1 s each); one metallicity is a set of one. The IMF
type is shared per batch; the slopes, `vdmc` and `mdave` vary per model, the mass limits are
shared. The IMF weights of every isochrone point are computed per model on the device, power-law
bins as (c/e) m1^e expm1(e ln(m2/m1)) from per-point logs (no cancellation in narrow bins: 30% of
MIST's bins are narrower than 1e-6), lognormal bins by erf or Gauss-Legendre (#43). Checked
against per-model `generate_ssp_grid` bases for every IMF type, a range of parameters, mass
limits and metallicities (`test_csp_batch`: FP64 ~1e-14 on CPU, FP32 ~6e-7). Cost at N = 10^4
(FP32, us/model): A100 19.9 (IMF free) and 32.2 (IMF and metallicity free) against 6.4-7.2 for
the resident basis; R9700 59.0 and 96.6 against 19.1 (`doc/GPU_RESULTS.md`).

Not supported per model: the isochrone modifications (`fbhb`, `sbss`, `delt`, `dell`, `agb`,
`pagb`, `redgb`, `agb_dust`, `evtype`, `masscut`). A batch whose models differ from the
operators' build returns status 2 (`ssp_op_mod_signature`). The design that keeps them on the
batch path (per-point group factors; operator sets grouped by `delt`, `dell`) is in its own issue.

Alongside, two input generalisations that reuse the same machinery:

- **Age x metallicity weight matrices:** extend the SSP-weight input from (ages x N) to (ages x
  metallicities x N), so metallicity can vary with age within a model (chemical-enrichment
  histories) and MDF sums become a special case (one of the blockers of priority 2).
- **[alpha/Fe] as a grid axis** (see priority 3b).

### 3b. [alpha/Fe] as an SSP axis (near term)

Alpha-enhanced isochrones and spectral libraries (MIST with an [Fe/H] x [alpha/Fe] layout, C3K
per [alpha/Fe]) make [alpha/Fe] a fit parameter. The priority-1 format carries it as a named
axis from the start (a library without it is [alpha/Fe] = 0), and its converter reads the
alpha-enhanced layouts. Then:

- SSP generation over (metallicity, [alpha/Fe]): the context's library and isochrone arrays
  and the metallicity legend gain an [alpha/Fe] dimension (the list of what changes is in
  `doc/DATA_FORMAT.md`);
- the batch interpolates per model in ([Fe/H], [alpha/Fe]) between resident bases, as it does
  in metallicity today (bilinear: four bases instead of two);
- nebular emission: no alpha-enhanced photoionization grids exist yet, so alpha-enhanced fits
  either use the solar-scaled nebular grids at the model's [Fe/H] (documented approximation)
  or omit nebular emission, until such grids are computed;
- gradients (priority 4) treat [alpha/Fe] like `logzsol` (kinked at grid values).

This can go before or alongside priority 3; it needs only priority 1.

### 4. Gradients

Batched CSP = SFH weights -> GEMM -> element-wise attenuation -> energy-balance integrals and
dust emission -> nebular / AGN / IGM -> filter GEMM: linear algebra plus smooth element-wise
steps. Hand-written reverse mode (transposed GEMMs, element-wise adjoints) gives the gradient of
a likelihood for ~2-4x a forward evaluation (A100: ~20-40 us per model with full physics),
storing a few intermediate spectra per model. FP32 gradients (~1e-5 relative) suffice for
HMC / NUTS and variational inference. Validate each against finite differences in the FP64
build.

| Parameter | Difficulty | Notes |
|---|---|---|
| `logmass`, `fagn` | trivial | linear |
| parametric SFH | easy | analytic derivatives of the SFR integrals |
| non-parametric SFH (weight matrix) | trivial | linear in the weights; chain to `logsfr_ratio` in the sampler |
| `dust2`, `dust1` / `dust_ratio`, `dust_index` | easy | analytic derivatives of exp(-tau); linear energy balance |
| `duste_gamma` | easy | linear mix |
| `duste_umin`, `duste_qpah`, `agn_tau` | easy, kinked | linear grid interpolation: piecewise-constant gradient |
| `gas_logz`, `gas_logu` | moderate, kinked | nebular grids; chain through the ionizing rate (linear in the SSP) |
| `logzsol` | moderate, kinked | kinks at the grid metallicities |
| `zred` | moderate | smooth distance and IGM; resampling needs the spectral derivative (noisy per pixel, smooth band-integrated) |
| `sigma_smooth` | easy | one convolution with the kernel's sigma derivative |
| IMF parameters (priority 3) | moderate | closed forms, including the integral of m^-alpha ln m; through S . w and the fold |
| isochrone modifications | hard / ill-posed | discontinuous; finite differences or fixed |

Kinks from linear grid interpolation are tolerable: reparameterised (pathwise) gradients stay
unbiased for Lipschitz functions, so HMC and variational inference work; an optional smooth
(cubic) interpolation mode would remove them at a small, validatable physics change.

Rejected alternatives: AD tools on the Fortran (Enzyme on flang IR is experimental with GPU
offload; Tapenade does not handle OpenACC / OpenMP) and a re-implementation in an AD framework
(duplicate physics, no portable Fortran path).

Why after priorities 2 and 3: the adjoints are written once, on the final hot path.

Status (2026-11-02, PRs #50-#56 and their successors, stacked on #44): `compute_csp_batch` takes the
gradient of the caller's loss with respect to its magnitudes and spectra (`mags_bar`, `spec_bar`)
and returns `params_bar(NBGRAD, N)` (rows `BGRAD_*`) and `weights_bar(ntfull, N)` (dL / d(SSP
weight), the non-parametric SFH), by a reverse pass in the same call (`batch_backward`). Each stage's
adjoint is the transpose of its forward map, evaluated with the forward's own transmissions,
tables and interpolation weights:

| Rows | Through |
|---|---|
| SSP weights, `fagn` | age sums, AGN (L_bol), magnitudes and the observed-frame resampling |
| `agn_tau` | the torus templates' interpolation (one-sided at the grid's nodes, zero outside it) |
| weights over ages and metallicities (`ssp_weights_z`: `weights_z_bar`) | the age sums over every resident basis, L_bol and the mass |
| `dust1`, `dust2`, `dust3`, `dust_index`, `dust1_index`, `frac_obrun`, `frac_nodust`, `mwr`, `uvb` | the three transmissions (the curve tables' derivatives), AGN's diffuse transmission |
| `duste_qpah`, `duste_umin`, `duste_gamma` | the energy balance (absorbed light, escaping fraction) and the Draine & Li interpolation |
| `gas_logz`, `gas_logu` (and `frac_obrun`) | the nebular continuum and lines (CLOUDY grids in log space), the EUV rows |
| `logzsol` | the bases (and L_bol, Q) interpolated in log Z, with every metallicity resident |
| `tage`, `tau`, `const`, `fburst`, `tburst`, `sf_start`, `sf_trunc`, `sf_slope` | the SFH weights' tangents (`opensps_sfh_tangent`; sfh = 0, 1, 4, 5) |
| `zred` | the resampling, the distance term, the IGM (`igm_tau_dz`) |
| `igm_factor` | the IGM transmission (-tau T, the forward's capped optical depth); per model, as `sigma_smooth` |
| `imf1`, `imf2`, `imf3`, `vdmc`, `mdave` (with `ssp_ops`) | the SSP operator in reverse (`ssp_batch_age_sums_adjoint`: templates transposed, the fold as gathers, L_bol and Q per point); the point weights' derivatives analytic for Kroupa and Dave (`imf_integral_dp`, `powerlaw_j_dj`), by central differences of their FP64 closed forms for van Dokkum (`vdmc`); only the rows each IMF type reads |
| `sigma_smooth` | the smoothing kernels' coefficients, every mode (the device operator; on the host the exact log-linear and the constant-width convolutions); one sigma per batch: per-model sensitivities |

Kinks are one-sided: interpolation nodes of the dust-emission and nebular grids, grid ages for
tage, observed pixels coinciding with rest-frame pixels for zred, z = 0. Precision: each model's
cotangents are scaled to unit maximum before they are stored at kernel precision (a magnitude of a
nearly dark band has a gradient of 1 / F), long sums are compensated in FP32 builds. Validation:
`tests/test_csp_batch_grad` against finite differences of the loss in FP64 (every row, every model,
20 configurations; agreement 1e-8 - 1e-5, limited by the differences), with linear-only checks in
FP32 builds.

Cotangents of the outputs `mass` and `lbol` are taken too (`mass_bar`, `lbol_bar`; with `ssp_ops` the
surviving mass, remnants included, is differentiated in the IMF parameters and logzsol by the
operator's central differences). SSP-level nebular emission
(`add_neb_emission = 2`) is in the basis, so its gas parameters do not act in the batch (rows zero).

Every gradient case is supported (no status 2). Not differentiated: `emlines`, `mdust`, indices as outputs.
Parameters without a row, and why: `dust_tesc` (the young/old split falls on a grid age, so the
model is a step function of it: zero gradient almost everywhere); the isochrone and SSP-build
parameters (`dell`, `delt`, `fbhb`, `sbss`, `pagb`, `redgb`, `agb`, `agb_dust`, `masscut`, `fcstar`,
`pmetals`, `frac_xrb`), which are in the bases or operators; `dust_clumps`, which no physics reads;
the smoothing window's edges; integer switches.

Cost (2026-11-03, after a first tuning pass; full physics: nebular, dust emission, AGN, IGM, SFH
parameters; forward / with gradients, per model):

| | forward | with gradients |
|---|---|---|
| gfortran FP64, one CPU thread, N = 100 | 1.35 ms | 3.50 ms (2.6x) |
| gfortran FP64, one CPU thread, IMF free (Kroupa, `ssp_ops`), N = 100 | 5.9 ms | 15.6 ms (2.6x) |
| R9700 FP32 (LLVM flang 22), N = 100 | 49 us | 181 us (3.7x) |
| R9700 FP32, N = 1000 | 24 us | 85 us (3.5x; 95 with amdflang 7.2.4 before the last change) |
| R9700 FP32, IMF free (Kroupa, `ssp_ops`), N = 1000 | 73 us | 417 us (5.7x) |

What the tuning found (rocprofv3 kernel traces on the R9700, stage timers on both): FP64 atomics in
the IMF operator's fold (4 s per call of 100 models: now gathers), FP64 arithmetic at the consumer
GPU's FP64 rate (the template transpose, the band integrals' transpose: now kernel precision,
compensated where the sums are long), accumulator arrays spilled to scratch memory (now scalar register tiles), passes
serial per model (observed frame, IGM cap, AGN, line profiles), a single-threaded host pass over the
cotangents, the IMF rows' finite differences (now analytic for Kroupa and Dave), and on CPUs
unvectorized running sums (now lane-vectorized dot products). Remaining: the nebular continuum's
table reads, the age sums' adjoint on GPUs (one basis column per thread), host-side setup, and the
SFH tangents on the host.

### 5. AMD and Intel GPUs (future work)

Deferred (2026-10-22). LLVM flang (22, with ROCm 7.1) builds `-Daccel=omp_gpu` for the Radeon
780M (gfx1103) and passes the unit and regression suites with `OMP_TARGET_OFFLOAD=DISABLED`.
On the device, `test_csp_batch` faults in the first kernel it launches (the SSP template sum,
`add_main_library_templates`): the kernels reach `ctx%state` through an implicit map of the
context, whose array descriptors flang does not attach on the device. NVIDIA tolerates this;
AMD does not.

The fixes found for LLVM flang (structure members named in `exit data` delete their parent;
data left mapped after `target data` regions; the implicit `tofrom` of the context, remedied by
`map(alloc: ctx)` in every kernel that uses it; no array-constructor temporaries; no libm
`log_gamma` on the device) are on branches of the FSPS fork (`fix/omp-device-no-malloc`,
`fix/omp-maps-llvm`, `fix/omp-flang-map-lifetimes`, `fix/omp-flang-component-maps`,
`fix/omp-flang-ctx-alloc`; build recipe on `gpu/780m-omp`) and were not ported here; even with
them, `test_csp_batch` still failed on the 780M. Remaining:

- port those fixes, with the context mapped whole or by component consistently, and every
  kernel's data named in its map clauses;
- run the suites on the 780M from a text console (a GPU fault there resets the display GPU)
  and fix what remains;
- a discrete AMD GPU (R9700) and Intel (ifx, Arc Pro B70) when available.

### Further items

- **Likelihood on the device** (moved out of priority 2 on 2026-10-21: it rests on assumptions
  about the observables): given observed fluxes, uncertainties and a mask, return chi^2 (or
  log-likelihood) per model without copying spectra back. A sampler needs one number per model,
  and the copy-back is the largest per-model cost when spectra are needed. Grid, units and the
  amplitude treatment to be agreed first (question 11 in `doc/GPU_RESULTS.md` has a proposal).

- **SFH weights at early ages:** the hat-kernel weights are differences of indefinite moments
  (`compute_ssp_weights`), good to ~5e-14 / (fraction formed) relative: 1.5e-6 at 2e-8 of the
  mass formed, so host and device math (or two compilers) differ there, and a 1-ulp change of
  the age moves truncated / Simha weights by 1e-8. Computing each bin's integral directly (e.g.
  `expm1` of the interval for the exponential terms) would make them accurate to rounding;
  matters for gradients (priority 4) and for the youngest all-ages outputs, and would let the
  all-ages SFH weights move to the device without breaking GPU = CPU (tried and reverted,
  `doc/GPU_RESULTS.md`).

- **Age-binned dust laws** (any number of age bins, each with its own attenuation curve) and
  **emission lines kept separate from the continuum until projection** (simpler smoothing and
  line-spread-function handling). Best done during priority 2.
- **Remaining performance work:** per-call scratch arrays owned by the context (9 descriptor
  uploads per batch call); the single call's ~40 small transfers (absorbed by priority 2); the
  asynchronous kernel queue (a race found in the first A100 session, still open); the host
  conversion of FP32 spectra to double (single-threaded) or handing callers kernel-precision
  spectra.
- **Performance-regression tracking:** record `benchmark_phases` and `benchmark_csp_batch` on
  every merge (CPU in CI, GPU from cluster sessions).
- **Release and a short software paper** (e.g. JOSS) once priorities 1 and 1b are done.

## Projected performance

Full physics: MIST + MILES (5994 pixels), Chabrier IMF, dust attenuation and DL07 dust
emission, nebular lines and continuum, IGM at z = 0.5, AGN torus, velocity smoothing at
150 km/s, spectral indices, 159 bands, metallicity per model; magnitudes only. **Bold** =
measured, the rest projected. GPU cells are FP32 / FP64. "SSP+CSP" frees SSP-level parameters
(e.g. the IMF) per model: *naive* builds each model's two bracketing SSP grids, *folded* is
priority 3 (not implemented).

| Hardware | SSP+CSP, single model | SSP+CSP, batched (folded) | CSP, single call | CSP, batched |
|---|---|---|---|---|
| Ryzen 7 7840U, 1 core | ~0.23 s naive / ~60 ms folded | ~60 ms | **6.7 ms** (fixed Z), ~8 ms | ~8 ms |
| 7840U, 8 cores | ~30 ms / ~8 ms | ~8 ms | ~1 ms (throughput) | ~1 ms |
| 2x Xeon Gold 6248R (48 cores) | — | ~1.5-2 ms (throughput) | ~0.15-0.2 ms (throughput) | ~0.15-0.2 ms |
| **Radeon 780M (iGPU)**, measured 2026-10-22/23 | ~0.2 s / ~5-10 ms (projected) | ~0.6-1 ms / ~5 ms (projected) | **6.9 / 9.5 ms** (median of 3; range over both series: FP32 4.0-8.6, FP64 5.9-12.8) | **217 / 284 us** (N = 10^4, median of 3; both series: FP32 214-295, FP64 280-390; 293 / 375 at N = 100) |
| Radeon 780M, projected after further optimization | ~0.2 s / ~5-10 ms | ~0.6-1 ms / ~5 ms | ~3.5-5 / ~6-9 ms | ~130-180 / ~230-300 us |
| Intel Arc Pro B70 | ~0.15 s / ~3-5 ms | ~100 / ~400 us | ~1.5-3 ms | ~20-30 / ~150-250 us |
| **AMD Radeon AI Pro R9700**, measured 2026-11-02 | **0.048 s** (SSP grid 44 ms + CSP 4 ms; FP32) | **175 / 231 us** (IMF and Z free, `feat/ssp-batch-v2-a100-zvar2`) | **6.0 / 6.3 ms** | **28.8 / 36.9 us** (N = 10^4; median of 6, spread < 4%) |
| NVIDIA P100 | ~0.5 s / ~3 ms | ~100 / ~140 us | ~1.5-2.5 ms | ~15-18 / ~25-30 us |
| NVIDIA V100 | ~0.15 s / ~2-3 ms | ~70 / ~100 us | ~1-1.5 ms | ~11-13 / ~18-20 us |
| NVIDIA A40 | ~0.15 s / ~2-3 ms | ~60 / ~500 us | ~1-1.5 ms | ~12-16 / ~100-150 us |
| **NVIDIA A100** | ~0.14 s (2 x **0.07 s** + CSP) / ~2 ms | ~40 / ~50 us | **1.05-1.1 ms** (batch path; batch of one **0.83 ms**) | **7.1 / 8.5 us** (cuBLAS **6.5**, OpenMP **7.8**) |
| NVIDIA H100 | ~0.1 s / ~1.5 ms | ~20-25 / ~25-30 us | ~0.8-1.2 ms | ~3.5-4.5 / ~5 us |
| GeForce RTX 4090 / 5090 class | ~0.1 s / ~2 ms | ~40 / ~1 ms | ~1 ms | ~6-9 / ~150 us |

The 780M row is measured with LLVM flang 22.1.8 + ROCm 7.1 (`perf/amd-780m`,
`benchmark_csp_batch 10000 200 full_smooth_z`, FP32 and FP64 builds; `full_smooth_z` is the
configuration this table describes: full physics, smoothing, metallicity per model, magnitudes
only; the A100 figures come from `full_smooth` and `full_smooth_z`, which differ by ~3% there).
The iGPU runs at 68-80 C and ~30 W under load and its time varies by up to ~30% from run to run,
so the cells are medians of three interleaved runs (a second series a day later agreed within the
range) with the observed range over both series. The R9700 row (LLVM flang 22.1.8 + Fedora ROCm
7.1.1; `doc/GPU_RESULTS.md`, 2026-11-02) is measured on `zvar2`, whose `*_z` configs load every
metallicity (`main`'s left the other bases empty: 26.4 / 34.2 us there); its "folded" cell is
`full_smooth_z_imf` (IMF and metallicity free per model), the SSP+CSP cell `benchmark_phases`.
The SSP+CSP columns
were not measured. Before the optimization work of 2026-10-22 the same runs took 12-29 ms per
single call and ~25 ms per model at N = 100, and did not finish at N = 1000.
Where the large-N time goes (FP32, N = 10^4, us per model): nebular continuum ~78 (FP64
exponentials; the 780M does 129 GFLOPS in FP64 against ~4 TFLOPS in FP32), continuum attenuation
~40 (FP64 powers and exponentials), magnitudes ~28, SSP age sums ~18, smoothing ~15, the rest ~25.
FP64 costs only ~60% more than FP32 here, so the old FP64 projection (~1.5-2.5 ms) was far too
pessimistic. A single call is ~22 kernel launches, each ~230 us of launch and kernel latency plus
a few small host-device copies; the setup of the call's data region adds ~85 more.
The projected-final row assumes the remaining known items: the FP32 form of the nebular exponential
progression, FP64 transcendental work moved out of per-pixel loops (attenuation), fewer launches
and a smaller data region per call. Each is an estimate with ~2x uncertainty, not a measurement.

Spectra on the host add per model ~1.7 us (A100, kernel precision; ~6-7 us converted to
double) or ~2-4 us on other PCIe GPUs; none on the 780M (shared memory). FP64 on the A40, R9700,
B70, 780M and consumer GPUs runs at 1/16-1/64 of FP32, so use FP32 there. Other libraries scale
with pixel count: C3K ~1.8-2x, BaSeL ~0.4-0.5x. Uncertainty: A100 ~20% (measured); other
NVIDIA ~1.5-2x (scaled by bandwidth and FP32 / FP64 rates); AMD and Intel ~3x (OpenMP path not
yet validated there); the folded columns ~3x (not implemented). Single calls are dominated by
fixed per-call overhead on every GPU: batching is what amortises it.

## Batching support against fitting pipelines

Free parameters in current pipelines: unresolved fits (`logmass`, `logzsol`, `dust2`,
`dust_ratio`, `dust_index`, the non-parametric SFH via `logsfr_ratio_i`, `fagn`, `agn_tau`,
`gas_logz`, `gas_logu`, `duste_umin`, `duste_gamma`, `duste_qpah`, `zred`); resolved fits per
pixel (`logmass`, `logzsol`, `dust2`, `dust_index`, SFH) with some global parameters.

All of these vary per model in `compute_csp_batch` (the non-parametric SFH through the
SSP-weight matrix; `logmass` and `dust1 = dust_ratio * dust2` formed by the caller), and the C
entry point takes one array per parameter. Shared per batch: the libraries and isochrones,
SSP-level parameters (until priority 3), the physics switches, the filter set and
`sigma_smooth`.
