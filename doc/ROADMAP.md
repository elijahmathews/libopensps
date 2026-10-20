# Roadmap

The single source of truth for what comes next (updated 2026-10-19). Related documents keep
their own scope: `doc/GPU_PLAN.md` is the runbook for GPU sessions (builds, validation, rules
learned), `doc/GPU_RESULTS.md` records measurements, and `doc/MIXED_PRECISION.md` the FP32
evidence. When a priority below is finished, move its outcome to the relevant document and
delete it here.

## Where things stand

- **SSP generation:** template-sum form with exact IMF integrals for every IMF type; one SSP
  grid (one metallicity, MIST + MILES) takes 0.11 s on one laptop core and 0.07 s with the
  template product on an A100.
- **Batched CSP** (`compute_csp_batch`, C entry point `fsps_context_compute_csp_batch`): full
  physics per model (SFH parameters or an SSP-weight matrix, dust, dust emission, nebular gas
  parameters, AGN, metallicity, redshift and IGM), with smoothing, indices and line
  luminosities. A100, FP32, N = 10^4, magnitudes only: 7.1 us per model with full physics and
  smoothing (6.5 with cuBLAS, 7.8 with OpenMP target), 2.9 us basic. Spectra back on the host
  add ~1.7 us per model at kernel precision.
- **Single calls:** one full-physics `compute_csp_scenario` call takes 1.4 ms on the A100; a
  batch of one 0.83 ms.
- **Accuracy:** FP64 GPU = CPU to 1.6e-11 per pixel; FP32 kernels reproduce FP64 to ~4e-7
  per pixel and 5e-7 mag. Legacy FSPS parity is checked case by case
  (`tools/legacy_parity`).
- **Portability:** OpenACC and OpenMP target on NVIDIA (all suites pass); the OpenMP path
  builds for AMD (gfx1103) with LLVM flang, where the batch tests are being debugged
  (flang mapping semantics; see priority 5).

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
  core as a Meson subproject pinned to a tag, or links an installed `libfsps` found through
  pkg-config; NumPy arrays in and out, the batch entry point with one array per parameter,
  device-resident outputs where available, and a layer compatible with python-fsps
  (`StellarPopulation`) so that Prospector and existing scripts can switch with few changes.
- Julia, R and others later, on the same C API.
- Anything a binding needs from the core (a getter, a version query) is added here, with C
  tests, in its own small pull request.

### 2. One code path

Today the single call (`compute_csp_scenario`) and the batch (`compute_csp_batch`) are separate
implementations of the same physics (agreeing to 6.7e-16 and tested against each other), and
the SSP generation has its own host and device variants. Every feature is written twice, and
gradients would have to be written twice. Goal: one hot path.

- `compute_csp_scenario` becomes a batch of one (`tage > 0`) or a batch of `ntfull` columns
  sharing one parameter set (`tage <= 0`). The mapping and the blockers (`redshift_colors`,
  light-weighted ages, MDF sums over several metallicities, invalid-dust semantics, non-fast
  smoothing) are listed in the design note in `doc/GPU_RESULTS.md` ("compute_csp_scenario as
  a batch of one"). Start with `tage > 0` and the common options.
- The CPU path runs the same kernels (OpenMP host threads instead of `-Daccel=multicore`).
- **Retire OpenACC** once OpenMP target meets the criteria below; until then OpenACC is
  frozen: no new OpenACC-only features, new kernels get OpenMP directives first.
- **Likelihood on the device:** given observed fluxes, uncertainties and a mask, return chi^2
  (or log-likelihood) per model without copying spectra back. A sampler needs one number per
  model, and the copy-back is now the largest per-model cost when spectra are needed.
- **During this work, prepare for gradients (priority 4):** keep the intermediates the
  adjoints need (per-stage spectra, transmissions) addressable rather than overwritten, and
  avoid hidden non-differentiable branches.

Criteria to remove OpenACC:

- [ ] OpenMP passes the unit + regression suites and the case matrix on NVIDIA (nvfortran) and
      AMD (LLVM flang / AOMP); ideally Intel (ifx).
- [ ] OpenMP within ~10% of OpenACC on the A100 at large N and in per-call overhead at N = 1
      (today 7.8 vs 7.1 us per model with full physics, 3.6 vs 2.9 basic).
- [ ] Feature parity: the single call and the SSP stages offloaded; cuBLAS through
      `use_device_addr`; asynchronous copies through `nowait` / `depend`; device-resident and
      kernel-precision outputs.
- [ ] `-Daccel=multicore` replaced by OpenMP host threading, CI updated.

Then, in one pull request: remove every `!$acc` directive and the `-Daccel=gpu` / `multicore`
options, and update `AGENTS.md`, `doc/GPU_PLAN.md`, `doc/CI.md` and the CI workflows.

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

### 5. AMD and Intel GPUs

The OpenMP path builds for the Radeon 780M (gfx1103, LLVM flang 22 + ROCm 7.1; recipe in
`doc/AMD_GPU.md` on branch `gpu/780m-omp`). Three LLVM flang mapping behaviours were found and
fixed on branches `fix/omp-device-no-malloc` -> `fix/omp-maps-llvm` (PR #13) ->
`fix/omp-flang-map-lifetimes` -> `fix/omp-flang-component-maps` (PR #16): structure members
named in `exit data` delete their parent, data left mapped after `target data` regions, and
implicit `tofrom` of the context. The rules are in `doc/GPU_PLAN.md`. Remaining:

- run the batch tests on the 780M from a text console (a GPU fault there resets the display
  GPU) and fix what remains;
- rebase the four AMD branches onto `develop` after the performance stack (PRs #10, #11, #14,
  #17), switch the batch output buffer (`o%spec`) to the `fsps_omp_data` helpers, and open one
  pull request;
- a discrete AMD GPU (R9700) and Intel (ifx, Arc Pro B70) when available.

### Further items

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
| Radeon 780M (iGPU) | ~0.2 s / ~5-10 ms | ~0.6-1 ms / ~5 ms | ~3-5 ms | ~100-250 us / ~1.5-2.5 ms |
| Intel Arc Pro B70 | ~0.15 s / ~3-5 ms | ~100 / ~400 us | ~1.5-3 ms | ~20-30 / ~150-250 us |
| AMD Radeon AI Pro R9700 | ~0.15 s / ~3-5 ms | ~80 / ~300 us | ~1.5-3 ms | ~15-25 / ~100-150 us |
| NVIDIA P100 | ~0.5 s / ~3 ms | ~100 / ~140 us | ~1.5-2.5 ms | ~15-18 / ~25-30 us |
| NVIDIA V100 | ~0.15 s / ~2-3 ms | ~70 / ~100 us | ~1-1.5 ms | ~11-13 / ~18-20 us |
| NVIDIA A40 | ~0.15 s / ~2-3 ms | ~60 / ~500 us | ~1-1.5 ms | ~12-16 / ~100-150 us |
| **NVIDIA A100** | ~0.14 s (2 x **0.07 s** + CSP) / ~2 ms | ~40 / ~50 us | **1.39 ms** (batch of one **0.83 ms**) | **7.1 / 8.5 us** (cuBLAS **6.5**, OpenMP **7.8**) |
| NVIDIA H100 | ~0.1 s / ~1.5 ms | ~20-25 / ~25-30 us | ~0.8-1.2 ms | ~3.5-4.5 / ~5 us |
| GeForce RTX 4090 / 5090 class | ~0.1 s / ~2 ms | ~40 / ~1 ms | ~1 ms | ~6-9 / ~150 us |

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
