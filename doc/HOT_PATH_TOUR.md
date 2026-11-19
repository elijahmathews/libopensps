# A tour of the hot path

For someone who knows stellar populations and not necessarily Fortran. It follows one call, from a caller's batch of models to
their magnitudes and spectra, and says where each step lives. Names are given instead of line numbers because lines move; search
for the name. `doc/HOT_PATHS.md` says which parameters may be free per model and what each costs; `doc/HOT_PATHS_PLAN.md` is the plan for
making the code match; this file is the map.

## The idea in one paragraph

A *batch* is N models (parameter sets) evaluated together against the same libraries. The same physics as one model
(`compute_csp_scenario`, which now just runs a batch), but each step runs **once over all models** (in parallel, on the GPU or on CPU
threads) instead of N times. Everything below is one routine, `compute_csp_batch` in `src/physics/opensps_csp_batch.f90`. Reading
that routine from top to bottom is reading the hot path.

## The route of a call

1. **A caller** (the C API or Fortran) fills one parameter set per model and calls `compute_csp_batch`. The C entry points are
   `opensps_context_compute_csp_batch*` in `src/bindings/c/opensps_c_api.f90` (`batch_handle` is the shared body: it turns "parameter
   names and per-model values" into one `params` structure per model).
2. **`compute_csp_batch`** does, in this order (the numbers are those of the comments in the routine):

| Step | What happens, in astronomy terms | Where (routine, file) |
|---|---|---|
| 0 | **Checks.** Sizes of every output, which parameters differ between models, anything this path cannot do (status 2/3/4/5). Decides *where the SSPs come from* (below). | `batch_check_shapes`, `batch_check_models`, `ssp_source_select`, `ssp_source_describe` (all `opensps_csp_batch.f90`) |
| 1 | **Per-model setup on the host.** Which age is the birth-cloud age, the star formation rate, whether there is dust emission, the interpolation indices into the dust-emission and AGN tables. | `batch_model_indices`, `plan_smoothing` |
| 1b | **Pack and send to the device.** The per-model integers, reals and flags go in one array each so they move in one transfer. | the `ipk`, `rpk`, `lpk`, `rout` arrays in `compute_csp_batch` |
| 2 | **SFH weights.** For each model, how much mass was formed in each SSP age bin (from tau, tage, ... or given by the caller), split at `dust_tesc` into *young* and *old* ages. | `batch_sfh_weights` (calls `sfh_weights_core`, `src/physics/opensps_sfh.f90`) |
| 3 | **Age sums.** Spectrum of each model = sum over ages of weight x SSP. Done twice per model, young and old (the dust acts differently on them). Also the mass and bolometric luminosity. | `ssp_source_age_sums`, `ssp_source_scalars`; the kernels `age_sums_host` and `age_sums_slabs.inc` (below) |
| 4 | **Nebular emission** (optional). Lyman-continuum photons Q from the young SSPs, the CLOUDY continuum and lines. | `add_nebular_batch`, `nebular_setup`, `attenuate_lines` |
| 5 | **Dust attenuation.** Birth-cloud and diffuse dust applied to the young and old spectra, giving the total spectrum and the absorbed light. | `attenuate_batch` (the curves are in `src/physics/opensps_dust.f90`) |
| 6 | **Dust emission**, **IGM absorption**, **AGN**. Energy balance puts the absorbed light back as Draine and Li emission; the IGM transmission is applied; the AGN torus is added. | `dust_emission_reductions` and `add_dust_emission`; `apply_igm_z` and `apply_igm`; `add_agn_batch` |
| 7 | **Velocity smoothing** by sigma (models with the same sigma are smoothed together). | `smooth_groups` and `smooth_columns_host` (`opensps_csp_batch.f90`); `smooth_columns` (`src/spectra/opensps_batch_photometry.f90`) |
| 8 | **Magnitudes** (and spectral indices): filter integrals of each model's spectrum at its redshift. | `compute_magnitudes_columns` (`opensps_batch_photometry.f90`), `compute_spectral_indices_columns` |
| 9 | **Reverse pass** (only if a gradient was asked): the derivative of the caller's loss with respect to every parameter, stage by stage in reverse. | `batch_chain_derivatives`, `batch_backward` and the `backward_*` routines |

The module header comment of `opensps_csp_batch.f90` describes steps 2 - 8 in physics terms and is the best companion to the table.

## Where the SSPs come from: the "SSP source"

Step 3 needs an SSP (a spectrum per age) for each model. There are three ways to get one, chosen once per batch by `ssp_source_select`:

* **The resident basis** (the usual case). The library built SSPs once for the context's parameters (IMF, isochrone modifications) at
  one or all metallicities. A model's SSP is a *blend* of those stored spectra, called **slabs**:
  1 slab (nothing varies), 2 slabs (`logzsol` varies: the two neighbouring metallicities, weights 1-d and d), or 4 slabs (`afe` varies too:
  two metallicities x two [alpha/Fe] planes). The number of slabs is **K**; each model has a small *descriptor* (which slabs, with
  what weights), formed by `ssp_source_describe` and used by every stage that needs the SSP (age sums, mass, L_bol, nebular Q).
* **The weights over metallicities** (`ssp_weights_z`): every metallicity's basis summed with its own weights.
* **The operators** (`ssp_ops`): the SSPs are not stored but computed per model from templates (`src/physics/opensps_ssp_operator.f90`,
  `opensps_ssp_batch.f90`), which is what lets the IMF, the isochrone modification factors, the TP-AGB shifts and the IMF mass limits
  be free per model.

`batch_info` (printed to standard error with `OPENSPS_DEBUG_PATH=1`) says which one a batch took and which parameters were free.

## Reading a kernel: the age sums

The simplest real kernel, `age_sums_host` (`opensps_csp_batch.f90`), is a good first read. For each model and block of 256 pixels it loops
over ages; each age adds `weight x (blend of the K slabs)` into the young or the old spectrum. The K = 1, 2, 4 cases differ only in
how many slabs the blend has. On the GPU the same sums run as `age_sums_slabs.inc`, a *template* (one text, included twice with K = 2
and K = 4) with a tile of models per GPU thread. Its adjoint (step 9) is `age_sums_adj_slabs.inc`.

## Fortran and OpenMP in the code: a glossary

* **`real(WP)`, `real(KP)`**: `WP` is always double precision; `KP` is the *kernel* precision: double by default, single (FP32) with
  `-Dkernel_precision=single`. The big spectral arrays are `KP`; setup, weights and parameters are `WP`.
* **`subroutine` / `function`, `intent(in)`, `intent(out)`**: arguments say whether a routine reads or writes them. `optional` arguments
  may be absent; `present(x)` tests it. (`compute_csp_batch` has about thirty optional outputs: each caller asks only for what it needs.)
* **Array arguments.** `x(:,:)` (*assumed shape*) carries its own sizes; `x(nspec, n_mod)` (*explicit shape*) is the plain contiguous
  block, which is what GPU kernels want. A `contiguous` or explicit-shape dummy never makes a copy.
* **`allocatable` and `pointer`**: arrays whose size is set at run time. The driver carves many per-model arrays out of a few packed ones
  with pointers (`it_tesc => ipk(1:n_mod)`), so that they move to the GPU together.
* **Modules** (`module ... contains ... end module`) hold routines and types; `use name, only: ...` imports. `type :: x ... end type`
  is a struct (`params`, `ssp_source_t`, `nebular_setup_t`).
* **`!$omp ...` lines** are OpenMP directives, read as comments by compilers without OpenMP. `target teams distribute parallel do` runs the
  loop on the GPU; `parallel do` on CPU threads; the `declare target` of a routine makes it callable from GPU code. Almost every
  loop appears with both, chosen by the preprocessor.
* **`#if defined(OMP_TARGET)` ... `#elif defined(_OPENMP)` ... `#endif`** is that choice: C-style preprocessor lines, resolved before
  compilation. About 580 of the 960 `#if` lines of `src/` are this pattern. The others select a compiler (`__flang__`, nvfortran) or a
  vendor path (`OPENSPS_HIP`). `tools/check_hotpath_structure.py` keeps their number from growing.
* **`.inc` files** (`age_sums_slabs.inc`, `age_sums_adj_slabs.inc`) are routine bodies included more than once with `#define` choosing a
  constant (K): Fortran has no templates, and a GPU kernel needs K at compile time.
* **`status`**: library routines return a status integer (or fill an optional `stat`) and never stop the program.

## When something looks wrong

* **A batch is refused (status 2, 3, 4, 5).** `batch_check_shapes` / `batch_check_models` (and the comments of `compute_csp_batch` for
  the codes) say why; the matching test is in `tests/test_csp_batch.f90`.
* **Which path did my batch take?** `OPENSPS_DEBUG_PATH=1`, or the `info` argument.
* **Does a batch equal each model alone?** `tests/test_batch_equivalence.f90` runs random subsets of free parameters against single models.
* **Are two builds the same?** `tools/bench_matrix.py --dump DIR` and `--dump-grad DIR`, then `tools/compare_forward_dump.py` and
  `tools/compare_builds/compare_grad_dump.py` (bit for bit; on the CPU compare builds made with `-Dfortran_args=-fno-tree-vectorize`,
  see `doc/HOT_PATHS_PLAN.md`).
* **Is it slow?** `tools/bench_matrix.py` times each free parameter alone, in pairs and all together (`doc/HOT_PATHS.md` has the results).

## What is not yet this simple

The driver still holds, inline, the nebular and attenuation calls, the smoothing and photometry calls and the reverse pass; the three
SSP sources are told apart by flags (`zmode`, `zw`, `imf`) in a few places instead of by the `ssp_source_t` alone. `doc/HOT_PATHS_PLAN.md`
lists what is done and what is next.
