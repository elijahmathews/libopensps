# FSPS C Driver API

This document describes the C driver API exposed by FSPS. This API allows FSPS to be used as a shared library in C, C++, Python, Julia, Rust, and other languages supporting C interoperability.

## Build

From the repository root:
```sh
meson setup build
meson compile -C build
```

The C header is located at `include/fsps.h`.

Language bindings that build the library as a Meson subproject use
`dependency('fsps')` (or the `fsps_dep` variable) and usually pass `-Dtests=false` (skip the
test programs) and `-Dinstall_data=false` (do not install the 1.3 GB data tree). Installed, the
library is found through pkg-config (`fsps.pc`).

## Architecture

The driver exposes two distinct APIs:

1.  **Context API (Recommended):** Uses opaque integer handles to manage independent, thread-safe FSPS instances.
2.  **Legacy API:** Manages a single global state (mirroring the original Fortran module behavior). Retained for backward compatibility.

## 1. Context API (Thread-Safe)

The Context API allows you to create multiple isolated instances of FSPS. This is the only way to use FSPS in a multi-threaded environment.

### Lifecycle

* **Create a context:**
    ```c
    void fsps_context_create(int *handle, int *status);
    ```
    Allocates a new FSPS instance and returns a positive integer `handle`.

* **Setup physics:**
    ```c
    void fsps_context_setup(int zin, const char *isoc, const char *spec, const char *dust, int handle, int *status);
    ```
    Loads the necessary data files for the context.
    * `zin`: Metallicity index (1 to `nz`). Use `-1` to load all metallicities (required for interpolation).
    * `isoc`, `spec`, `dust`: String identifiers for libraries (e.g., "MIST", "MILES", "DL07"). Pass empty strings to use defaults.

* **Destroy a context:**
    ```c
    void fsps_context_destroy(int handle, int *status);
    ```
    Frees memory associated with the handle.

### Configuration

Parameters are set using key-value pairs. The keys correspond to the FSPS parameter names (e.g., `tage`, `imf_type`, `logzsol`).

**Performance Toggles (MCMC):**
* `compute_mags`: Set to `0` to skip magnitude calculations (Default: `1`).
* `compute_indices`: Set to `0` to skip spectral index calculations (Default: `1`).

```c
void fsps_context_set_int(int handle, const char *key, int value, int *status);
void fsps_context_set_float(int handle, const char *key, double value, int *status);
void fsps_context_set_str(int handle, const char *key, const char *value, int *status);

```

### API version

`FSPS_API_VERSION_MAJOR`, `_MINOR` and `_PATCH` in `fsps.h` equal what
`fsps_get_driver_version` returns at run time, so a binding can check that the header it was
built against matches the library it loads. The minor number grows when entry points are added
(1.1: the grid getters, the tabular SFH and the single CSP below).

### Grids and filter data

```c
void fsps_context_get_lambda(int handle, double *lambda, int *status);      // nspec, A
void fsps_context_get_ages(int handle, double *log_age, int *status);       // ntfull, log10(yr)
void fsps_context_get_zlegend(int handle, double *z, int *status);          // nz, Z
void fsps_context_get_emlambda(int handle, double *emlambda, int *status);  // nemline, A
void fsps_context_get_filter_data(int handle, double *wave_eff, double *mag_vega,
                                  double *mag_sun, int *status);            // nbands each
```

Wavelengths are in air with `vactoair_flag = 1`, in vacuum otherwise. `mag_vega` is the AB
magnitude of Vega (AB - Vega offset per band); `mag_sun` the absolute AB magnitude of the Sun.
Any of the three filter pointers may be `NULL`. Status 1 for a bad handle, 301 before setup.

### Single CSP

```c
void fsps_context_set_sfh_tab(int handle, int ntab, const double *age, const double *sfr,
                              const double *met, int *status);
void fsps_context_compute_csp(int handle, int zcontinuous, int *n_out, int *status);
void fsps_context_get_csp(int handle, double *age, double *mass, double *lbol, double *sfr,
                          double *mdust, double *mformed, double *spec, double *mags,
                          double *emlines, double *indx, int *status);
```

`fsps_context_compute_csp` runs `compute_csp_scenario` (the reference single-model path) for the
context's parameters. `zcontinuous` selects the SSPs: 0 the grid at `zmet`; 1 the grids
interpolated linearly in log Z at `logzsol`; 2 a metallicity distribution (`pmetals`); 3 all
metallicities, with the metallicity column of a tabular SFH. SSP grids are computed per
metallicity when first needed and kept with the handle until an SSP parameter is set (the
nebular parameters count only with `add_neb_emission = 2`), so a sequence of calls that changes
only SFH, dust or redshift parameters does not rebuild them. `*n_out` is 1 for `tage > 0` and
`ntfull` (one output per SSP age) otherwise.

`fsps_context_get_csp` copies the outputs out; any pointer may be `NULL`. Arrays are
column-major (`spec` is `nspec x n_out`, `mags` `nbands x n_out`, `emlines` `nemline x n_out`,
`indx` `nindx x n_out`); ages in log10(yr), masses in M_sun, L_bol in log10(L_sun), spectra in
L_sun/Hz, lines in L_sun, all per solar mass formed except for tabular SFHs (`sfh = 2, 3`),
whose SFRs are absolute (M_sun/yr).

The tabular SFH takes ages in Gyr since the start of star formation and SFRs in M_sun/yr; `met`
may be `NULL`.

Status codes: 301 not set up, 302 no CSP computed yet, 304 `zmet` out of range, 306 bad
`zcontinuous`, 309 magnitudes or indices requested but not computed (fast mode), 311 the
computation failed (or `sfh = 3` without a table); 3 from `fsps_context_set_sfh_tab` for `ntab`
outside 1 to `NTABMAX`.

### Young and old components of the batch

```c
void fsps_context_compute_csp_batch_parts(int handle, int n, int n_params, const char *const *names,
                                          const double *const *values, const double *ssp_weights,
                                          double *mags, double *spec, double *mass, double *lbol,
                                          double *mdust, double *sfr, double *emlines, double *indx,
                                          double *young, double *old, double *mass_young,
                                          double *mass_old, double *lbol_young, double *lbol_old,
                                          int *status);
```

The batch (same arguments and outputs as `fsps_context_compute_csp_batch`), plus the intrinsic
spectra of its young (ages <= `dust_tesc`) and old parts, `nspec x n` each, in L_sun/Hz per
solar mass formed, for applying dust afterwards (emulator training sets). They are what the
batch's attenuation stage receives: the age sums with nebular emission included (nebular
emission belongs to the young ages: their ionizing flux is replaced by it, as in the full
model), before dust, dust emission, AGN, IGM and smoothing. Without those four,

    spec = (young * (T_birth * (1 - frac_obrun) + frac_obrun) + old * T_dust3)
           * (T_diffuse * (1 - frac_nodust) + frac_nodust)

with `T = exp(-tau)`, `tau_birth = dust1 * (lambda / 5500 A)^dust1_index` (0 for `dust1 = 0`),
`tau_diffuse = dust2 * curve(lambda)` and `tau_dust3 = dust3 * curve(lambda)`, `curve` the
attenuation curve of `dust_type` (for `dust_type = 0`, `(lambda / 5500 A)^dust_index`). Dust
emission (energy balance), the AGN torus, IGM absorption and smoothing act on the attenuated
total. `mass_young`, `mass_old` are the surviving stellar masses and `lbol_young`, `lbol_old`
the log10 stellar L_bol of the two parts (they add up to `mass` and `lbol`); these four may be
`NULL`, `young` and `old` may not (status 3). Tested in `tests/test_c_batch_parts.c`, which
rebuilds the full spectrum from the parts.

### SSP grid

```c
void fsps_context_compute_ssp(int handle, double *spec, double *mass, double *lbol, int *status);

```

* **Inputs:** `handle`
* **Outputs:**
* `spec`: Spectrum array. Dimensions: `[nspec, ntfull]`.
* `mass`: Surviving mass array. Dimensions: `[ntfull]`.
* `lbol`: Bolometric luminosity array. Dimensions: `[ntfull]`.


* **Note:** The caller is responsible for allocating the output buffers before calling this function. Use the Metadata functions below to determine the required sizes.

### Metadata & Dimensions

Use these functions to determine array sizes for allocation.

```c
// Get basic dimensions
void fsps_context_get_dims(int handle, int *n_spec, int *n_time, int *status);

// Individual getters
void fsps_context_get_nspec(int handle, int *n_spec, int *status);
void fsps_context_get_ntfull(int handle, int *n_time, int *status);
void fsps_context_get_nbands(int handle, int *n_bands, int *status);
void fsps_context_get_nindx(int handle, int *n_indices, int *status);
void fsps_context_get_nz(int handle, int *n_z, int *status);
void fsps_context_get_nemline(int handle, int *n_line, int *status);

// Get context-specific properties
void fsps_context_get_zsol(int handle, double *z_sol, int *status);
void fsps_context_get_paths(int handle, char *sps_home, int sps_len,
                            char *data_home, int data_len,
                            char *output_home, int out_len);

```

---

## 2. Legacy API (Global State)

This API manipulates the module-level global variables. It is **not** thread-safe.

### Initialization & Cleanup

```c
void fsps_initialize(int zin);
void fsps_finalize(void);

```

### Configuration

```c
void fsps_set_int(const char *key, int value);
void fsps_set_float(const char *key, double value);
void fsps_set_str(const char *key, const char *value);

// Advanced Setters
void fsps_set_sfh_tab(int ntab, double *age, double *sfr, double *met);
void fsps_set_mag_compute(int n_bands, int *mask);
void fsps_set_ssp_gen_age(int n_age, int *mask);
void fsps_set_ssp_lsf(int nsv, double *sigma, double wlo, double whi);

```

### Computation

Computes the model based on current global parameters (including `sfh` setting).

```c
void fsps_compute(double *spec_out);

```

### Data Access

These functions retrieve data calculated by the last call to `fsps_compute`.

```c
void fsps_get_mags(double zred, double *mags);
void fsps_get_stats(double *age, double *mass, double *lbol, double *sfr, 
                    double *mdust, double *mformed, double *emlines);
void fsps_get_spec(double *spec); // Retrieve internal buffer

```

---

## Error Handling

Most functions return a `status` integer.

* `0`: **FSPS_STATUS_OK**
* `>0`: Error or Warning code.

To retrieve the descriptive error message for the last failure:

```c
void fsps_get_last_error(int *status, char *message, int message_len);
void fsps_clear_error(void);

```

### Status Codes

| Range | Type | Description |
| --- | --- | --- |
| **0** | OK | Success |
| **100-199** | Error | Unknown parameter key |
| **200-299** | Warning | Parameter out of physical range (clamped or ignored) |
| **300-399** | Error | Initialization error or computation failure |
| **400-499** | Error | Lock/Threading error |

Enable debug printing to stdout:

```c
fsps_set_debug(1);

```

## Array Layout

All arrays are passed as flat C buffers. Internally, FSPS uses Fortran column-major order.

* **1D:** Continuous.
* **2D `[X, Y]`:** The `X` index varies fastest (contiguous in memory).
* **3D `[X, Y, Z]`:** `X` varies fastest, then `Y`.

**Common Dimensions:**

* `spec`: `[nspec, ntfull]`
* `mags`: `[nbands, ntfull]`
* `ssp_spec`: `[nspec, ntfull, nz]`

## Locking

The driver provides a cooperative lock. This is **not** a system mutex; it is a flag to help coordinate access if mixing Context and Legacy calls within a single process.

```c
void fsps_lock(int *status);
void fsps_unlock(int *status);

```

## Minimal Example (C)

```c
#include "fsps.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
    int handle, status, nspec, nt;
    
    // 1. Create Context
    fsps_context_create(&handle, &status);
    
    // 2. Setup (Load all metallicities, default libraries)
    fsps_context_setup(-1, "", "", "", handle, &status);
    
    // 3. Get Dimensions to allocate buffers
    fsps_context_get_dims(handle, &nspec, &nt, &status);
    double *spec = (double*) malloc(nspec * nt * sizeof(double));
    double *mass = (double*) malloc(nt * sizeof(double));
    double *lbol = (double*) malloc(nt * sizeof(double));
    
    // 4. Set Parameters (e.g., Solar metallicity)
    fsps_context_set_int(handle, "zmet", 1, &status);
    
    // 5. Compute SSP
    fsps_context_compute_ssp(handle, spec, mass, lbol, &status);
    
    if (status != 0) {
        char msg[256];
        fsps_get_last_error(&status, msg, 256);
        printf("Error: %s\n", msg);
    }

    // 6. Cleanup
    fsps_context_destroy(handle, &status);
    free(spec); free(mass); free(lbol);
    
    return 0;
}

```