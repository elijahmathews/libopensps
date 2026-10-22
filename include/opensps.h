#ifndef OPENSPS_H
#define OPENSPS_H

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// API version (also returned by opensps_get_driver_version). The minor number grows when entry
// points are added; the major number changes only for incompatible changes. 1.1 added the
// context grids (lambda, ages, zlegend, emlambda, filter data), the tabular SFH and the single
// CSP (opensps_context_compute_csp / opensps_context_get_csp). 1.2 added the batch's young and old
// components (opensps_context_compute_csp_batch_parts). 2.0 dropped light-weighted ages:
// setting compute_light_ages to a value other than 0 returns OPENSPS_ERR_UNSUPPORTED_PARAM.
// ----------------------------------------------------------------------------
#define OPENSPS_API_VERSION_MAJOR 2
#define OPENSPS_API_VERSION_MINOR 0
#define OPENSPS_API_VERSION_PATCH 0

// ----------------------------------------------------------------------------
// Status Codes
// ----------------------------------------------------------------------------
#define OPENSPS_STATUS_OK 0
#define OPENSPS_ERR_UNKNOWN_INT_PARAM 101
#define OPENSPS_ERR_UNKNOWN_FLOAT_PARAM 102
#define OPENSPS_ERR_UNKNOWN_STRING_PARAM 103
#define OPENSPS_ERR_UNSUPPORTED_PARAM 104
#define OPENSPS_WARN_ZMET_RANGE 201
#define OPENSPS_WARN_DUST_RANGE 202
#define OPENSPS_WARN_IMF_RANGE 203
#define OPENSPS_WARN_SF_START_GT_TAGE 204
#define OPENSPS_WARN_CONST_FBURST 205
#define OPENSPS_WARN_ZCONTINUOUS_NEB 206
#define OPENSPS_WARN_NMASS_RANGE 207
#define OPENSPS_WARN_MAG_MASK_RANGE 208
#define OPENSPS_WARN_SSP_AGE_RANGE 209
#define OPENSPS_ERR_COMPUTE_UNINIT 301
#define OPENSPS_ERR_CSP_UNINIT 302
#define OPENSPS_ERR_SSP_UNINIT 303
#define OPENSPS_ERR_SSP_Z_RANGE 304
#define OPENSPS_ERR_ZDEP_UNINIT 305
#define OPENSPS_ERR_ZDEP_TYPE 306
#define OPENSPS_ERR_INTERP_UNINIT 307
#define OPENSPS_ERR_STELLAR_UNINIT 308
#define OPENSPS_ERR_MAG_MASK_UNINIT 309
#define OPENSPS_ERR_SSP_AGE_UNINIT 310
#define OPENSPS_ERR_CSP_SCENARIO_FAILED 311
#define OPENSPS_ERR_CSP_ZDEP_FAILED_0 312
#define OPENSPS_ERR_CSP_ZDEP_FAILED_1 313
#define OPENSPS_ERR_CSP_ZDEP_FAILED_2 314
#define OPENSPS_ERR_CSP_ZDEP_FAILED_3 315
#define OPENSPS_ERR_LOCK_HELD 401
#define OPENSPS_ERR_UNLOCK_NOLOCK 402

// ----------------------------------------------------------------------------
// Driver Utilities (Logging, Version, Locking)
// ----------------------------------------------------------------------------
void opensps_set_debug(int flag);
void opensps_clear_error(void);
void opensps_get_last_error(int *status, char *message, int message_len);
void opensps_lock(int *status);
void opensps_unlock(int *status);
void opensps_get_driver_version(int *major, int *minor, int *patch);
void opensps_validate_params(int *status); // Note: Checks legacy global state context

// ----------------------------------------------------------------------------
// Context Management (Thread-Safe API)
// ----------------------------------------------------------------------------
void opensps_context_create(int *handle, int *status);
void opensps_context_destroy(int handle, int *status);
void opensps_context_setup(int zin, const char *isoc_type, const char *spec_type,
                        const char *dust_type, int handle, int *status);

// Parameter Setters
void opensps_context_set_int(int handle, const char *key, int value, int *status);
void opensps_context_set_float(int handle, const char *key, double value, int *status);
void opensps_context_set_str(int handle, const char *key, const char *value, int *status);
void opensps_context_set_fast_mode(int handle, int fast_mode, int *status);

// Dimension Queries
void opensps_context_get_dims(int handle, int *n_spec, int *n_time, int *status);
void opensps_context_get_nspec(int handle, int *n_spec, int *status);
void opensps_context_get_ntfull(int handle, int *n_time, int *status);
void opensps_context_get_nbands(int handle, int *n_bands, int *status);
void opensps_context_get_nindx(int handle, int *n_indices, int *status);
void opensps_context_get_nz(int handle, int *n_z, int *status);
void opensps_context_get_nemline(int handle, int *n_line, int *status);
void opensps_context_get_zsol(int handle, double *z_sol, int *status);

// Grids and filter data of a set-up context (status 1 bad handle, 301 not set up)
// lambda: nspec wavelengths [A] (air with vactoair_flag = 1); log_age: ntfull SSP ages,
// log10(yr); z: nz metallicities (mass fraction); emlambda: nemline line wavelengths [A];
// filter data, nbands each, any may be NULL: effective wavelength [A], AB magnitude of Vega,
// absolute AB magnitude of the Sun.
void opensps_context_get_lambda(int handle, double *lambda, int *status);
void opensps_context_get_ages(int handle, double *log_age, int *status);
void opensps_context_get_zlegend(int handle, double *z, int *status);
void opensps_context_get_emlambda(int handle, double *emlambda, int *status);
void opensps_context_get_filter_data(int handle, double *wave_eff, double *mag_vega,
                                  double *mag_sun, int *status);

// Tabular SFH for sfh = 3: ntab points of age [Gyr], SFR [M_sun/yr] and metallicity Z (used
// with zcontinuous = 3; met may be NULL). status 3 if ntab is out of range.
void opensps_context_set_sfh_tab(int handle, int ntab, const double *age, const double *sfr,
                              const double *met, int *status);

// Path Queries
void opensps_context_get_paths(int handle, char *sps_home, int sps_len,
                            char *data_home, int data_len, char *output_home, int out_len);

// Context Computation
void opensps_context_compute_ssp(int handle, double *spec, double *mass, double *lbol, int *status);

// Single CSP for the context's parameters (the reference path, compute_csp_scenario).
// zcontinuous: 0 SSPs at zmet; 1 interpolated in log Z at logzsol; 2 weighted by a metallicity
// distribution (pmetals); 3 all metallicities (tabular SFH metallicity). SSP grids are computed
// per metallicity when first needed and kept until an SSP parameter is set. *n_out is 1 for
// tage > 0, else ntfull (one output per SSP age). status 301 not set up, 304 zmet out of range,
// 306 bad zcontinuous, 311 computation failed.
void opensps_context_compute_csp(int handle, int zcontinuous, int *n_out, int *status);

// Outputs of the last opensps_context_compute_csp (n_out each; any pointer may be NULL),
// column-major, per solar mass formed: age log10(yr), surviving stellar mass [M_sun], log10
// L_bol [L_sun], SFR [M_sun/yr], dust mass [M_sun], mass formed [M_sun], spec (nspec x n_out,
// L_sun/Hz), mags (nbands x n_out), emlines (nemline x n_out, L_sun), indx (nindx x n_out).
// status 302 nothing computed, 309 mags / indx requested with compute_mags / compute_indices 0.
void opensps_context_get_csp(int handle, double *age, double *mass, double *lbol, double *sfr,
                          double *mdust, double *mformed, double *spec, double *mags,
                          double *emlines, double *indx, int *status);

// Batched CSPs (fitting pipelines): n models of the context's parameters with n_params of them
// varied per model, names[i] taking the n values values[i][0..n-1]; ssp_weights (ntfull x n) or
// NULL replaces the SFH parameters. Outputs column-major (quantity x n); all but mags may be
// NULL. status 0 on success, 1 bad handle, 2 unsupported (e.g. an SSP parameter varied),
// 3 unknown parameter or bad shape.
void opensps_context_compute_csp_batch(int handle, int n, int n_params, const char *const *names,
                                    const double *const *values, const double *ssp_weights,
                                    double *mags, double *spec, double *mass, double *lbol,
                                    double *mdust, double *sfr, double *emlines, double *indx,
                                    int *status);

// The batch with its intrinsic young and old components as well (e.g. emulator training sets
// that apply birth-cloud and diffuse dust afterwards). Arguments as opensps_context_compute_csp_batch,
// plus young, old (nspec x n, column-major, required): the age sums of the ages <= dust_tesc and
// > dust_tesc, L_sun/Hz per solar mass formed, with nebular emission included (it belongs to the
// young ages, whose ionizing flux it replaces), before dust, dust emission, AGN, IGM and
// smoothing. Without those, spec = (young (T_birth (1 - frac_obrun) + frac_obrun) + old T_dust3)
// (T_diffuse (1 - frac_nodust) + frac_nodust), with T = exp(-tau): tau_birth = dust1 (lambda /
// 5500 A)^dust1_index, tau_diffuse = dust2 curve(lambda), tau_dust3 = dust3 curve(lambda), curve
// the dust_type's attenuation curve. mass_young, mass_old: surviving stellar masses [M_sun];
// lbol_young, lbol_old: log10 L_bol [L_sun] of the stars (log10 of the smallest double when a
// part is empty); any of these four may be NULL. status as the batch, 3 also for NULL young / old.
void opensps_context_compute_csp_batch_parts(int handle, int n, int n_params, const char *const *names,
                                          const double *const *values, const double *ssp_weights,
                                          double *mags, double *spec, double *mass, double *lbol,
                                          double *mdust, double *sfr, double *emlines, double *indx,
                                          double *young, double *old, double *mass_young,
                                          double *mass_old, double *lbol_young, double *lbol_old,
                                          int *status);

// The last batch's spectra where they were computed, for callers that keep working on the GPU
// (pass spec = NULL to the batch to skip the copy to the host): *ptr is a device address (host
// address in CPU builds) of nspec x n values, column-major, *bytes per value (4 in FP32-kernel
// builds, 8 in FP64), L_sun/Hz per solar mass formed. Valid until the next batch call on the
// handle; *ptr is NULL and *n 0 before the first. status 1 for a bad handle.
void opensps_context_batch_spec_device(int handle, void **ptr, int *nspec, int *n, int *bytes,
                                    int *status);

// The same spectra copied to a page-locked host buffer owned by the context, in kernel precision
// (no conversion to double: the fastest way to get the spectra on the host when single
// precision is enough). *ptr is a host address; valid until the next batch call on the handle.
void opensps_context_batch_spec_host(int handle, void **ptr, int *nspec, int *n, int *bytes,
                                  int *status);

// ----------------------------------------------------------------------------
// Legacy Global State API (Backward Compatibility)
// ----------------------------------------------------------------------------

// Initialization
#ifdef OPENSPS_ENABLE_LEGACY
void opensps_initialize(int zin);
void opensps_initialize_full(int zin, int compute_vega_mags0, int vactoair_flag0,
                          const char *isoc_type, const char *spec_type, const char *dust_type);
#endif

// Parameter Setters (Global)
void opensps_set_int(const char *key, int value);
void opensps_set_float(const char *key, double value);
void opensps_set_str(const char *key, const char *value);

// Specific Setters
void opensps_set_sfh_tab(int ntab, double *age, double *sfr, double *met);
void opensps_set_mag_compute(int n_bands, int *mask);
void opensps_set_ssp_gen_age(int n_age, int *mask);
void opensps_set_ssp_lsf(int nsv, double *sigma, double wlo, double whi);

// Computation (Operates on Global State)
void opensps_compute(double *spec_out); // Main driver (SSP or CSP based on `sfh`)
void opensps_compute_csp(int zcontinuous);
void opensps_compute_ssp(int zin);
void opensps_compute_ssps(void);
void opensps_compute_zdep(int ztype);

// Interpolation
void opensps_interp_ssp(double zpos, double tpos, double *spec, double *mass, double *lbol);

// ----------------------------------------------------------------------------
// Data Access & Post-Processing
// ----------------------------------------------------------------------------

// Photometry
void opensps_get_mags(double zred, double *mags);
void opensps_get_mags_mask(double zred, double *mags, int *mask);

// Spectral Data
void opensps_get_spec(double *spec);
void opensps_get_spec_peraa(double *spec);
void opensps_get_indices(double *spec, double *indices);
void opensps_smooth_spectrum(double *wave, double *spec, double sigma_broad, double minw, double maxw);

// Physical Properties
void opensps_get_stats(double *age, double *mass, double *lbol, double *sfr, 
                    double *mdust, double *mformed, double *emlines);
void opensps_get_ssp_weights(double *wghts);
void opensps_get_csp_components(double *young, double *old);
void opensps_get_ssp_spec(double *spec, double *mass, double *lbol);

// Stellar Utility
void opensps_stellar_spectrum(double mact, double logt, double lbol, double logg, 
                           double phase, double ffco, double lmdot, double wght, double *spec);

// Metadata / Arrays
void opensps_get_dims(int *n_spec, int *n_time);
void opensps_get_nbands(int *n_bands);
void opensps_get_nindx(int *n_indices);
void opensps_get_nspec(int *n_spec);
void opensps_get_ntfull(int *n_time);
void opensps_get_nt(int *n_time);
void opensps_get_nm(int *n_mass);
void opensps_get_ntabmax(int *n_tabmax);
void opensps_get_nz(int *n_z);
void opensps_get_nemline(int *n_line);

// Array Getters
void opensps_get_zsol(double *z_sol);
void opensps_get_zlegend(double *zlegend);
void opensps_get_timefull(double *timefull);
void opensps_get_lambda(double *lambda);
void opensps_get_emlambda(double *emlambda);
void opensps_get_res(double *res);
void opensps_get_filter_data(double *wave_eff, double *mag_vega, double *mag_sun);
#ifdef OPENSPS_ENABLE_LEGACY
void opensps_get_setup_vars(int *compute_vega_mags, int *vactoair_flag);
void opensps_get_libraries(char *isoc, int isoc_len, char *spec, int spec_len, char *dust, int dust_len);
#endif

// Isochrone Specifics
void opensps_get_isochrone_dimensions(int *n_age, int *n_mass);
void opensps_get_nmass_isochrone(int z_idx, int t_idx, int *n_mass);
void opensps_write_isochrone(const char *outfile);

// Cleanup
void opensps_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // OPENSPS_H