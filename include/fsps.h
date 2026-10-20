/*
 * fsps.h: libopensps compatibility header (deprecated).
 *
 * The C API of libopensps is declared in opensps.h, with the prefixes opensps_ (functions)
 * and OPENSPS_ (macros). This header maps the names of the elijahmathews/fsps C API to
 * them, so that existing C sources compile unchanged. It is frozen: entry points added
 * after the rename have only their opensps_ names. It will be removed in a future major
 * version; include opensps.h in new code.
 */
#ifndef FSPS_COMPAT_H
#define FSPS_COMPAT_H

/* The legacy declarations were enabled by FSPS_ENABLE_LEGACY */
#if defined(FSPS_ENABLE_LEGACY) && !defined(OPENSPS_ENABLE_LEGACY)
#define OPENSPS_ENABLE_LEGACY
#endif

#include "opensps.h"

/* Functions */
#define fsps_clear_error                         opensps_clear_error
#define fsps_compute                             opensps_compute
#define fsps_compute_csp                         opensps_compute_csp
#define fsps_compute_ssp                         opensps_compute_ssp
#define fsps_compute_ssps                        opensps_compute_ssps
#define fsps_compute_zdep                        opensps_compute_zdep
#define fsps_context_batch_spec_device           opensps_context_batch_spec_device
#define fsps_context_batch_spec_host             opensps_context_batch_spec_host
#define fsps_context_compute_csp                 opensps_context_compute_csp
#define fsps_context_compute_csp_batch           opensps_context_compute_csp_batch
#define fsps_context_compute_csp_batch_parts     opensps_context_compute_csp_batch_parts
#define fsps_context_compute_ssp                 opensps_context_compute_ssp
#define fsps_context_create                      opensps_context_create
#define fsps_context_destroy                     opensps_context_destroy
#define fsps_context_get_ages                    opensps_context_get_ages
#define fsps_context_get_csp                     opensps_context_get_csp
#define fsps_context_get_dims                    opensps_context_get_dims
#define fsps_context_get_emlambda                opensps_context_get_emlambda
#define fsps_context_get_filter_data             opensps_context_get_filter_data
#define fsps_context_get_lambda                  opensps_context_get_lambda
#define fsps_context_get_nbands                  opensps_context_get_nbands
#define fsps_context_get_nemline                 opensps_context_get_nemline
#define fsps_context_get_nindx                   opensps_context_get_nindx
#define fsps_context_get_nspec                   opensps_context_get_nspec
#define fsps_context_get_ntfull                  opensps_context_get_ntfull
#define fsps_context_get_nz                      opensps_context_get_nz
#define fsps_context_get_paths                   opensps_context_get_paths
#define fsps_context_get_zlegend                 opensps_context_get_zlegend
#define fsps_context_get_zsol                    opensps_context_get_zsol
#define fsps_context_set_fast_mode               opensps_context_set_fast_mode
#define fsps_context_set_float                   opensps_context_set_float
#define fsps_context_set_int                     opensps_context_set_int
#define fsps_context_set_sfh_tab                 opensps_context_set_sfh_tab
#define fsps_context_set_str                     opensps_context_set_str
#define fsps_context_setup                       opensps_context_setup
#define fsps_finalize                            opensps_finalize
#define fsps_get_csp_components                  opensps_get_csp_components
#define fsps_get_dims                            opensps_get_dims
#define fsps_get_driver_version                  opensps_get_driver_version
#define fsps_get_emlambda                        opensps_get_emlambda
#define fsps_get_filter_data                     opensps_get_filter_data
#define fsps_get_indices                         opensps_get_indices
#define fsps_get_isochrone_dimensions            opensps_get_isochrone_dimensions
#define fsps_get_lambda                          opensps_get_lambda
#define fsps_get_last_error                      opensps_get_last_error
#define fsps_get_libraries                       opensps_get_libraries
#define fsps_get_mags                            opensps_get_mags
#define fsps_get_mags_mask                       opensps_get_mags_mask
#define fsps_get_nbands                          opensps_get_nbands
#define fsps_get_nemline                         opensps_get_nemline
#define fsps_get_nindx                           opensps_get_nindx
#define fsps_get_nm                              opensps_get_nm
#define fsps_get_nmass_isochrone                 opensps_get_nmass_isochrone
#define fsps_get_nspec                           opensps_get_nspec
#define fsps_get_nt                              opensps_get_nt
#define fsps_get_ntabmax                         opensps_get_ntabmax
#define fsps_get_ntfull                          opensps_get_ntfull
#define fsps_get_nz                              opensps_get_nz
#define fsps_get_res                             opensps_get_res
#define fsps_get_setup_vars                      opensps_get_setup_vars
#define fsps_get_spec                            opensps_get_spec
#define fsps_get_spec_peraa                      opensps_get_spec_peraa
#define fsps_get_ssp_spec                        opensps_get_ssp_spec
#define fsps_get_ssp_weights                     opensps_get_ssp_weights
#define fsps_get_stats                           opensps_get_stats
#define fsps_get_timefull                        opensps_get_timefull
#define fsps_get_zlegend                         opensps_get_zlegend
#define fsps_get_zsol                            opensps_get_zsol
#define fsps_initialize                          opensps_initialize
#define fsps_initialize_full                     opensps_initialize_full
#define fsps_interp_ssp                          opensps_interp_ssp
#define fsps_lock                                opensps_lock
#define fsps_set_debug                           opensps_set_debug
#define fsps_set_float                           opensps_set_float
#define fsps_set_int                             opensps_set_int
#define fsps_set_mag_compute                     opensps_set_mag_compute
#define fsps_set_sfh_tab                         opensps_set_sfh_tab
#define fsps_set_ssp_gen_age                     opensps_set_ssp_gen_age
#define fsps_set_ssp_lsf                         opensps_set_ssp_lsf
#define fsps_set_str                             opensps_set_str
#define fsps_smooth_spectrum                     opensps_smooth_spectrum
#define fsps_stellar_spectrum                    opensps_stellar_spectrum
#define fsps_unlock                              opensps_unlock
#define fsps_validate_params                     opensps_validate_params
#define fsps_write_isochrone                     opensps_write_isochrone

/* Status codes and API version */
#define FSPS_API_VERSION_MAJOR                   OPENSPS_API_VERSION_MAJOR
#define FSPS_API_VERSION_MINOR                   OPENSPS_API_VERSION_MINOR
#define FSPS_API_VERSION_PATCH                   OPENSPS_API_VERSION_PATCH
#define FSPS_ERR_COMPUTE_UNINIT                  OPENSPS_ERR_COMPUTE_UNINIT
#define FSPS_ERR_CSP_SCENARIO_FAILED             OPENSPS_ERR_CSP_SCENARIO_FAILED
#define FSPS_ERR_CSP_UNINIT                      OPENSPS_ERR_CSP_UNINIT
#define FSPS_ERR_CSP_ZDEP_FAILED_0               OPENSPS_ERR_CSP_ZDEP_FAILED_0
#define FSPS_ERR_CSP_ZDEP_FAILED_1               OPENSPS_ERR_CSP_ZDEP_FAILED_1
#define FSPS_ERR_CSP_ZDEP_FAILED_2               OPENSPS_ERR_CSP_ZDEP_FAILED_2
#define FSPS_ERR_CSP_ZDEP_FAILED_3               OPENSPS_ERR_CSP_ZDEP_FAILED_3
#define FSPS_ERR_INTERP_UNINIT                   OPENSPS_ERR_INTERP_UNINIT
#define FSPS_ERR_LOCK_HELD                       OPENSPS_ERR_LOCK_HELD
#define FSPS_ERR_MAG_MASK_UNINIT                 OPENSPS_ERR_MAG_MASK_UNINIT
#define FSPS_ERR_SSP_AGE_UNINIT                  OPENSPS_ERR_SSP_AGE_UNINIT
#define FSPS_ERR_SSP_UNINIT                      OPENSPS_ERR_SSP_UNINIT
#define FSPS_ERR_SSP_Z_RANGE                     OPENSPS_ERR_SSP_Z_RANGE
#define FSPS_ERR_STELLAR_UNINIT                  OPENSPS_ERR_STELLAR_UNINIT
#define FSPS_ERR_UNKNOWN_FLOAT_PARAM             OPENSPS_ERR_UNKNOWN_FLOAT_PARAM
#define FSPS_ERR_UNKNOWN_INT_PARAM               OPENSPS_ERR_UNKNOWN_INT_PARAM
#define FSPS_ERR_UNKNOWN_STRING_PARAM            OPENSPS_ERR_UNKNOWN_STRING_PARAM
#define FSPS_ERR_UNLOCK_NOLOCK                   OPENSPS_ERR_UNLOCK_NOLOCK
#define FSPS_ERR_ZDEP_TYPE                       OPENSPS_ERR_ZDEP_TYPE
#define FSPS_ERR_ZDEP_UNINIT                     OPENSPS_ERR_ZDEP_UNINIT
#define FSPS_STATUS_OK                           OPENSPS_STATUS_OK
#define FSPS_WARN_CONST_FBURST                   OPENSPS_WARN_CONST_FBURST
#define FSPS_WARN_DUST_RANGE                     OPENSPS_WARN_DUST_RANGE
#define FSPS_WARN_IMF_RANGE                      OPENSPS_WARN_IMF_RANGE
#define FSPS_WARN_MAG_MASK_RANGE                 OPENSPS_WARN_MAG_MASK_RANGE
#define FSPS_WARN_NMASS_RANGE                    OPENSPS_WARN_NMASS_RANGE
#define FSPS_WARN_SF_START_GT_TAGE               OPENSPS_WARN_SF_START_GT_TAGE
#define FSPS_WARN_SSP_AGE_RANGE                  OPENSPS_WARN_SSP_AGE_RANGE
#define FSPS_WARN_ZCONTINUOUS_NEB                OPENSPS_WARN_ZCONTINUOUS_NEB
#define FSPS_WARN_ZMET_RANGE                     OPENSPS_WARN_ZMET_RANGE

#endif /* FSPS_COMPAT_H */
