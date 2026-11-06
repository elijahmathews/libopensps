#include "opensps.h"
#include "test_c_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_MOD 2

int main(void) {
    skip_without_data();
    int status = 0, h = 0, nbands = 0, nspec = 0, ntfull = 0, failures = 0;
    int nbgrad = 0;

    /* 1. Test metadata queries */
    opensps_get_nbgrad(&nbgrad);
    if (nbgrad != OPENSPS_NBGRAD || nbgrad != 33) {
        printf("FAIL: opensps_get_nbgrad=%d expected 33\n", nbgrad);
        failures++;
    }

    char name_buf[64];
    opensps_get_grad_param_name(OPENSPS_BGRAD_FAGN, name_buf, 64, &status);
    if (status != 0 || strcmp(name_buf, "fagn") != 0) {
        printf("FAIL: name for BGRAD_FAGN = '%s' status=%d\n", name_buf, status);
        failures++;
    }
    opensps_get_grad_param_name(OPENSPS_BGRAD_DUST2, name_buf, 64, &status);
    if (status != 0 || strcmp(name_buf, "dust2") != 0) {
        printf("FAIL: name for BGRAD_DUST2 = '%s' status=%d\n", name_buf, status);
        failures++;
    }
    opensps_get_grad_param_name(OPENSPS_BGRAD_IMF1, name_buf, 64, &status);
    if (status != 0 || strcmp(name_buf, "imf1") != 0) {
        printf("FAIL: name for BGRAD_IMF1 = '%s' status=%d\n", name_buf, status);
        failures++;
    }

    /* Test all 33 parameter names */
    for (int idx = 0; idx < OPENSPS_NBGRAD; idx++) {
        name_buf[0] = '\0';
        opensps_get_grad_param_name(idx, name_buf, 64, &status);
        if (status != 0 || strlen(name_buf) == 0) {
            printf("FAIL: opensps_get_grad_param_name for idx=%d (status=%d, name='%s')\n", idx, status, name_buf);
            failures++;
        }
    }

    /* Error handling: out of range indices */
    opensps_get_grad_param_name(-1, name_buf, 64, &status);
    if (status != 1) { printf("FAIL: expected status 1 for idx=-1\n"); failures++; }
    opensps_get_grad_param_name(OPENSPS_NBGRAD, name_buf, 64, &status);
    if (status != 1) { printf("FAIL: expected status 1 for idx=%d\n", OPENSPS_NBGRAD); failures++; }

    /* 2. Setup context */
    opensps_context_create(&h, &status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("FAIL: setup\n"); return 1; }

    opensps_context_set_int(h, "add_dust_emission", 0, &status);
    opensps_context_set_int(h, "add_neb_emission", 0, &status);
    opensps_context_set_int(h, "add_agn_dust", 1, &status);
    opensps_context_set_float(h, "tage", 10.0, &status);

    opensps_context_get_nbands(h, &nbands, &status);
    opensps_context_get_nspec(h, &nspec, &status);
    opensps_context_get_ntfull(h, &ntfull, &status);

    /* 3. Batch gradient test against finite differences */
    const char *env_rtol = getenv("FSPS_TEST_RTOL");
    /* Finite-difference validation tolerances. In single precision (-Dkernel_precision=single,
       where FSPS_TEST_RTOL >= 1e-4), roundoff in central differences is ~1e-3 to 1e-2
       (matching tests/test_csp_batch_grad.f90 DUST_RTOL32 = 2e-2). */
    double tol_dust = 5e-3;
    double tol_fagn = 1e-3;
    if (env_rtol != NULL && atof(env_rtol) >= 1.0e-4) {
        tol_dust = 2.0e-2;
        tol_fagn = 2.0e-3;
    }

    const char *names[2] = {"dust2", "fagn"};
    double dust2_base[N_MOD] = {0.2, 0.5};
    double fagn_base[N_MOD] = {0.05, 0.1};
    double dust2_val[N_MOD], fagn_val[N_MOD];
    for (int m = 0; m < N_MOD; m++) {
        dust2_val[m] = dust2_base[m];
        fagn_val[m] = fagn_base[m];
    }
    const double *values[2] = {dust2_val, fagn_val};

    double *mags = (double *)malloc(sizeof(double) * nbands * N_MOD);
    double *mags_bar = (double *)malloc(sizeof(double) * nbands * N_MOD);
    double *params_bar = (double *)malloc(sizeof(double) * OPENSPS_NBGRAD * N_MOD);
    memset(params_bar, 0, sizeof(double) * OPENSPS_NBGRAD * N_MOD);

    /* Loss = sum of all mags */
    for (int i = 0; i < nbands * N_MOD; i++) mags_bar[i] = 1.0;

    opensps_context_compute_csp_batch_grad(
        h, N_MOD, 2, names, values, NULL, NULL, NULL,
        mags_bar, NULL, NULL, NULL,
        mags, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        params_bar, NULL, NULL, &status
    );

    if (status != 0) {
        char err_msg[256];
        int err_st = 0;
        opensps_get_last_error(&err_st, err_msg, 256);
        printf("FAIL: compute_csp_batch_grad status=%d msg='%s'\n", status, err_msg);
        failures++;
    }

    /* Finite differences check for dust2 and fagn */
    double h_step = 1e-5;
    for (int m = 0; m < N_MOD; m++) {
        /* Check dust2 gradient */
        dust2_val[m] = dust2_base[m] + h_step;
        double *mags_plus = (double *)malloc(sizeof(double) * nbands * N_MOD);
        opensps_context_compute_csp_batch(h, N_MOD, 2, names, values, NULL,
                                          mags_plus, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        dust2_val[m] = dust2_base[m] - h_step;
        double *mags_minus = (double *)malloc(sizeof(double) * nbands * N_MOD);
        opensps_context_compute_csp_batch(h, N_MOD, 2, names, values, NULL,
                                          mags_minus, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        dust2_val[m] = dust2_base[m];

        double loss_plus = 0.0, loss_minus = 0.0;
        for (int b = 0; b < nbands; b++) {
            loss_plus += mags_plus[m * nbands + b];
            loss_minus += mags_minus[m * nbands + b];
        }
        double fd_grad = (loss_plus - loss_minus) / (2.0 * h_step);
        /* params_bar is column-major: row * N_MOD + m, row = OPENSPS_BGRAD_DUST2 */
        double adj_grad = params_bar[OPENSPS_BGRAD_DUST2 + m * OPENSPS_NBGRAD];
        double rel_err = fabs(adj_grad - fd_grad) / fmax(fabs(fd_grad), 1e-4);
        printf("[dust2 m=%d] adjoint=%e fd=%e rel_err=%e\n", m, adj_grad, fd_grad, rel_err);
        if (rel_err > tol_dust) {
            printf("FAIL: dust2 gradient mismatch m=%d (rel_err=%e, tol=%e)\n", m, rel_err, tol_dust);
            failures++;
        }
        free(mags_plus);
        free(mags_minus);

        /* Check fagn gradient */
        fagn_val[m] = fagn_base[m] + h_step;
        mags_plus = (double *)malloc(sizeof(double) * nbands * N_MOD);
        opensps_context_compute_csp_batch(h, N_MOD, 2, names, values, NULL,
                                          mags_plus, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        fagn_val[m] = fagn_base[m] - h_step;
        mags_minus = (double *)malloc(sizeof(double) * nbands * N_MOD);
        opensps_context_compute_csp_batch(h, N_MOD, 2, names, values, NULL,
                                          mags_minus, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        fagn_val[m] = fagn_base[m];

        loss_plus = 0.0; loss_minus = 0.0;
        for (int b = 0; b < nbands; b++) {
            loss_plus += mags_plus[m * nbands + b];
            loss_minus += mags_minus[m * nbands + b];
        }
        fd_grad = (loss_plus - loss_minus) / (2.0 * h_step);
        adj_grad = params_bar[OPENSPS_BGRAD_FAGN + m * OPENSPS_NBGRAD];
        rel_err = fabs(adj_grad - fd_grad) / fmax(fabs(fd_grad), 1e-4);
        printf("[fagn m=%d] adjoint=%e fd=%e rel_err=%e\n", m, adj_grad, fd_grad, rel_err);
        if (rel_err > tol_fagn) {
            printf("FAIL: fagn gradient mismatch m=%d (rel_err=%e, tol=%e)\n", m, rel_err, tol_fagn);
            failures++;
        }
        free(mags_plus);
        free(mags_minus);
    }

    /* 4. Test free IMF parameters with SSP operator */
    opensps_context_set_int(h, "imf_type", 2, &status);
    opensps_context_build_ssp_ops(h, 0, &status);
    if (status != 0) {
        printf("FAIL: build_ssp_ops status=%d\n", status);
        failures++;
    }

    const char *imf_names[3] = {"dust2", "imf1", "imf2"};
    double imf1_val[N_MOD] = {1.2, 1.4};
    double imf2_val[N_MOD] = {2.2, 2.4};
    const double *imf_values[3] = {dust2_val, imf1_val, imf2_val};

    memset(params_bar, 0, sizeof(double) * OPENSPS_NBGRAD * N_MOD);
    opensps_context_compute_csp_batch_grad(
        h, N_MOD, 3, imf_names, imf_values, NULL, NULL, NULL,
        mags_bar, NULL, NULL, NULL,
        mags, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        params_bar, NULL, NULL, &status
    );
    if (status != 0) {
        printf("FAIL: compute_csp_batch_grad with ssp_ops status=%d\n", status);
        failures++;
    } else {
        for (int m = 0; m < N_MOD; m++) {
            double g_imf1 = params_bar[OPENSPS_BGRAD_IMF1 + m * OPENSPS_NBGRAD];
            double g_imf2 = params_bar[OPENSPS_BGRAD_IMF2 + m * OPENSPS_NBGRAD];
            printf("[imf ssp_ops m=%d] g_imf1=%e g_imf2=%e\n", m, g_imf1, g_imf2);
            if (isnan(g_imf1) || isinf(g_imf1) || g_imf1 == 0.0) {
                printf("FAIL: g_imf1 is invalid\n");
                failures++;
            }
            if (isnan(g_imf2) || isinf(g_imf2) || g_imf2 == 0.0) {
                printf("FAIL: g_imf2 is invalid\n");
                failures++;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Test 4: Struct-based entry point opensps_context_compute_csp_batch_ext
    // ------------------------------------------------------------------------
    {
        opensps_batch_options_t opts = {0};
        opts.n = N_MOD;
        opts.n_params = 3;
        opts.names = imf_names;
        opts.values = imf_values;
        opts.mags_bar = mags_bar;

        opensps_batch_results_t res = {0};
        res.mags = mags;
        res.params_bar = params_bar;

        status = 0;
        opensps_context_compute_csp_batch_ext(h, &opts, &res, &status);
        if (status != 0) {
            printf("FAIL: opensps_context_compute_csp_batch_ext failed with status %d\n", status);
            failures++;
        } else {
            double g_imf1 = res.params_bar[OPENSPS_BGRAD_IMF1];
            if (isnan(g_imf1) || isinf(g_imf1) || g_imf1 == 0.0) {
                printf("FAIL: opensps_context_compute_csp_batch_ext returned invalid gradient\n");
                failures++;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Test 5: Enrichment history weights (ssp_weights_z and weights_z_bar)
    // ------------------------------------------------------------------------
    {
        /* Free ssp_ops because ssp_weights_z requires standard grid evaluation */
        opensps_context_free_ssp_ops(h, &status);

        int nz = 0, ntfull = 0;
        opensps_context_get_nz(h, &nz, &status);
        opensps_context_get_ntfull(h, &ntfull, &status);
        int wz_len = ntfull * nz * N_MOD;
        double *wz = (double *)calloc(wz_len, sizeof(double));
        double *wz_bar = (double *)calloc(wz_len, sizeof(double));
        // Put some weights on the first age and first metallicity
        for (int m = 0; m < N_MOD; m++) {
            wz[m * (ntfull * nz)] = 1.0;
        }

        opensps_batch_options_t opts = {0};
        opts.n = N_MOD;
        opts.n_params = 2;
        opts.names = names;
        opts.values = values;
        opts.ssp_weights_z = wz;
        opts.mags_bar = mags_bar;

        opensps_batch_results_t res = {0};
        res.mags = mags;
        res.weights_z_bar = wz_bar;

        status = 0;
        opensps_context_compute_csp_batch_ext(h, &opts, &res, &status);
        if (status != 0) {
            printf("FAIL: ssp_weights_z batch failed with status %d\n", status);
            failures++;
        }

        free(wz);
        free(wz_bar);
    }

    free(mags);
    free(mags_bar);
    free(params_bar);
    opensps_context_destroy(h, &status);

    if (failures == 0) {
        printf("test_c_batch_grad: ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("test_c_batch_grad: %d FAILURES\n", failures);
        return 1;
    }
}
