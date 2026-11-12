/* The [alpha/Fe] axis through the C API, what needs no alpha-enhanced library: a batch that names `afe`
 * without the bases of opensps_context_build_afe_basis returns status 2, and the builder rejects invalid
 * arguments (fewer than two values, values not increasing, a bad handle). The axis on the real libraries is
 * tests/test_afe_library.f90. Requires a data root, skipped without one. */
#include "opensps.h"
#include "test_c_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_MOD 3

int main(void) {
    skip_without_data();
    int status = 0, h = 0, nbands = 0, failures = 0;
    const char *names[1] = {"afe"};
    double afe_m[N_MOD] = {0.0, 0.1, 0.3};
    const double *values[1] = {afe_m};
    double bad[3] = {0.4, 0.0, 0.2};
    double one[1] = {0.0};

    opensps_context_create(&h, &status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed\n"); return 1; }
    opensps_context_get_nbands(h, &nbands, &status);
    /* use_lw_tpagb (O-rich TP-AGB spectra from the Lancon & Mouhcine library or from the main one) is a setting */
    opensps_context_set_int(h, "use_lw_tpagb", 0, &status);
    if (status != 0) { printf("FAIL: set use_lw_tpagb: status %d\n", status); failures++; }
    opensps_context_set_int(h, "use_lw_tpagb", 1, &status);

    double *mags = (double *)malloc(sizeof(double) * nbands * N_MOD);
    opensps_context_compute_csp_batch(h, N_MOD, 1, names, values, NULL, mags, NULL, NULL, NULL, NULL, NULL, NULL,
                                      NULL, &status);
    if (status != 2) { printf("FAIL: afe without the bases: status %d, expected 2\n", status); failures++; }

    opensps_context_build_afe_basis(h, "mist", "miles", 1, one, &status);
    if (status == 0) { printf("FAIL: one [alpha/Fe] value accepted\n"); failures++; }
    opensps_context_build_afe_basis(h, "mist", "miles", 3, bad, &status);
    if (status == 0) { printf("FAIL: values not increasing accepted\n"); failures++; }
    opensps_context_build_afe_basis(9999, "mist", "miles", 3, bad, &status);
    if (status != 1) { printf("FAIL: invalid handle: status %d, expected 1\n", status); failures++; }

    free(mags);
    opensps_context_destroy(h, &status);

    /* On the real libraries (OPENSPS_AFE_DATA: a data root with mist_afe and c3k_lr) or the variants derived from
     * the test library (OPENSPS_AFE_ISOC, OPENSPS_AFE_SPEC: tests/run_afe_derived_test.py): the stack through the C
     * API, a batch with logzsol and afe free per model, and d L / d afe against a central difference */
    const char *afe_root = getenv("OPENSPS_AFE_DATA");
    if (afe_root != NULL && afe_root[0] != '\0') {
        const double grid[5] = {-0.2, 0.0, 0.2, 0.4, 0.6};
        const char *nm[2] = {"logzsol", "afe"};
        double lz[N_MOD] = {-0.6, -0.2, 0.1}, af[N_MOD] = {0.07, 0.31, 0.52}, afp[N_MOD], afm[N_MOD];
        const double *v[2] = {lz, af}, *vp[2] = {lz, afp}, *vm[2] = {lz, afm};
        int h2 = 0, nb2 = 0, nt2 = 0;
        const char *isoc = getenv("OPENSPS_AFE_ISOC"), *spec = getenv("OPENSPS_AFE_SPEC");
        char sel_isoc[96], sel_spec[96];
        if (isoc == NULL || isoc[0] == '\0') isoc = "mist_afe";
        if (spec == NULL || spec[0] == '\0') spec = "c3k_lr";
        snprintf(sel_isoc, sizeof sel_isoc, "%s:afe=0.0", isoc);
        snprintf(sel_spec, sizeof sel_spec, "%s:afe=0.0", spec);
        /* (the magnitudes are linear in [alpha/Fe] inside an interval of the grid: a step of 1e-2 within it is exact
         * and resolved in FP32 builds too) */
        const double dh = 1.0e-2;
        setenv("OPENSPS_DATA_HOME", afe_root, 1);
        setenv("FSPS_DATA_HOME", afe_root, 1);
        opensps_context_create(&h2, &status);
        opensps_context_setup(-1, sel_isoc, sel_spec, "DL07", h2, &status);
        if (status != 0) { printf("FAIL: setup of %s + %s: %d\n", isoc, spec, status); return 1; }
        opensps_context_set_int(h2, "imf_type", 1, &status);
        opensps_context_set_int(h2, "sfh", 1, &status);
        opensps_context_set_float(h2, "tau", 1.5, &status);
        opensps_context_set_float(h2, "tage", 7.0, &status);
        opensps_context_build_afe_basis(h2, isoc, spec, 5, grid, &status);
        if (status != 0) { printf("FAIL: build_afe_basis: %d\n", status); return 1; }
        opensps_context_get_nbands(h2, &nb2, &status);
        opensps_context_get_ntfull(h2, &nt2, &status);
        double *m0 = (double *)malloc(sizeof(double) * nb2 * N_MOD), *m1 = (double *)malloc(sizeof(double) * nb2 * N_MOD);
        double *m2 = (double *)malloc(sizeof(double) * nb2 * N_MOD), *mbar = (double *)malloc(sizeof(double) * nb2 * N_MOD);
        double *pbar = (double *)malloc(sizeof(double) * OPENSPS_NBGRAD * N_MOD);
        for (int i = 0; i < nb2 * N_MOD; i++) mbar[i] = 1.0;
        for (int m = 0; m < N_MOD; m++) { afp[m] = af[m] + dh; afm[m] = af[m] - dh; }
        opensps_clear_error();
        opensps_context_compute_csp_batch(h2, N_MOD, 2, nm, v, NULL, m0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        if (status != 0) {
            char msg[256] = "";
            int st2 = 0;
            opensps_get_last_error(&st2, msg, 256);
            printf("FAIL: batch: %d (%s)\n", status, msg);
            return 1;
        }
        opensps_context_compute_csp_batch_grad(h2, N_MOD, 2, nm, v, NULL, NULL, NULL, mbar, NULL, NULL, NULL, m0, NULL, NULL,
                                               NULL, NULL, NULL, NULL, NULL, pbar, NULL, NULL, &status);
        if (status != 0) {
            char msg[256] = "";
            int st2 = 0;
            opensps_get_last_error(&st2, msg, 256);
            printf("FAIL: batch with gradients: %d (%s)\n", status, msg);
            return 1;
        }
        opensps_context_compute_csp_batch(h2, N_MOD, 2, nm, vp, NULL, m1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        opensps_context_compute_csp_batch(h2, N_MOD, 2, nm, vm, NULL, m2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &status);
        for (int m = 0; m < N_MOD; m++) {
            double lp = 0.0, lm = 0.0;
            for (int b = 0; b < nb2; b++) { lp += m1[m * nb2 + b]; lm += m2[m * nb2 + b]; }
            double fd = (lp - lm) / (2.0 * dh), ad = pbar[m * OPENSPS_NBGRAD + OPENSPS_BGRAD_AFE];
            printf("# model %d: d L / d afe = %.6e, finite difference %.6e\n", m, ad, fd);
            if (!(fabs(ad - fd) <= 2.0e-3 * fabs(fd) + 1.0e-8)) { printf("FAIL: d L / d afe\n"); failures++; }
        }
        free(m0); free(m1); free(m2); free(mbar); free(pbar);
        opensps_context_destroy(h2, &status);
    }
    printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
