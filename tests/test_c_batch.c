/* The C batch entry point (fsps_context_compute_csp_batch) with the fitting parameters of a
 * real pipeline: logzsol, dust2, dust1 (= dust_ratio * dust2, formed by the caller), dust_index,
 * fagn, agn_tau, gas_logz, gas_logu, duste_umin, duste_gamma, duste_qpah, zred, and a
 * non-parametric SFH as an SSP-weight matrix (logmass is a caller-side scaling). The arrays
 * (one per parameter, length n) are checked against the same models set one at a time on the
 * context, and the error statuses for a parameter that changes the SSPs and an unknown one.
 * Requires the FSPS data tree (SPS_HOME), like the regression tests. */
#include "fsps.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_MOD 6
#define N_PAR 12

int main(void) {
    int status = 0, h = 0, nspec = 0, ntfull = 0, nbands = 0, failures = 0;
    const char *names[N_PAR] = {"logzsol", "dust2", "dust1", "dust_index", "fagn", "agn_tau",
                                "gas_logz", "gas_logu", "duste_umin", "duste_gamma", "duste_qpah", "zred"};
    double logz[N_MOD] = {-1.5, -1.0, -0.5, 0.0, 0.15, -0.8};
    double dust2[N_MOD] = {0.05, 0.3, 0.6, 1.0, 0.1, 0.4};
    double dust_ratio[N_MOD] = {0.5, 1.0, 1.5, 0.8, 2.0, 1.2};
    double dust1[N_MOD];
    double dindex[N_MOD] = {-0.7, -0.3, -1.0, 0.0, -0.5, 0.2};
    double fagn[N_MOD] = {0.0, 0.05, 0.1, 0.3, 0.01, 0.2};
    double agn_tau[N_MOD] = {5.0, 10.0, 20.0, 40.0, 80.0, 15.0};
    double gas_logz[N_MOD] = {-1.5, -1.0, -0.5, 0.0, 0.15, -0.8};
    double gas_logu[N_MOD] = {-3.5, -3.0, -2.5, -2.0, -1.5, -2.2};
    double umin[N_MOD] = {1.0, 2.0, 5.0, 10.0, 0.5, 20.0};
    double gamma_[N_MOD] = {0.01, 0.05, 0.1, 0.3, 0.0, 0.5};
    double qpah[N_MOD] = {0.5, 1.5, 2.5, 3.5, 4.5, 2.0};
    double zred[N_MOD] = {0.0, 0.1, 0.5, 1.0, 2.0, 0.3};
    const double *values[N_PAR] = {logz, dust2, dust1, dindex, fagn, agn_tau, gas_logz, gas_logu,
                                   umin, gamma_, qpah, zred};
    for (int m = 0; m < N_MOD; m++) dust1[m] = dust_ratio[m] * dust2[m];

    fsps_context_create(&h, &status);
    fsps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed\n"); return 1; }
    fsps_context_set_int(h, "add_neb_emission", 1, &status);
    fsps_context_set_int(h, "add_dust_emission", 1, &status);
    fsps_context_set_int(h, "add_igm_absorption", 1, &status);
    fsps_context_set_int(h, "add_agn_dust", 1, &status);
    fsps_context_get_nspec(h, &nspec, &status);
    fsps_context_get_ntfull(h, &ntfull, &status);
    fsps_context_get_nbands(h, &nbands, &status);

    /* Non-parametric SFHs: per model a smooth bump over the SSP ages, normalized to one solar
     * mass formed (column-major, ntfull x n) */
    double *w = malloc(sizeof(double) * ntfull * N_MOD);
    for (int m = 0; m < N_MOD; m++) {
        double c = 0.3 * ntfull + 0.1 * ntfull * m, sum = 0.0;
        for (int k = 0; k < ntfull; k++) {
            double x = (k - c) / (0.15 * ntfull);
            w[(size_t)m * ntfull + k] = exp(-0.5 * x * x) + 1e-3;
            sum += w[(size_t)m * ntfull + k];
        }
        for (int k = 0; k < ntfull; k++) w[(size_t)m * ntfull + k] /= sum;
    }

    double *mags = malloc(sizeof(double) * nbands * N_MOD), *spec = malloc(sizeof(double) * nspec * N_MOD);
    double *mags1 = malloc(sizeof(double) * nbands), *spec1 = malloc(sizeof(double) * nspec);
    double mass[N_MOD], lbol[N_MOD], mass1, lbol1;

    fsps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, w, mags, spec, mass, lbol,
                                   NULL, NULL, NULL, NULL, &status);
    printf("batch of %d models: status %d\n", N_MOD, status);
    if (status != 0) {
        char msg[256];
        int st2 = 0;
        fsps_get_last_error(&st2, msg, 256);
        printf("error: %s\n", msg);
        return 1;
    }

    /* Where the batch left its spectra (device address in GPU builds): shape and precision */
    {
        void *dptr = NULL;
        int dn = 0, dnspec = 0, dbytes = 0, dst = 0;
        fsps_context_batch_spec_device(h, &dptr, &dnspec, &dn, &dbytes, &dst);
        printf("batch spectra: %d x %d, %d bytes per value, status %d\n", dnspec, dn, dbytes, dst);
        if (dst != 0 || dptr == NULL || dnspec != nspec || dn != N_MOD || (dbytes != 4 && dbytes != 8)) {
            printf("FAIL: fsps_context_batch_spec_device\n");
            failures++;
        }
        fsps_context_batch_spec_device(-7, &dptr, &dnspec, &dn, &dbytes, &dst);
        if (dst != 1) {
            printf("FAIL: fsps_context_batch_spec_device, bad handle gave status %d (expected 1)\n", dst);
            failures++;
        }

        /* The same spectra on the host at kernel precision: equal to the double ones */
        void *hptr = NULL;
        fsps_context_batch_spec_host(h, &hptr, &dnspec, &dn, &dbytes, &dst);
        int same = dst == 0 && hptr != NULL && dnspec == nspec && dn == N_MOD;
        for (size_t i = 0; same && i < (size_t)nspec * N_MOD; i++) {
            double v = dbytes == 4 ? (double)((const float *)hptr)[i] : ((const double *)hptr)[i];
            same = v == spec[i];
        }
        if (!same) {
            printf("FAIL: fsps_context_batch_spec_host\n");
            failures++;
        }
    }

    double d_mag = 0.0, d_spec = 0.0, d_scal = 0.0;
    for (int m = 0; m < N_MOD; m++) {
        /* The same model, its parameters set on the context (logzsol through the arrays, so the
         * basis is the per-metallicity one in both cases; the SFH as that model's weights) */
        for (int i = 1; i < N_PAR; i++) fsps_context_set_float(h, names[i], values[i][m], &status);
        const char *zname[1] = {"logzsol"};
        const double *zval[1] = {&logz[m]};
        fsps_context_compute_csp_batch(h, 1, 1, zname, zval, w + (size_t)m * ntfull, mags1, spec1, &mass1,
                                       &lbol1, NULL, NULL, NULL, NULL, &status);
        if (status != 0) { printf("single model %d: status %d\n", m, status); return 1; }
        double peak = 0.0;
        for (int i = 0; i < nspec; i++) peak = fmax(peak, spec1[i]);
        for (int i = 0; i < nspec; i++)
            if (spec1[i] > 1e-6 * peak)
                d_spec = fmax(d_spec, fabs(spec[(size_t)m * nspec + i] / spec1[i] - 1.0));
        double mmin = 1e30;
        for (int b = 0; b < nbands; b++)
            if (mags1[b] == mags1[b]) mmin = fmin(mmin, mags1[b]);
        for (int b = 0; b < nbands; b++)
            if (mags1[b] == mags1[b] && mags1[b] < mmin + 30.0)
                d_mag = fmax(d_mag, fabs(mags[(size_t)m * nbands + b] - mags1[b]));
        d_scal = fmax(d_scal, fmax(fabs(mass[m] / mass1 - 1.0), fabs(lbol[m] - lbol1)));
    }
    printf("max difference, arrays vs one at a time (spectrum, mag, mass/lbol): %.2e %.2e %.2e\n",
           d_spec, d_mag, d_scal);
    if (!(d_spec <= 1e-5 && d_mag <= 1e-5 && d_scal <= 1e-12)) { printf("FAIL: parameter arrays\n"); failures++; }

    /* A parameter that changes the SSPs cannot vary per model; unknown names are rejected */
    const char *bad1[1] = {"imf1"};
    const char *bad2[1] = {"not_a_parameter"};
    const double *bv[1] = {dust2};
    fsps_context_compute_csp_batch(h, N_MOD, 1, bad1, bv, NULL, mags, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                   &status);
    if (status != 2) { printf("FAIL: imf1 per model gave status %d (expected 2)\n", status); failures++; }
    fsps_context_compute_csp_batch(h, N_MOD, 1, bad2, bv, NULL, mags, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                   &status);
    if (status != 3) { printf("FAIL: unknown parameter gave status %d (expected 3)\n", status); failures++; }

    fsps_context_destroy(h, &status);
    free(w); free(mags); free(spec); free(mags1); free(spec1);
    printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
