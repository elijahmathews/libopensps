/* The batch's intrinsic young and old components (fsps_context_compute_csp_batch_parts):
 *  - the full attenuated spectrum is rebuilt from the parts with the documented dust model
 *    (power-law curve, dust_type 0; birth-cloud dust1 with dust1_index, diffuse dust2, dust3 on
 *    the old part, frac_obrun, frac_nodust; nebular emission on), per model, per pixel;
 *  - the masses and luminosities of the parts add up to the totals;
 *  - the other outputs equal those of fsps_context_compute_csp_batch;
 *  - young / old are required (status 3).
 * Tolerances: 1e-12 with FP64 kernels; with FP32 kernels the parts are rounded to single
 * precision before the rebuild, so 1e-6. Requires the FSPS data tree (SPS_HOME). */
#include "fsps.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_MOD 6
#define N_PAR 10

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

int main(void) {
    int status = 0, h = 0, nspec = 0, nbands = 0;
    const char *names[N_PAR] = {"logzsol", "tau", "tage", "dust1", "dust2", "dust_index", "dust1_index",
                                "frac_obrun", "frac_nodust", "dust3"};
    double logz[N_MOD] = {-1.2, -0.6, -0.2, 0.0, 0.1, -0.9};
    double tau[N_MOD] = {0.5, 1.0, 3.0, 8.0, 0.3, 2.0};
    double tage[N_MOD] = {0.05, 1.0, 4.0, 10.0, 0.5, 2.5};
    double dust1[N_MOD] = {0.0, 0.3, 1.0, 0.6, 2.0, 0.4};
    double dust2[N_MOD] = {0.0, 0.2, 0.5, 0.3, 1.0, 0.1};
    double dindex[N_MOD] = {-0.7, -0.3, -1.0, 0.0, -0.5, 0.2};
    double d1index[N_MOD] = {-1.0, -1.0, -0.7, -1.3, -1.0, -0.5};
    double fob[N_MOD] = {0.0, 0.0, 0.1, 0.0, 0.3, 0.05};
    double fnd[N_MOD] = {0.0, 0.1, 0.0, 0.0, 0.2, 0.0};
    double dust3[N_MOD] = {0.0, 0.0, 0.0, 0.2, 0.0, 0.1};
    const double *values[N_PAR] = {logz, tau, tage, dust1, dust2, dindex, d1index, fob, fnd, dust3};

    fsps_context_create(&h, &status);
    fsps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed\n"); return 1; }
    fsps_context_set_int(h, "sfh", 4, &status);
    fsps_context_set_int(h, "dust_type", 0, &status);
    fsps_context_set_int(h, "add_dust_emission", 0, &status);
    fsps_context_set_int(h, "add_neb_emission", 1, &status);
    fsps_context_get_nspec(h, &nspec, &status);
    fsps_context_get_nbands(h, &nbands, &status);
    double *lam = malloc(sizeof(double) * nspec);
    fsps_context_get_lambda(h, lam, &status);

    size_t ns = (size_t)nspec * N_MOD;
    double *spec = malloc(sizeof(double) * ns), *young = malloc(sizeof(double) * ns);
    double *old = malloc(sizeof(double) * ns), *spec_ref = malloc(sizeof(double) * ns);
    double *mags = malloc(sizeof(double) * nbands * N_MOD), *mags_ref = malloc(sizeof(double) * nbands * N_MOD);
    double mass[N_MOD], lbol[N_MOD], my[N_MOD], mo[N_MOD], ly[N_MOD], lo[N_MOD], mass_ref[N_MOD], lbol_ref[N_MOD];

    fsps_context_compute_csp_batch_parts(h, N_MOD, N_PAR, names, values, NULL, mags, spec, mass, lbol, NULL, NULL,
                                         NULL, NULL, young, old, my, mo, ly, lo, &status);
    check(status == 0, "batch with parts: status 0");
    if (status != 0) return 1;

    /* Kernel precision: the bytes per value of the batch spectra */
    void *p = NULL;
    int bn = 0, bnspec = 0, bytes = 0, st2 = 0;
    fsps_context_batch_spec_host(h, &p, &bnspec, &bn, &bytes, &st2);
    double tol = bytes == 4 ? 1e-6 : 1e-12;
    printf("# kernel precision: %d bytes, tolerance %.0e\n", bytes, tol);

    /* The full spectrum from the parts (dust_type 0: curve = (lambda / 5500 A)^dust_index) */
    double d_rebuild = 0.0;
    for (int m = 0; m < N_MOD; m++) {
        double peak = 0.0;
        for (int i = 0; i < nspec; i++) peak = fmax(peak, spec[(size_t)m * nspec + i]);
        for (int i = 0; i < nspec; i++) {
            double x = lam[i] / 5500.0, curve = pow(x, dindex[m]);
            double tb = dust1[m] > 0.0 ? exp(-dust1[m] * pow(x, d1index[m])) : 1.0;
            double to = dust3[m] != 0.0 ? exp(-dust3[m] * curve) : 1.0;
            double td = exp(-dust2[m] * curve);
            size_t k = (size_t)m * nspec + i;
            double rebuilt = (young[k] * (tb * (1.0 - fob[m]) + fob[m]) + old[k] * to) * (td * (1.0 - fnd[m]) + fnd[m]);
            if (spec[k] > 1e-8 * peak) d_rebuild = fmax(d_rebuild, fabs(rebuilt / spec[k] - 1.0));
        }
    }
    printf("# spectrum rebuilt from the young and old parts: max relative difference %.2e\n", d_rebuild);
    check(d_rebuild <= tol, "full spectrum = dust model applied to the parts");

    /* Masses and luminosities of the parts add up */
    double d_mass = 0.0, d_lbol = 0.0;
    int both = 1;
    for (int m = 0; m < N_MOD; m++) {
        d_mass = fmax(d_mass, fabs((my[m] + mo[m]) / mass[m] - 1.0));
        d_lbol = fmax(d_lbol, fabs(log10(pow(10.0, ly[m]) + pow(10.0, lo[m])) - lbol[m]));
        both &= my[m] > 0.0 && mo[m] > 0.0 && ly[m] > -300.0 && lo[m] > -300.0;
    }
    printf("# parts vs totals: mass %.2e, L_bol %.2e dex\n", d_mass, d_lbol);
    check(d_mass <= 1e-12 && d_lbol <= 1e-12, "young + old mass and L_bol = totals");
    check(both, "young and old parts non-empty (every model is older than dust_tesc)");

    /* The other outputs equal those of the plain batch */
    fsps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, NULL, mags_ref, spec_ref, mass_ref, lbol_ref,
                                   NULL, NULL, NULL, NULL, &status);
    double d_spec = 0.0, d_mag = 0.0;
    for (size_t k = 0; k < ns; k++)
        if (spec_ref[k] > 0.0) d_spec = fmax(d_spec, fabs(spec[k] / spec_ref[k] - 1.0));
    for (int k = 0; k < nbands * N_MOD; k++)
        if (mags_ref[k] < 60.0) d_mag = fmax(d_mag, fabs(mags[k] - mags_ref[k]));
    printf("# parts entry vs plain batch: spec %.2e, mags %.2e\n", d_spec, d_mag);
    check(status == 0 && d_spec <= tol && d_mag <= 1e3 * tol, "outputs = plain batch");

    /* young and old are required */
    fsps_context_compute_csp_batch_parts(h, N_MOD, N_PAR, names, values, NULL, mags, spec, mass, lbol, NULL, NULL,
                                         NULL, NULL, NULL, old, NULL, NULL, NULL, NULL, &status);
    check(status == 3, "young NULL: status 3");

    fsps_context_destroy(h, &status);
    free(lam); free(spec); free(young); free(old); free(spec_ref); free(mags); free(mags_ref);
    printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
