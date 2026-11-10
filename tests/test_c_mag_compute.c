/* opensps_context_set_mag_compute: the bands a context computes (and an integer parameter, sfh, per model in a batch). A batch with a mask returns, for the
 * bands in the mask, the same magnitudes as the batch without it (bit for bit: the band integrals do not
 * depend on the other bands), NaN for the others; a NULL mask restores all bands; a bad handle and an
 * out-of-range count are rejected. Also with the magnitudes at per-model redshifts (the fitting
 * configuration: the redshift is a batch parameter). Requires a data root, skipped without one. */
#include "opensps.h"
#include "test_c_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_MOD 48   /* a batch of a size on which the per-model IGM and the other batched kernels run on many threads */
#define N_PAR 3

int main(void) {
    skip_without_data();
    int status = 0, h = 0, ntfull = 0, nbands = 0, failures = 0;
    const char *names[N_PAR] = {"dust2", "logzsol", "zred"};
    double dust2[N_MOD], logz[N_MOD], zred[N_MOD];
    for (int m = 0; m < N_MOD; m++) {
        dust2[m] = 0.05 + 0.02 * (m % 37);
        logz[m] = -1.5 + 0.03 * (m % 53);
        zred[m] = 0.1 + 0.05 * (m % 47);
    }
    const double *values[N_PAR] = {dust2, logz, zred};

    opensps_context_create(&h, &status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed\n"); return 1; }
    opensps_context_set_int(h, "add_igm_absorption", 1, &status);
    opensps_context_get_ntfull(h, &ntfull, &status);
    opensps_context_get_nbands(h, &nbands, &status);

    double *w = malloc(sizeof(double) * ntfull * N_MOD);
    for (int m = 0; m < N_MOD; m++) {
        double sum = 0.0;
        for (int k = 0; k < ntfull; k++) {
            double x = (k - (0.3 + 0.1 * m) * ntfull) / (0.15 * ntfull);
            w[(size_t)m * ntfull + k] = exp(-0.5 * x * x) + 1e-3;
            sum += w[(size_t)m * ntfull + k];
        }
        for (int k = 0; k < ntfull; k++) w[(size_t)m * ntfull + k] /= sum;
    }
    double *all = malloc(sizeof(double) * nbands * N_MOD), *sel = malloc(sizeof(double) * nbands * N_MOD);
    int *mask = calloc(nbands, sizeof(int));
    double mass[N_MOD], lbol[N_MOD];

    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, w, all, NULL, mass, lbol,
                                   NULL, NULL, NULL, NULL, &status);
    if (status != 0) { printf("batch: status %d\n", status); return 1; }

    /* A few bands, one of them the last */
    int chosen[4] = {0, 3, nbands / 2, nbands - 1};
    for (int i = 0; i < 4; i++) mask[chosen[i]] = 1;
    opensps_context_set_mag_compute(h, nbands, mask, &status);
    if (status != 0) { printf("FAIL: set_mag_compute status %d\n", status); failures++; }
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, w, sel, NULL, mass, lbol,
                                   NULL, NULL, NULL, NULL, &status);
    if (status != 0) { printf("batch with a mask: status %d\n", status); return 1; }
    int ok = 1;
    for (int m = 0; m < N_MOD; m++)
        for (int b = 0; b < nbands; b++) {
            double v = sel[(size_t)m * nbands + b], a = all[(size_t)m * nbands + b];
            if (mask[b]) { if (!(v == a || (v != v && a != a))) ok = 0; }
            else if (v == v) ok = 0;
        }
    if (!ok) { printf("FAIL: masked bands are not the same as unmasked, or the others are not NaN\n"); failures++; }

    /* A shorter mask: the bands beyond it are not computed */
    int shortmask[2] = {1, 1};
    opensps_context_set_mag_compute(h, 2, shortmask, &status);
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, w, sel, NULL, mass, lbol,
                                   NULL, NULL, NULL, NULL, &status);
    ok = status == 0;
    for (int m = 0; ok && m < N_MOD; m++)
        for (int b = 0; b < nbands; b++) {
            double v = sel[(size_t)m * nbands + b], a = all[(size_t)m * nbands + b];
            if (b < 2) { if (!(v == a || (v != v && a != a))) ok = 0; } else if (v == v) ok = 0;
        }
    if (!ok) { printf("FAIL: a mask shorter than the bands\n"); failures++; }

    /* NULL restores every band */
    opensps_context_set_mag_compute(h, 0, NULL, &status);
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, w, sel, NULL, mass, lbol,
                                   NULL, NULL, NULL, NULL, &status);
    ok = status == 0 && memcmp(sel, all, sizeof(double) * nbands * N_MOD) == 0;
    if (!ok) { printf("FAIL: a NULL mask does not restore all bands\n"); failures++; }

    opensps_context_set_mag_compute(-3, nbands, mask, &status);
    if (status != 1) { printf("FAIL: bad handle gave status %d (expected 1)\n", status); failures++; }
    opensps_context_set_mag_compute(h, nbands + 1, mask, &status);
    if (status != 3) { printf("FAIL: n_bands too large gave status %d (expected 3)\n", status); failures++; }
    opensps_context_set_mag_compute(h, 0, mask, &status);
    if (status != 3) { printf("FAIL: n_bands = 0 gave status %d (expected 3)\n", status); failures++; }

    /* sfh is an integer parameter: in a batch it goes through the setter per model, not the word-per-parameter
     * path of the real scalars; the same models with sfh set on the context must agree bit for bit */
    {
        double sfh[N_MOD], tage[N_MOD];
        for (int m = 0; m < N_MOD; m++) { sfh[m] = 1.0; tage[m] = 1.0 + 0.2 * m; }
        const char *n3[3] = {"sfh", "tage", "dust2"};
        const double *v3[3] = {sfh, tage, dust2};
        const char *n2[2] = {"tage", "dust2"};
        const double *v2[2] = {tage, dust2};
        double *a = malloc(sizeof(double) * nbands * N_MOD), *b = malloc(sizeof(double) * nbands * N_MOD);
        opensps_context_compute_csp_batch(h, N_MOD, 3, n3, v3, NULL, a, NULL, mass, lbol, NULL, NULL, NULL, NULL, &status);
        if (status != 0) { printf("FAIL: batch with sfh as a parameter: status %d\n", status); failures++; }
        opensps_context_set_int(h, "sfh", 1, &status);
        opensps_context_compute_csp_batch(h, N_MOD, 2, n2, v2, NULL, b, NULL, mass, lbol, NULL, NULL, NULL, NULL, &status);
        if (status != 0 || memcmp(a, b, sizeof(double) * nbands * N_MOD) != 0) {
            printf("FAIL: sfh per model differs from sfh set on the context\n");
            failures++;
        }
        free(a); free(b);
    }

    free(w); free(all); free(sel); free(mask);
    opensps_context_destroy(h, &status);
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
