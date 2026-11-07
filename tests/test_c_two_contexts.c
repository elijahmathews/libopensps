/* Contexts are independent on the device (issue #77): creating, using or destroying a second
 * context must not change what the first computes. Contexts with the same library selection
 * share one read-only setup cache, so their device mappings of the shared arrays overlap; the
 * teardown of one context must release only its own reference to them.
 *
 * Context A runs a batch, context B is created (and used, in the second round) and destroyed,
 * then A runs the same batch again: the results must be identical to the first. In a GPU build
 * the broken behaviour is a runtime abort ("data in use_device clause was not found on device" on
 * NVIDIA, "explicit extension not allowed" on AMD), which fails this test as a crash.
 * Requires a data root (OPENSPS_DATA_HOME), like the regression tests; skipped without one. */
#include "opensps.h"
#include "test_c_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_MOD 2

static int make_context(void) {
    int h = 0, s = 0;
    opensps_context_create(&h, &s);
    if (s) return -1;
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &s);
    return s ? -1 : h;
}

/* One batch on context h; mags and spec (nbands x N_MOD, nspec x N_MOD) and mass go to the caller */
static int batch(int h, double *mags, double *spec, double *mass) {
    int s = 0;
    const char *names[2] = {"tage", "dust2"};
    double tage[N_MOD] = {3.0, 9.0}, dust2[N_MOD] = {0.2, 0.8};
    const double *values[2] = {tage, dust2};
    opensps_context_compute_csp_batch(h, N_MOD, 2, names, values, NULL, mags, spec, mass, NULL, NULL, NULL,
                                      NULL, NULL, &s);
    return s;
}

/* Magnitudes of bands with no flux (> 100 mag, e.g. X-ray bands of an old population) are noise
 * around the floor and change from one identical batch to the next, even on the host; the
 * physical ones must agree to SAME_MAG_TOL. The band sums are not in a fixed order on the A100
 * (nvfortran), so identical batches differ by an ulp or so (7e-15 mag seen, with one context
 * and no second one); an unmapped or stale device array changes results by far more. */
#define SAME_MAG_TOL 1e-10
static int same_mags(const double *x, const double *y, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (fabs(x[i] - y[i]) > SAME_MAG_TOL && (x[i] < 100.0 || y[i] < 100.0)) return 0;
    return 1;
}

int main(void) {
    skip_without_data();
    int s = 0, failures = 0, nspec = 0, nbands = 0;

    int a = make_context();
    if (a < 0) { printf("FAIL: setup of context A\n"); return 1; }
    opensps_context_get_nspec(a, &nspec, &s);
    opensps_context_get_nbands(a, &nbands, &s);
    size_t nm = (size_t)nbands * N_MOD, ns = (size_t)nspec * N_MOD;
    double *mags0 = calloc(nm, sizeof(double)), *spec0 = calloc(ns, sizeof(double)), mass0[N_MOD];
    double *mags1 = calloc(nm, sizeof(double)), *spec1 = calloc(ns, sizeof(double)), mass1[N_MOD];

    s = batch(a, mags0, spec0, mass0);
    printf("context A, first batch: status %d, mass = %g %g\n", s, mass0[0], mass0[1]);
    if (s) { printf("FAIL: first batch\n"); return 1; }

    for (int round = 0; round < 2; round++) {
        int b = make_context();
        if (b < 0) { printf("FAIL: setup of context B\n"); return 1; }
        if (round == 1) {   /* the second round also runs a batch on B before destroying it */
            double *mb = calloc(nm, sizeof(double)), *sb = calloc(ns, sizeof(double)), massb[N_MOD];
            s = batch(b, mb, sb, massb);
            if (s || memcmp(massb, mass0, sizeof mass0) != 0) {
                printf("FAIL: context B batch (status %d, mass = %g %g)\n", s, massb[0], massb[1]);
                failures++;
            }
            free(mb); free(sb);
        }
        opensps_context_destroy(b, &s);

        s = batch(a, mags1, spec1, mass1);
        printf("context A, after B (round %d): status %d, mass = %g %g\n", round, s, mass1[0], mass1[1]);
        if (s || memcmp(mass1, mass0, sizeof mass0) != 0 || !same_mags(mags1, mags0, nm) ||
            memcmp(spec1, spec0, sizeof(double) * ns) != 0) {
            printf("FAIL: context A changed after context B (round %d): mass %s, mags %s, spec %s\n", round,
                   memcmp(mass1, mass0, sizeof mass0) ? "differ" : "equal",
                   same_mags(mags1, mags0, nm) ? "equal" : "differ",
                   memcmp(spec1, spec0, sizeof(double) * ns) ? "differ" : "equal");
            failures++;
        }
    }

    opensps_context_destroy(a, &s);
    free(mags0); free(spec0); free(mags1); free(spec1);
    printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
