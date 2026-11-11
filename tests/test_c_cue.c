/* cue's nebular emission through the C API: the context option nebular_model = 1 and the gas
 * parameters gas_logn, gas_logno and gas_logco varied per model in a batch. Without
 * OPENSPS_CUE_DATA the batch returns status 4 (data missing). With it: status 0, and the [O II]
 * 3726/3729 doublet behaves as a density diagnostic (Osterbrock & Ferland 2006, Sec. 5.6): the
 * ratio L(3729) / L(3726) falls as n_H rises from 10 to 1e4 cm^-3, from near its low-density limit
 * (~1.5) toward its high-density one (~0.35); [N II] 6585 grows with N/O at fixed everything
 * else; an option value other than 0 or 1 is rejected. Requires a data root, skipped without one. */
#include "opensps.h"
#include "test_c_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_MOD 8
#define N_PAR 3

static int nearest(const double *lam, int n, double x) {
    int k = 0;
    for (int i = 1; i < n; i++)
        if (fabs(lam[i] - x) < fabs(lam[k] - x)) k = i;
    return k;
}

int main(void) {
    skip_without_data();
    int status = 0, h = 0, ntfull = 0, nbands = 0, nline = 0, failures = 0;
    const char *names[N_PAR] = {"gas_logn", "gas_logno", "gas_logco"};
    double logn[N_MOD], logno[N_MOD], logco[N_MOD];
    for (int m = 0; m < N_MOD; m++) {
        /* models 0-3: density 1e1 to 1e4 at solar N/O; models 4-7: N/O 0.1 to 3x solar at n = 100 */
        logn[m] = m < 4 ? 1.0 + m : 2.0;
        logno[m] = m < 4 ? 0.0 : -1.0 + 0.5 * (m - 4);
        logco[m] = 0.0;
    }
    const double *values[N_PAR] = {logn, logno, logco};

    opensps_context_create(&h, &status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed\n"); return 1; }
    opensps_context_set_int(h, "add_neb_emission", 1, &status);
    opensps_context_set_int(h, "nebular_model", 7, &status);
    if (status == 0) { printf("FAIL: nebular_model = 7 accepted\n"); failures++; }
    opensps_context_set_int(h, "nebular_model", 1, &status);
    if (status != 0) { printf("FAIL: nebular_model = 1 rejected (%d)\n", status); return 1; }
    opensps_context_set_int(h, "cue_fp64", 1, &status);
    if (status != 0) { printf("FAIL: cue_fp64 rejected (%d)\n", status); return 1; }
    opensps_context_get_ntfull(h, &ntfull, &status);
    opensps_context_get_nbands(h, &nbands, &status);
    opensps_context_get_nemline(h, &nline, &status);

    /* A young population: constant SFR over the last 30 Myr */
    double *w = calloc((size_t)ntfull * N_MOD, sizeof(double));
    double *age = malloc(sizeof(double) * ntfull);
    opensps_context_get_ages(h, age, &status);
    for (int m = 0; m < N_MOD; m++)
        for (int k = 0; k < ntfull; k++)
            if (age[k] <= log10(3e7)) w[(size_t)m * ntfull + k] = 1e-3;
    double *mags = malloc(sizeof(double) * nbands * N_MOD), *eml = malloc(sizeof(double) * nline * N_MOD);
    double *lam = malloc(sizeof(double) * nline);
    opensps_context_get_emlambda(h, lam, &status);

    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, names, values, w, mags, NULL, NULL, NULL,
                                   NULL, NULL, eml, NULL, &status);
    if (getenv("OPENSPS_CUE_DATA") == NULL) {
        if (status != 4) { printf("FAIL: without OPENSPS_CUE_DATA: status %d, expected 4\n", status); failures++; }
        else printf("# OPENSPS_CUE_DATA is not set: status 4 as expected; the cue checks are skipped\n");
        opensps_context_destroy(h, &status);
        printf(failures ? "FAILED\n" : "PASSED\n");
        return failures ? 1 : 0;
    }
    if (status != 0) { printf("FAIL: batch status %d\n", status); return 1; }

    int i3726 = nearest(lam, nline, 3727.1), i3729 = nearest(lam, nline, 3729.9), i6585 = nearest(lam, nline, 6585.3);
    double prev = 1e30;
    for (int m = 0; m < 4; m++) {
        double r = eml[(size_t)m * nline + i3729] / eml[(size_t)m * nline + i3726];
        printf("# log n_H = %.0f: L(3729) / L(3726) = %.3f\n", logn[m], r);
        if (!(r < prev)) { printf("FAIL: [O II] ratio does not fall with density\n"); failures++; }
        prev = r;
    }
    double r_lo = eml[i3729] / eml[i3726], r_hi = eml[(size_t)3 * nline + i3729] / eml[(size_t)3 * nline + i3726];
    if (!(r_lo > 1.2 && r_hi < 0.7)) {
        printf("FAIL: [O II] ratio %.3f at 1e1 cm^-3 and %.3f at 1e4 cm^-3 (expected > 1.2 and < 0.7)\n", r_lo, r_hi);
        failures++;
    }
    prev = 0.0;
    for (int m = 4; m < 8; m++) {
        double n2 = eml[(size_t)m * nline + i6585];
        printf("# log N/O = %+.1f: L([N II] 6585) = %.4e L_sun\n", logno[m], n2);
        if (!(n2 > prev)) { printf("FAIL: [N II] does not grow with N/O\n"); failures++; }
        prev = n2;
    }

    opensps_context_destroy(h, &status);
    printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
