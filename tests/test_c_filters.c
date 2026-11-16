/* Bandpass curves through the C API (doc/FILTERS_PLAN.md): filter sets built from arrays, a filter library
 * written by tools/opensps_data/filters_tool.py (its content hashes were computed by that tool in Python:
 * the Fortran SHA-256 and serialization must agree), the error codes, linear fluxes (maggies) of the batch,
 * and the band count of every output following the filters. Run by tests/run_filters_library_test.py with
 * OPENSPS_FILTERS_LIB (the library), OPENSPS_FILTERS_LIB_BAD (the same with a wrong hash) and
 * OPENSPS_FILTERS_HASH0/1 (the hashes of its two filters). */
#include "opensps.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_MOD 12
#define N_PAR 2

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

int main(void) {
    const char *lib = getenv("OPENSPS_FILTERS_LIB"), *bad = getenv("OPENSPS_FILTERS_LIB_BAD");
    const char *hash0 = getenv("OPENSPS_FILTERS_HASH0"), *hash1 = getenv("OPENSPS_FILTERS_HASH1");
    if (!lib || !bad || !hash0 || !hash1) { printf("SKIP: run by tests/run_filters_library_test.py\n"); return 77; }
    int status = 0, h = 0, nb0 = 0, nbands = 0, ntfull = 0;

    opensps_context_create(&h, &status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed\n"); return 1; }
    opensps_context_get_nbands(h, &nb0, &status);
    opensps_context_get_ntfull(h, &ntfull, &status);

    /* The default filters report their hashes */
    char *names = calloc(65 * nb0, 1), *hashes = calloc(65 * nb0, 1), grid_hash[65];
    double *cov = malloc(sizeof(double) * nb0);
    int *nfwhm = malloc(sizeof(int) * nb0), *flags = malloc(sizeof(int) * nb0), *det = malloc(sizeof(int) * nb0);
    int resample = -1, policy = -1;
    opensps_context_get_filter_report(h, names, hashes, cov, nfwhm, flags, det, grid_hash, &resample, &policy, &status);
    CHECK(status == 0 && strlen(hashes) == 64 && strlen(grid_hash) == 64, "default report: status %d", status);
    CHECK(resample == OPENSPS_RESAMPLE_LINEAR && policy == OPENSPS_GRID_POLICY_ALLOW, "default list: linear, allow");
    CHECK(strcmp(names, "V") == 0, "first default filter is V, got '%s'", names);

    /* Batch inputs */
    const char *pn[N_PAR] = {"dust2", "logzsol"};
    double dust2[N_MOD], logz[N_MOD];
    for (int m = 0; m < N_MOD; m++) { dust2[m] = 0.05 + 0.03 * m; logz[m] = -1.0 + 0.1 * m; }
    const double *pv[N_PAR] = {dust2, logz};
    double *w = malloc(sizeof(double) * ntfull * N_MOD);
    for (int m = 0; m < N_MOD; m++) {
        double sum = 0.0;
        for (int k = 0; k < ntfull; k++) {
            double x = (k - (0.3 + 0.02 * m) * ntfull) / (0.15 * ntfull);
            w[(size_t)m * ntfull + k] = exp(-0.5 * x * x) + 1e-3;
            sum += w[(size_t)m * ntfull + k];
        }
        for (int k = 0; k < ntfull; k++) w[(size_t)m * ntfull + k] /= sum;
    }
    double mass[N_MOD], lbol[N_MOD];

    /* Linear fluxes are opt-in */
    int n1, nb1;
    opensps_context_get_batch_maggies(h, NULL, &nb1, &n1, &status);
    CHECK(status == 302, "maggies before any batch: status %d (expected 302)", status);

    /* A set from arrays: a top-hat (photon), an energy-counting bell, and a calibrated one */
    int fs = 0;
    double lam1[2] = {4500.0, 5500.0}, r1[2] = {1.0, 1.0};
    double lam2[5] = {6000.0, 6500.0, 7000.0, 7500.0, 8000.0}, r2[5] = {0.0, 0.5, 1.0, 0.5, 0.0};
    opensps_filterset_create(&fs, &status);
    CHECK(status == 0 && fs >= 1, "filterset_create: status %d", status);
    opensps_filterset_add(fs, "tophat", 2, lam1, r1, OPENSPS_DETECTOR_PHOTON, 0.0, 0.0, &status);
    CHECK(status == 0, "filterset_add: status %d", status);
    opensps_filterset_add(fs, "bell", 5, lam2, r2, OPENSPS_DETECTOR_ENERGY, 7000.0, -1.0, &status);
    CHECK(status == 0, "filterset_add (energy, calibrated): status %d", status);
    opensps_filterset_add(fs, "short", 1, lam1, r1, 0, 0.0, 0.0, &status);
    CHECK(status == 3, "a one-point curve: status %d (expected 3)", status);
    opensps_filterset_add(fs, "baddet", 2, lam1, r1, 5, 0.0, 0.0, &status);
    CHECK(status == 3, "a bad detector: status %d (expected 3)", status);
    int cnt = 0;
    opensps_filterset_count(fs, &cnt, &status);
    CHECK(cnt == 2, "the set has %d filters (expected 2)", cnt);

    opensps_context_set_filters(h, fs, OPENSPS_RESAMPLE_CONSERVING, OPENSPS_GRID_POLICY_ERROR, 0, &status);
    CHECK(status == 0, "set_filters: status %d", status);
    opensps_filterset_free(fs, &status);   /* the context holds its own copy */
    opensps_context_get_nbands(h, &nbands, &status);
    CHECK(nbands == 2, "nbands %d after set_filters (expected 2)", nbands);
    double leff[2], magv[2], mags_sun[2];
    opensps_context_get_filter_data(h, leff, magv, mags_sun, &status);
    CHECK(status == 0 && leff[0] > 4500.0 && leff[0] < 5500.0 && leff[1] > 6000.0 && leff[1] < 8000.0,
          "pivot wavelengths %g %g", leff[0], leff[1]);
    char nm2[130], hs2[130];
    int cv2[2], fl2[2], dt2[2], nf2[2];
    double co2[2];
    opensps_context_get_filter_report(h, nm2, hs2, co2, nf2, fl2, dt2, grid_hash, &resample, &policy, &status);
    CHECK(status == 0 && strcmp(nm2, "tophat") == 0 && strcmp(nm2 + 65, "bell") == 0 && dt2[0] == 0 && dt2[1] == 1 &&
          resample == OPENSPS_RESAMPLE_CONSERVING && policy == OPENSPS_GRID_POLICY_ERROR && co2[0] > 0.999 &&
          fl2[0] == 0 && fl2[1] == 0, "report of the user set (%s, %s)", nm2, nm2 + 65);
    (void)cv2;

    /* Batch: bands follow the set, AB is the default, and the linear fluxes are 10^(-0.4 m) */
    double *mags = malloc(sizeof(double) * nbands * N_MOD), *mag2 = malloc(sizeof(double) * nbands * N_MOD);
    double *mg = malloc(sizeof(double) * nbands * N_MOD);
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, pn, pv, w, mags, NULL, mass, lbol, NULL, NULL, NULL, NULL, &status);
    CHECK(status == 0, "batch with 2 bands: status %d", status);
    int finite = 1;
    for (int i = 0; i < nbands * N_MOD; i++) if (!isfinite(mags[i])) finite = 0;
    CHECK(finite, "the batch magnitudes are finite");
    opensps_context_set_int(h, "keep_maggies", 1, &status);
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, pn, pv, w, mag2, NULL, mass, lbol, NULL, NULL, NULL, NULL, &status);
    CHECK(status == 0 && memcmp(mags, mag2, sizeof(double) * nbands * N_MOD) == 0,
          "keep_maggies does not change the magnitudes (status %d)", status);
    opensps_context_get_batch_maggies(h, mg, &nb1, &n1, &status);
    CHECK(status == 0 && nb1 == 2 && n1 == N_MOD, "maggies dimensions %d x %d, status %d", nb1, n1, status);
    double worst = 0.0;
    for (int i = 0; i < nbands * N_MOD; i++) {
        double e = fabs(mg[i] - pow(10.0, -0.4 * mags[i])) / mg[i];
        if (e > worst) worst = e;
    }
    CHECK(worst < 1e-9, "maggies = 10^(-0.4 m): worst relative difference %g", worst);
    opensps_context_set_int(h, "compute_vega_mags", 1, &status);
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, pn, pv, w, mag2, NULL, mass, lbol, NULL, NULL, NULL, NULL, &status);
    double *mg2 = malloc(sizeof(double) * nbands * N_MOD);
    opensps_context_get_batch_maggies(h, mg2, &nb1, &n1, &status);
    CHECK(memcmp(mg, mg2, sizeof(double) * nbands * N_MOD) == 0, "the linear fluxes are AB whatever the Vega setting");
    int differ = 0;
    for (int i = 0; i < nbands * N_MOD; i++) if (fabs(mag2[i] - mags[i]) > 1e-6) differ = 1;
    CHECK(differ, "with compute_vega_mags = 1 the magnitudes are in the Vega system (they differ from AB)");
    opensps_context_set_int(h, "compute_vega_mags", 0, &status);
    int mask[1] = {1};
    opensps_context_set_mag_compute(h, 1, mask, &status);
    opensps_context_compute_csp_batch(h, N_MOD, N_PAR, pn, pv, w, mag2, NULL, mass, lbol, NULL, NULL, NULL, NULL, &status);
    opensps_context_get_batch_maggies(h, mg2, &nb1, &n1, &status);
    int ok = 1;
    for (int m = 0; m < N_MOD; m++) { if (mg2[2 * m] != mg[2 * m] || mg2[2 * m + 1] == mg2[2 * m + 1]) ok = 0; }
    CHECK(ok, "a masked band is NaN in the linear fluxes, the others unchanged");
    /* The other ways of computing the magnitudes (per-model magnitude redshifts, per-model "zred" parameters), each
     * with a band masked out and without: the linear fluxes follow in every one */
    double zm[N_MOD], zpar[N_MOD], *mg3 = malloc(sizeof(double) * nbands * N_MOD), *mag3 = malloc(sizeof(double) * nbands * N_MOD);
    for (int m = 0; m < N_MOD; m++) { zm[m] = 0.05 * m; zpar[m] = 0.02 * m; }
    const char *pnz[N_PAR + 1] = {"dust2", "logzsol", "zred"};
    const double *pvz[N_PAR + 1] = {dust2, logz, zpar};
    for (int variant = 0; variant < 4; variant++) {
        int masked = variant % 2, per_param = variant / 2;
        opensps_context_set_mag_compute(h, masked ? 1 : 0, masked ? mask : NULL, &status);
        opensps_batch_options_t opts = {0};
        opts.n = N_MOD; opts.n_params = per_param ? N_PAR + 1 : N_PAR;
        opts.names = per_param ? pnz : pn; opts.values = per_param ? pvz : pv;
        opts.ssp_weights = w; opts.zred_mags = per_param ? NULL : zm;
        opensps_batch_results_t res = {0};
        res.mags = mag3; res.mass = mass; res.lbol = lbol;
        opensps_context_compute_csp_batch_ext(h, &opts, &res, &status);
        CHECK(status == 0, "variant %d (masked %d, per-model %s): status %d", variant, masked,
              per_param ? "zred" : "zred_mags", status);
        opensps_context_get_batch_maggies(h, mg3, &nb1, &n1, &status);
        int good = (status == 0 && nb1 == 2 && n1 == N_MOD);
        for (int m = 0; m < N_MOD && good; m++) {
            if (!(fabs(mg3[2 * m] - pow(10.0, -0.4 * mag3[2 * m])) <= 1e-9 * mg3[2 * m])) good = 0;
            if (masked ? (mg3[2 * m + 1] == mg3[2 * m + 1]) : !(fabs(mg3[2 * m + 1] - pow(10.0, -0.4 * mag3[2 * m + 1])) <=
                                                                1e-9 * mg3[2 * m + 1])) good = 0;
        }
        CHECK(good, "variant %d (masked %d, per-model %s): maggies = 10^(-0.4 m), a masked band is NaN", variant, masked,
              per_param ? "zred" : "zred_mags");
        int moved = 0;
        for (int m = 1; m < N_MOD; m++) if (fabs(mag3[2 * m] - mags[2 * m]) > 1e-6) moved = 1;
        CHECK(moved, "variant %d: the redshift changes the magnitudes", variant);
    }
    free(mg3); free(mag3);
    opensps_context_set_mag_compute(h, 0, NULL, &status);
    opensps_context_set_int(h, "keep_maggies", 0, &status);

    /* A library written by filters_tool.py: its hashes (Python) agree with the Fortran ones */
    opensps_context_set_filters_library(h, lib, OPENSPS_RESAMPLE_CONSERVING, OPENSPS_GRID_POLICY_ERROR, 0, &status);
    CHECK(status == 0, "set_filters_library: status %d", status);
    opensps_context_get_nbands(h, &nbands, &status);
    CHECK(nbands == 2, "library nbands %d", nbands);
    opensps_context_get_filter_report(h, nm2, hs2, co2, nf2, fl2, dt2, grid_hash, &resample, &policy, &status);
    CHECK(strcmp(hs2, hash0) == 0 && strcmp(hs2 + 65, hash1) == 0,
          "hashes: C/Fortran %s %s, Python %s %s", hs2, hs2 + 65, hash0, hash1);
    CHECK(dt2[0] == 0 && dt2[1] == 1, "detectors from the manifest: %d %d (expected 0 1)", dt2[0], dt2[1]);

    /* Errors: the context keeps its filters */
    opensps_context_set_filters_library(h, bad, OPENSPS_RESAMPLE_CONSERVING, OPENSPS_GRID_POLICY_ERROR, 0, &status);
    CHECK(status == 323, "a library with a wrong hash: status %d (expected 323)", status);
    char msg[256];
    int st2;
    opensps_get_last_error(&st2, msg, sizeof msg);
    CHECK(st2 == 323 && strstr(msg, "hash") != NULL, "the message names the hash: %s", msg);
    opensps_context_set_filters_library(h, "no_such_library", 1, 0, 0, &status);
    CHECK(status == 324, "a missing library: status %d (expected 324)", status);
    opensps_context_get_nbands(h, &nbands, &status);
    CHECK(nbands == 2, "unchanged after the errors: nbands %d", nbands);

    double nan_l[3] = {4000.0, NAN, 5000.0}, one3[3] = {1.0, 1.0, 1.0};
    opensps_filterset_create(&fs, &status);
    opensps_filterset_add(fs, "nan_curve", 3, nan_l, one3, 0, 0.0, 0.0, &status);
    opensps_context_set_filters(h, fs, 1, 0, 0, &status);
    CHECK(status == 320, "NaN wavelength: status %d (expected 320)", status);
    opensps_context_set_filters(h, fs, 1, 0, 9, &status);
    CHECK(status == 3, "a Vega reference band beyond the set: status %d (expected 3)", status);
    opensps_context_set_filters(-3, fs, 1, 0, 0, &status);
    CHECK(status == 1, "bad context handle: status %d (expected 1)", status);
    opensps_context_set_filters(h, 9999, 1, 0, 0, &status);
    CHECK(status == 1, "bad set handle: status %d (expected 1)", status);
    opensps_filterset_free(fs, &status);
    opensps_filterset_free(fs, &status);
    CHECK(status == 1, "freeing a set twice: status %d (expected 1)", status);

    /* The next setup restores the default list */
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    opensps_context_get_nbands(h, &nbands, &status);
    CHECK(nbands == nb0, "re-setup: nbands %d (expected %d)", nbands, nb0);

    opensps_context_destroy(h, &status);
    if (failures == 0) printf("all filter checks passed\n");
    return failures ? 1 : 0;
}
