/* Filter sets through the C API that need no data (the checks with a data root are in test_c_filters.c):
 * the handle pool (growth past its initial size, slot reuse after a free), the statuses of a bad handle, a
 * NULL array and a bad argument, and the filter calls on a context that was never set up. */
#include "opensps.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

#define N_SETS 11 /* more than the initial pool (4), enough for two doublings */

int main(void) {
    int status = 0, n = -1, fs[N_SETS], i, j;
    const double lam[3] = {4000.0, 5000.0, 6000.0}, resp[3] = {0.0, 1.0, 0.0};

    /* Pool growth: every handle is distinct and positive, and the sets stay independent */
    for (i = 0; i < N_SETS; i++) {
        status = -1;
        opensps_filterset_create(&fs[i], &status);
        CHECK(status == 0 && fs[i] >= 1, "create %d: status %d handle %d", i, status, fs[i]);
        for (j = 0; j < i; j++) CHECK(fs[i] != fs[j], "handles %d and %d coincide (%d)", i, j, fs[i]);
        for (j = 0; j <= i % 3; j++) {
            opensps_filterset_add(fs[i], "f", 3, lam, resp, OPENSPS_DETECTOR_PHOTON, 0.0, 0.0, &status);
            CHECK(status == 0, "add to set %d: status %d", i, status);
        }
    }
    for (i = 0; i < N_SETS; i++) {
        opensps_filterset_count(fs[i], &n, &status);
        CHECK(status == 0 && n == i % 3 + 1, "set %d holds %d filters (expected %d)", i, n, i % 3 + 1);
    }

    /* A freed slot is reused, empty, and the other sets are untouched */
    opensps_filterset_free(fs[2], &status);
    CHECK(status == 0, "free: status %d", status);
    opensps_filterset_count(fs[2], &n, &status);
    CHECK(status == 1 && n == 0, "count of a freed set: status %d n %d (expected 1, 0)", status, n);
    int again = 0;
    opensps_filterset_create(&again, &status);
    CHECK(status == 0 && again == fs[2], "the freed slot %d is reused (got %d)", fs[2], again);
    opensps_filterset_count(again, &n, &status);
    CHECK(status == 0 && n == 0, "the reused set is empty: n %d", n);
    opensps_filterset_count(fs[3], &n, &status);
    CHECK(status == 0 && n == 1, "a neighbour is untouched: n %d", n);
    fs[2] = again;

    /* Bad handles: 0, negative, beyond the pool */
    int bad_handles[3] = {0, -2, 100000};
    for (i = 0; i < 3; i++) {
        status = 0;
        opensps_filterset_add(bad_handles[i], "f", 3, lam, resp, 0, 0.0, 0.0, &status);
        CHECK(status == 1, "add to handle %d: status %d (expected 1)", bad_handles[i], status);
        status = 0; n = -1;
        opensps_filterset_count(bad_handles[i], &n, &status);
        CHECK(status == 1 && n == 0, "count of handle %d: status %d n %d", bad_handles[i], status, n);
        status = 0;
        opensps_filterset_free(bad_handles[i], &status);
        CHECK(status == 1, "free of handle %d: status %d (expected 1)", bad_handles[i], status);
    }

    /* Bad arguments leave the set as it was */
    opensps_filterset_count(fs[0], &n, &status);
    int before = n;
    opensps_filterset_add(fs[0], "null_lambda", 3, NULL, resp, 0, 0.0, 0.0, &status);
    CHECK(status == 3, "NULL wavelengths: status %d (expected 3)", status);
    opensps_filterset_add(fs[0], "null_response", 3, lam, NULL, 0, 0.0, 0.0, &status);
    CHECK(status == 3, "NULL response: status %d (expected 3)", status);
    opensps_filterset_add(fs[0], "one_point", 1, lam, resp, 0, 0.0, 0.0, &status);
    CHECK(status == 3, "one point: status %d (expected 3)", status);
    opensps_filterset_add(fs[0], "bad_detector", 3, lam, resp, 7, 0.0, 0.0, &status);
    CHECK(status == 3, "bad detector: status %d (expected 3)", status);
    opensps_filterset_add(fs[0], "bad_calibration", 3, lam, resp, 0, 5000.0, NAN, &status);
    CHECK(status == 3, "NaN calibration exponent: status %d (expected 3)", status);
    opensps_filterset_count(fs[0], &n, &status);
    CHECK(status == 0 && n == before, "the refused calls left the set at %d filters (now %d)", before, n);
    opensps_filterset_add(fs[0], "calibrated", 3, lam, resp, OPENSPS_DETECTOR_ENERGY, 5000.0, -1.0, &status);
    CHECK(status == 0, "a calibrated filter is accepted: status %d", status);

    /* A context that was never set up has no filters to replace or report */
    int h = 0;
    opensps_context_create(&h, &status);
    opensps_context_set_filters(h, fs[0], 1, 0, 0, &status);
    CHECK(status == 301, "set_filters on a context that is not set up: status %d (expected 301)", status);
    opensps_context_set_filters(h, 100000, 1, 0, 0, &status);
    CHECK(status == 1, "set_filters with a bad set handle: status %d (expected 1)", status);
    opensps_context_set_filters(-3, fs[0], 1, 0, 0, &status);
    CHECK(status == 1, "set_filters with a bad context handle: status %d (expected 1)", status);

    char names[65 * 4], hashes[65 * 4], grid_hash[65];
    double coverage[4];
    int n_fwhm[4], flags[4], detector[4], resample = -1, policy = -1;
    opensps_context_get_filter_report(h, names, hashes, coverage, n_fwhm, flags, detector, grid_hash,
                                      &resample, &policy, &status);
    CHECK(status == 301, "filter report of a context that is not set up: status %d (expected 301)", status);
    opensps_context_get_filter_report(h, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &resample, &policy, &status);
    CHECK(status == 301, "the same with every output NULL: status %d (expected 301)", status);
    opensps_context_get_filter_report(-3, names, hashes, coverage, n_fwhm, flags, detector, grid_hash,
                                      &resample, &policy, &status);
    CHECK(status == 1, "filter report of a bad handle: status %d (expected 1)", status);

    opensps_context_set_filters_library(h, "no_such_library", 1, 0, 0, &status);
    CHECK(status != 0, "a library on a context that is not set up: status %d (expected an error)", status);
    opensps_context_set_filters_library(-3, "no_such_library", 1, 0, 0, &status);
    CHECK(status == 1, "a library on a bad handle: status %d (expected 1)", status);
    opensps_context_destroy(h, &status);

    for (i = 0; i < N_SETS; i++) opensps_filterset_free(fs[i], &status);
    if (failures == 0) printf("all filter-set checks passed\n");
    return failures ? 1 : 0;
}
