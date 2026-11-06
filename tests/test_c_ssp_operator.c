#include "opensps.h"
#include "test_c_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    skip_without_data();
    int status = 0, h = 0, nspec = 0, ntfull = 0, has_ops = 0, failures = 0;

    opensps_context_create(&h, &status);
    if (status != 0) { printf("FAIL: context_create\n"); return 1; }

    /* Test build_ssp_ops before setup (status 301) */
    opensps_context_build_ssp_ops(h, 0, &status);
    if (status != 301) {
        printf("FAIL: build_ssp_ops before setup expected 301, got %d\n", status);
        failures++;
    }

    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("FAIL: context_setup\n"); return 1; }

    opensps_context_get_nspec(h, &nspec, &status);
    opensps_context_get_ntfull(h, &ntfull, &status);

    double *mass = (double *)malloc(sizeof(double) * ntfull);
    double *lbol = (double *)malloc(sizeof(double) * ntfull);
    double *spec = (double *)malloc(sizeof(double) * nspec * ntfull);

    opensps_context_has_ssp_ops(h, &has_ops, &status);
    if (status != 0 || has_ops != 0) {
        printf("FAIL: has_ssp_ops before build (has_ops=%d)\n", has_ops);
        failures++;
    }

    /* Error handling: bad handles */
    int bad_h = 999;
    opensps_context_build_ssp_ops(bad_h, 0, &status);
    if (status != 1) { printf("FAIL: build_ssp_ops bad handle\n"); failures++; }
    opensps_context_free_ssp_ops(bad_h, &status);
    if (status != 1) { printf("FAIL: free_ssp_ops bad handle\n"); failures++; }
    opensps_context_has_ssp_ops(bad_h, &has_ops, &status);
    if (status != 1) { printf("FAIL: has_ssp_ops bad handle\n"); failures++; }
    opensps_context_evaluate_ssp_op(bad_h, 2, NULL, NULL, NULL, NULL, &status);
    if (status != 1) { printf("FAIL: evaluate_ssp_op bad handle\n"); failures++; }

    /* Error handling: evaluate before build */
    opensps_context_evaluate_ssp_op(h, 2, NULL, mass, lbol, spec, &status);
    if (status != 303) { printf("FAIL: evaluate_ssp_op before build (status=%d)\n", status); failures++; }

    /* Build single-metallicity SSP operator */
    opensps_context_build_ssp_ops(h, 0, &status);
    if (status != 0) {
        printf("FAIL: build_ssp_ops status=%d\n", status);
        failures++;
    }

    /* Re-build over existing operator (exercises deallocation of previous opset) */
    opensps_context_build_ssp_ops(h, 0, &status);
    if (status != 0) {
        printf("FAIL: re-build_ssp_ops status=%d\n", status);
        failures++;
    }

    /* Error handling: evaluate with NULL output pointers */
    opensps_context_evaluate_ssp_op(h, 2, NULL, NULL, lbol, spec, &status);
    if (status != 3) { printf("FAIL: evaluate_ssp_op NULL pointer\n"); failures++; }

    /* Evaluate with NULL imf_params (uses context IMF parameters) */
    opensps_context_evaluate_ssp_op(h, 2, NULL, mass, lbol, spec, &status);
    if (status != 0) { printf("FAIL: evaluate_ssp_op NULL imf_params\n"); failures++; }

    /* Build all-metallicities SSP operator */
    opensps_context_build_ssp_ops(h, 1, &status);
    if (status != 0) {
        printf("FAIL: build_ssp_ops all_metallicities status=%d\n", status);
        failures++;
    }

    /* Evaluate with all-metallicities operator active */
    opensps_context_evaluate_ssp_op(h, 2, NULL, mass, lbol, spec, &status);
    if (status != 0) {
        printf("FAIL: evaluate_ssp_op with all-metallicities opset status=%d\n", status);
        failures++;
    }

    /* Parameter change invalidates active ssp_ops */
    opensps_context_set_int(h, "imf_type", 1, &status);
    opensps_context_has_ssp_ops(h, &has_ops, &status);
    if (has_ops != 0) {
        printf("FAIL: ssp_ops was not invalidated after imf_type change\n");
        failures++;
    }

    /* Rebuild for further testing */
    opensps_context_set_int(h, "imf_type", 2, &status);
    opensps_context_build_ssp_ops(h, 0, &status);

    opensps_context_has_ssp_ops(h, &has_ops, &status);
    if (status != 0 || has_ops != 1) {
        printf("FAIL: has_ssp_ops after build (has_ops=%d)\n", has_ops);
        failures++;
    }

    /* Evaluate SSP operator for Kroupa IMF */
    double kroupa_params[5] = {1.3, 2.3, 2.3, 0.0, 0.0};
    opensps_context_evaluate_ssp_op(h, 2, kroupa_params, mass, lbol, spec, &status);
    if (status != 0) {
        printf("FAIL: evaluate_ssp_op status=%d\n", status);
        failures++;
    }

    /* Check outputs are positive and finite */
    for (int t = 0; t < ntfull; t++) {
        if (mass[t] <= 0.0 || isnan(mass[t]) || isinf(mass[t])) {
            printf("FAIL: mass[%d]=%e\n", t, mass[t]);
            failures++;
            break;
        }
        if (isnan(lbol[t]) || isinf(lbol[t])) {
            printf("FAIL: lbol[%d]=%e\n", t, lbol[t]);
            failures++;
            break;
        }
    }

    double spec_sum = 0.0;
    for (int i = 0; i < nspec * ntfull; i++) {
        spec_sum += spec[i];
    }
    if (spec_sum <= 0.0 || isnan(spec_sum)) {
        printf("FAIL: spec_sum=%e\n", spec_sum);
        failures++;
    }

    /* Free SSP operator */
    opensps_context_free_ssp_ops(h, &status);
    if (status != 0) {
        printf("FAIL: free_ssp_ops\n");
        failures++;
    }

    opensps_context_has_ssp_ops(h, &has_ops, &status);
    if (has_ops != 0) {
        printf("FAIL: has_ssp_ops after free (has_ops=%d)\n", has_ops);
        failures++;
    }

    free(mass);
    free(lbol);
    free(spec);
    opensps_context_destroy(h, &status);

    if (failures == 0) {
        printf("test_c_ssp_operator: ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("test_c_ssp_operator: %d FAILURES\n", failures);
        return 1;
    }
}
