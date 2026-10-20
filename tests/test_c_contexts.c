#include "fsps.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int nearly_equal(double a, double b, double tol) {
    return fabs(a - b) <= tol;
}

int main(void) {
    int status = 0;
    int h1 = 0, h2 = 0;
    int nspec = 0, ntfull = 0;
    size_t n_spec = 0;
    double *spec1 = NULL, *spec2 = NULL;
    double *mass1 = NULL, *mass2 = NULL;
    double *lbol1 = NULL, *lbol2 = NULL;

    printf("--- FSPS Context Isolation Test ---\n");

    fsps_context_create(&h1, &status);
    if (status != 0) {
        printf("[C] Failed to create context 1\n");
        return 1;
    }
    fsps_context_create(&h2, &status);
    if (status != 0) {
        printf("[C] Failed to create context 2\n");
        return 1;
    }

    fsps_context_setup(-1, "mist", "miles", "DL07", h1, &status);
    if (status != 0) {
        printf("[C] Failed to setup context 1\n");
        return 1;
    }
    fsps_context_setup(-1, "mist", "miles", "DL07", h2, &status);
    if (status != 0) {
        printf("[C] Failed to setup context 2\n");
        return 1;
    }

    fsps_context_set_int(h1, "zmet", 1, &status);
    if (status != 0) {
        printf("[C] Failed to set zmet for context 1\n");
        return 1;
    }
    fsps_context_set_int(h2, "zmet", 2, &status);
    if (status != 0) {
        printf("[C] Failed to set zmet for context 2\n");
        return 1;
    }

    fsps_context_get_nspec(h1, &nspec, &status);
    if (status != 0) {
        printf("[C] Failed to get nspec for context 1\n");
        return 1;
    }
    fsps_context_get_ntfull(h1, &ntfull, &status);
    if (status != 0) {
        printf("[C] Failed to get ntfull for context 1\n");
        return 1;
    }
    if (nspec <= 0 || ntfull <= 0) {
        printf("[C] Invalid dimensions: nspec=%d ntfull=%d\n", nspec, ntfull);
        return 1;
    }

    n_spec = (size_t)nspec * (size_t)ntfull;
    spec1 = (double *)calloc(n_spec, sizeof(double));
    spec2 = (double *)calloc(n_spec, sizeof(double));
    mass1 = (double *)calloc((size_t)ntfull, sizeof(double));
    mass2 = (double *)calloc((size_t)ntfull, sizeof(double));
    lbol1 = (double *)calloc((size_t)ntfull, sizeof(double));
    lbol2 = (double *)calloc((size_t)ntfull, sizeof(double));

    if (!spec1 || !spec2 || !mass1 || !mass2 || !lbol1 || !lbol2) {
        printf("[C] Allocation failed\n");
        return 1;
    }

    fsps_context_compute_ssp(h1, spec1, mass1, lbol1, &status);
    if (status != 0) {
        printf("[C] SSP compute failed for context 1\n");
        return 1;
    }
    fsps_context_compute_ssp(h2, spec2, mass2, lbol2, &status);
    if (status != 0) {
        printf("[C] SSP compute failed for context 2\n");
        return 1;
    }

    printf("[C] Spec1[0]=%e Spec2[0]=%e\n", spec1[0], spec2[0]);
    printf("[C] Spec1[last]=%e Spec2[last]=%e\n", spec1[n_spec - 1], spec2[n_spec - 1]);
    printf("[C] Lbol1[0]=%e Lbol2[0]=%e\n", lbol1[0], lbol2[0]);

    if (nearly_equal(spec1[0], spec2[0], 1e-20) &&
        nearly_equal(spec1[n_spec - 1], spec2[n_spec - 1], 1e-20) &&
        nearly_equal(lbol1[0], lbol2[0], 1e-20)) {
        printf("[C] Context outputs are unexpectedly identical\n");
        return 1;
    }

    fsps_context_destroy(h1, &status);
    fsps_context_destroy(h2, &status);

    free(spec1);
    free(spec2);
    free(mass1);
    free(mass2);
    free(lbol1);
    free(lbol2);

    printf("[C] Context isolation test PASS\n");
    return 0;
}
