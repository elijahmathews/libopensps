#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fsps.h"

static void check_alloc(void *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "[C] Error: Memory allocation failed!\n");
        exit(1);
    }
}

static int check_status(int status, const char *msg) {
    if (status != FSPS_STATUS_OK) {
        fprintf(stderr, "[C] Error: %s (status=%d)\n", msg, status);
        return 1;
    }
    return 0;
}

static int check_nonempty(const char *label, const char *value) {
    if (value == NULL || value[0] == '\0') {
        fprintf(stderr, "[C] Error: %s is empty\n", label);
        return 1;
    }
    return 0;
}

int main() {
    printf("--- C Driver API Test for FSPS ---\n");

    int status = 0;
    int handle = 0;
    int nspec = 0, ntfull = 0, nbands = 0;
    int n_spec = 0, n_time = 0;
    int nindx = 0, nz = 0, nemline = 0;
    double zsol = 0.0;
    int v_major = 0, v_minor = 0, v_patch = 0;
    int err_status = 0;
    char err_msg[256] = {0};
    char sps_home[256] = {0};
    char data_home[256] = {0};
    char out_home[256] = {0};

    fsps_clear_error();
    fsps_get_last_error(&err_status, err_msg, (int)sizeof(err_msg));
    if (err_status != 0) {
        fprintf(stderr, "[C] Error: unexpected error state at start: %s\n", err_msg);
        return 1;
    }

    fsps_get_driver_version(&v_major, &v_minor, &v_patch);
    printf("[C] driver_version=%d.%d.%d\n", v_major, v_minor, v_patch);

    fsps_unlock(&status);
    if (status == FSPS_STATUS_OK) {
        fprintf(stderr, "[C] Error: unlock should fail when no lock is held\n");
        return 1;
    }
    fsps_get_last_error(&err_status, err_msg, (int)sizeof(err_msg));
    printf("[C] unlock(no lock) status=%d msg=%s\n", err_status, err_msg);

    fsps_clear_error();
    fsps_lock(&status);
    if (check_status(status, "fsps_lock failed")) {
        return 1;
    }
    fsps_lock(&status);
    if (status == FSPS_STATUS_OK) {
        fprintf(stderr, "[C] Error: second lock should fail\n");
        return 1;
    }
    fsps_get_last_error(&err_status, err_msg, (int)sizeof(err_msg));
    printf("[C] lock(held) status=%d msg=%s\n", err_status, err_msg);
    fsps_unlock(&status);
    if (check_status(status, "fsps_unlock failed")) {
        return 1;
    }

    fsps_context_create(&handle, &status);
    if (check_status(status, "failed to create context")) {
        return 1;
    }

    fsps_context_setup(-1, "mist", "miles", "DL07", handle, &status);
    if (check_status(status, "failed to setup context")) {
        return 1;
    }

    fsps_context_get_paths(handle, sps_home, (int)sizeof(sps_home),
                           data_home, (int)sizeof(data_home),
                           out_home, (int)sizeof(out_home));
    if (check_nonempty("SPS_HOME", sps_home) ||
        check_nonempty("FSPS_DATA_HOME", data_home) ||
        check_nonempty("FSPS_OUTPUT_HOME", out_home)) {
        return 1;
    }
    printf("[C] paths: sps=%s data=%s out=%s\n", sps_home, data_home, out_home);

    fsps_context_get_dims(handle, &n_spec, &n_time, &status);
    if (check_status(status, "failed to get dims")) {
        return 1;
    }
    fsps_context_get_nspec(handle, &nspec, &status);
    if (check_status(status, "failed to get nspec")) {
        return 1;
    }
    fsps_context_get_ntfull(handle, &ntfull, &status);
    if (check_status(status, "failed to get ntfull")) {
        return 1;
    }
    fsps_context_get_nbands(handle, &nbands, &status);
    if (check_status(status, "failed to get nbands")) {
        return 1;
    }
    fsps_context_get_nindx(handle, &nindx, &status);
    if (check_status(status, "failed to get nindx")) {
        return 1;
    }
    fsps_context_get_nz(handle, &nz, &status);
    if (check_status(status, "failed to get nz")) {
        return 1;
    }
    fsps_context_get_nemline(handle, &nemline, &status);
    if (check_status(status, "failed to get nemline")) {
        return 1;
    }
    fsps_context_get_zsol(handle, &zsol, &status);
    if (check_status(status, "failed to get zsol")) {
        return 1;
    }

    if (nspec <= 0 || ntfull <= 0 || nbands <= 0 || nindx < 0 || nz <= 0 || nemline <= 0) {
        fprintf(stderr, "[C] Error: invalid dimensions nspec=%d ntfull=%d nbands=%d nindx=%d nz=%d nemline=%d\n",
                nspec, ntfull, nbands, nindx, nz, nemline);
        return 1;
    }
    if (nspec != n_spec || ntfull != n_time) {
        fprintf(stderr, "[C] Error: dims mismatch: get_dims=(%d,%d) get_nspec/ntfull=(%d,%d)\n",
                n_spec, n_time, nspec, ntfull);
        return 1;
    }
    if (!isfinite(zsol) || zsol <= 0.0) {
        fprintf(stderr, "[C] Error: invalid zsol=%g\n", zsol);
        return 1;
    }
    printf("[C] Dimensions: nspec=%d ntfull=%d nbands=%d nindx=%d nz=%d nemline=%d zsol=%g\n",
           nspec, ntfull, nbands, nindx, nz, nemline, zsol);

    fsps_context_set_int(handle, "sfh", 0, &status);
    if (check_status(status, "set sfh")) {
        return 1;
    }
    fsps_context_set_int(handle, "imf_type", 2, &status);
    if (check_status(status, "set imf_type")) {
        return 1;
    }
    fsps_context_set_float(handle, "imf_upper_limit", 100.0, &status);
    if (check_status(status, "set imf_upper_limit")) {
        return 1;
    }
    fsps_context_set_float(handle, "imf_lower_limit", 0.1, &status);
    if (check_status(status, "set imf_lower_limit")) {
        return 1;
    }
    fsps_context_set_str(handle, "imf_filename", "imf.dat", &status);
    if (check_status(status, "set imf_filename")) {
        return 1;
    }
    fsps_context_set_int(handle, "add_neb_emission", 1, &status);
    if (check_status(status, "set add_neb_emission")) {
        return 1;
    }

    fsps_context_set_fast_mode(handle, 1, &status);
    if (check_status(status, "set fast_mode=1")) {
        return 1;
    }
    fsps_context_set_fast_mode(handle, 0, &status);
    if (check_status(status, "set fast_mode=0")) {
        return 1;
    }

    fsps_context_set_int(handle, "not_a_param", 1, &status);
    if (status != FSPS_ERR_UNKNOWN_INT_PARAM) {
        fprintf(stderr, "[C] Error: expected unknown int param error, got status=%d\n", status);
        return 1;
    }
    fsps_context_set_float(handle, "no_float", 1.0, &status);
    if (status != FSPS_ERR_UNKNOWN_FLOAT_PARAM) {
        fprintf(stderr, "[C] Error: expected unknown float param error, got status=%d\n", status);
        return 1;
    }
    fsps_context_set_str(handle, "no_string", "x", &status);
    if (status != FSPS_ERR_UNKNOWN_STRING_PARAM) {
        fprintf(stderr, "[C] Error: expected unknown string param error, got status=%d\n", status);
        return 1;
    }

    size_t spec_elements = (size_t)nspec * (size_t)ntfull;
    double *spec_array = (double*)calloc(spec_elements, sizeof(double));
    double *mass = (double*)calloc((size_t)ntfull, sizeof(double));
    double *lbol = (double*)calloc((size_t)ntfull, sizeof(double));
    check_alloc(spec_array);
    check_alloc(mass);
    check_alloc(lbol);

    printf("[C] Computing SSP...\n");
    fsps_context_compute_ssp(handle, spec_array, mass, lbol, &status);
    if (check_status(status, "SSP compute failed")) {
        return 1;
    }

    if (!isfinite(spec_array[0]) || !isfinite(spec_array[spec_elements - 1]) ||
        !isfinite(mass[0]) || !isfinite(lbol[0])) {
        fprintf(stderr, "[C] Error: non-finite outputs detected\n");
        return 1;
    }

    printf("[C] Spec[0]=%e Spec[last]=%e Lbol[0]=%e\n",
           spec_array[0], spec_array[spec_elements - 1], lbol[0]);

    printf("[C] Simulating CSP-only parameter loop (dust2 updates)...\n");
    for (int i = 0; i < 50; ++i) {
        double dust2 = 0.05 + 0.002 * (double)i;
        fsps_context_set_float(handle, "dust2", dust2, &status);
        if (check_status(status, "set dust2 in CSP-only loop")) {
            return 1;
        }
    }

    fsps_context_destroy(handle, &status);
    if (check_status(status, "failed to destroy context")) {
        return 1;
    }

    free(spec_array);
    free(mass);
    free(lbol);

    printf("\n[C] C driver API test PASS\n");
    return 0;
}