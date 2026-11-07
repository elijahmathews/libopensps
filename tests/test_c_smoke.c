#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opensps.h>

int main(void) {
    printf("=== OpenSPS Installed Library Smoke Test ===\n");

    // 1. Version Queries
    int major = 0, minor = 0, patch = 0;
    opensps_get_driver_version(&major, &minor, &patch);
    printf("1. Driver version: %d.%d.%d\n", major, minor, patch);
    if (major != OPENSPS_API_VERSION_MAJOR ||
        minor != OPENSPS_API_VERSION_MINOR ||
        patch != OPENSPS_API_VERSION_PATCH) {
        fprintf(stderr, "FAIL: driver version mismatch with header macros\n");
        return 1;
    }

    const char *vstr = opensps_get_version_string();
    printf("2. Library version string: %s\n", vstr);
    if (!vstr || strcmp(vstr, OPENSPS_VERSION_STRING) != 0) {
        fprintf(stderr, "FAIL: version string mismatch\n");
        return 1;
    }

    const char *bid = opensps_get_build_id();
    printf("   Build id: \"%s\"\n", bid ? bid : "(null)");
    if (!bid) {
        fprintf(stderr, "FAIL: null build id\n");
        return 1;
    }
    // The release workflow sets this on tag builds: a release must not carry a build id.
    const char *expect_release = getenv("OPENSPS_EXPECT_RELEASE");  // CI sets it to "" off tags
    if (expect_release && expect_release[0] != '\0' && bid[0] != '\0') {
        fprintf(stderr, "FAIL: release build reports build id \"%s\"\n", bid);
        return 1;
    }

    // 2. FFI Metadata Queries
    int nbgrad = 0;
    opensps_get_nbgrad(&nbgrad);
    printf("3. Parameter gradients: %d\n", nbgrad);
    if (nbgrad != OPENSPS_NBGRAD) {
        fprintf(stderr, "FAIL: expected %d gradient parameters, got %d\n", OPENSPS_NBGRAD, nbgrad);
        return 1;
    }

    char name[64];
    int status = 0;
    opensps_get_grad_param_name(0, name, sizeof(name), &status);
    if (status != 0 || strlen(name) == 0) {
        fprintf(stderr, "FAIL: failed to get gradient parameter name for index 0\n");
        return 1;
    }
    printf("4. First gradient parameter: '%s'\n", name);

    // 3. Thread-safety Locking
    opensps_lock(&status);
    if (status != 0) { fprintf(stderr, "FAIL: opensps_lock status=%d\n", status); return 1; }
    opensps_unlock(&status);
    if (status != 0) { fprintf(stderr, "FAIL: opensps_unlock status=%d\n", status); return 1; }
    printf("5. Driver locking verified.\n");

    // 4. Context Lifecycle
    int h = 0;
    opensps_context_create(&h, &status);
    if (status != 0 || h <= 0) {
        fprintf(stderr, "FAIL: opensps_context_create status=%d handle=%d\n", status, h);
        return 1;
    }
    printf("6. Created context handle: %d\n", h);

    // 5. Physics Computation (if test data available)
    const char *data_home = getenv("OPENSPS_DATA_HOME");
    if (data_home && strlen(data_home) > 0) {
        printf("7. Initializing context with test data from: %s\n", data_home);
        opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
        if (status != 0) {
            fprintf(stderr, "FAIL: opensps_context_setup returned status %d\n", status);
            return 1;
        }

        int nbands = 0, nspec = 0, ntfull = 0;
        opensps_context_get_nbands(h, &nbands, &status);
        opensps_context_get_nspec(h, &nspec, &status);
        opensps_context_get_ntfull(h, &ntfull, &status);
        printf("   Grid dimensions: %d bands, %d wavelengths, %d SSP ages\n", nbands, nspec, ntfull);
        if (nbands <= 0 || nspec <= 0 || ntfull <= 0) {
            fprintf(stderr, "FAIL: invalid grid dimensions\n");
            return 1;
        }

        double *spec = (double *)malloc(sizeof(double) * nspec * ntfull);
        double *mass = (double *)malloc(sizeof(double) * ntfull);
        double *lbol = (double *)malloc(sizeof(double) * ntfull);

        opensps_context_compute_ssp(h, spec, mass, lbol, &status);
        if (status != 0) {
            fprintf(stderr, "FAIL: opensps_context_compute_ssp returned status %d\n", status);
            return 1;
        }

        printf("8. Computed SSP grid: mass[0]=%.4e, lbol[0]=%.4e, spec[0]=%.4e\n", mass[0], lbol[0], spec[0]);
        if (!isfinite(mass[0]) || mass[0] <= 0.0 || !isfinite(lbol[0]) || !isfinite(spec[0])) {
            fprintf(stderr, "FAIL: computed non-finite or invalid physical values\n");
            return 1;
        }

        free(spec);
        free(mass);
        free(lbol);
    } else {
        printf("7. OPENSPS_DATA_HOME not set; skipping data-dependent grid computation.\n");
    }

    // 6. Context Cleanup
    opensps_context_destroy(h, &status);
    if (status != 0) {
        fprintf(stderr, "FAIL: opensps_context_destroy status=%d\n", status);
        return 1;
    }
    printf("9. Context destroyed cleanly.\n");

    printf("=== SUCCESS: Installed libopensps package verified fully functional! ===\n");
    return 0;
}
