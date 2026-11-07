/* The forms of opensps_context_setup that name fewer libraries (the C API chooses the Fortran
 * setup by which of the isochrone, spectral and dust arguments are empty): all forms give a context
 * of the same dimensions, because the defaults are the libraries the full form names
 * (mist, miles, DL07). Contexts with the same selection share one setup cache, so the three
 * extra setups cost little after the first. Requires a data root (OPENSPS_DATA_HOME), like the
 * regression tests; skipped without one. */
#include "opensps.h"
#include "test_c_data.h"

#include <stdio.h>

int main(void) {
    skip_without_data();
    const char *isoc[6] = {"mist", "mist", "mist", "", "", ""};
    const char *spec[6] = {"miles", "miles", "", "", "", "miles"};
    const char *dust[6] = {"DL07", "", "", "DL07", "", ""};
    const char *form[6] = {"isochrone, spectral and dust", "isochrone and spectral", "isochrone only (spectral and dust empty)",
                           "dust only (isochrone and spectral empty)", "none (all three empty)",
                           "spectral only (isochrone and dust empty)"};
    int failures = 0, nspec0 = 0, ntfull0 = 0;

    for (int k = 0; k < 6; k++) {
        int h = 0, s = 0, nspec = 0, ntfull = 0, nindx = 0;
        double zsol = 0.0;
        opensps_context_create(&h, &s);
        opensps_context_setup(-1, isoc[k], spec[k], dust[k], h, &s);
        if (s) { printf("FAIL: setup with %s, status %d\n", form[k], s); failures++; continue; }
        opensps_context_get_dims(h, &nspec, &ntfull, &s);
        opensps_context_get_nindx(h, &nindx, &s);
        opensps_context_get_zsol(h, &zsol, &s);
        printf("setup with %s: %d pixels, %d ages, %d indices, zsol %g\n", form[k], nspec, ntfull, nindx, zsol);
        if (nspec <= 0 || ntfull <= 0 || nindx <= 0 || zsol <= 0.0) {
            printf("FAIL: empty context after setup with %s\n", form[k]);
            failures++;
        }
        if (k == 0) { nspec0 = nspec; ntfull0 = ntfull; }
        else if (nspec != nspec0 || ntfull != ntfull0) {
            printf("FAIL: setup with %s gives %d x %d, the full form %d x %d\n", form[k], nspec, ntfull, nspec0, ntfull0);
            failures++;
        }
        opensps_context_destroy(h, &s);
    }
    printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
