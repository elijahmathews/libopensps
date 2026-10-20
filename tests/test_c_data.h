/* Shared by the C test programs that need the data: skip (exit status 77, meson's SKIP) when no
 * data root is found, instead of letting setup stop the program with an error. The search is
 * the library's (fsps_find_data_root in src/core/fsps_environment.f90), in its order:
 * OPENSPS_DATA_HOME, SPS_HOME, FSPS_DATA_HOME, $XDG_DATA_HOME/fsps, ~/.local/share/fsps,
 * /usr/share/fsps, /usr/local/share/fsps; a root is a standard one (opensps-data.json, or the
 * older fsps-data.json) or a legacy tree (data/allfilters.dat or data/FILTER_LIST). */
#ifndef TEST_C_DATA_H
#define TEST_C_DATA_H

#include <stdio.h>
#include <stdlib.h>

static int test_data_file_exists(const char *root, const char *suffix, const char *file) {
    char path[8192];
    FILE *f;
    if (snprintf(path, sizeof path, "%s%s/%s", root, suffix, file) >= (int)sizeof path) return 0;
    f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static int test_data_root_ok(const char *root, const char *suffix) {
    static const char *const markers[] = {"opensps-data.json", "fsps-data.json", "data/allfilters.dat",
                                          "data/FILTER_LIST"};
    if (root == NULL || root[0] == '\0') return 0;
    for (size_t i = 0; i < sizeof markers / sizeof markers[0]; i++)
        if (test_data_file_exists(root, suffix, markers[i])) return 1;
    return 0;
}

static void skip_without_data(void) {
    if (test_data_root_ok(getenv("OPENSPS_DATA_HOME"), "") || test_data_root_ok(getenv("SPS_HOME"), "") ||
        test_data_root_ok(getenv("FSPS_DATA_HOME"), "") || test_data_root_ok(getenv("XDG_DATA_HOME"), "/fsps") ||
        test_data_root_ok(getenv("HOME"), "/.local/share/fsps") || test_data_root_ok("/usr/share/fsps", "") ||
        test_data_root_ok("/usr/local/share/fsps", ""))
        return;
    printf("SKIP: no data root found (fetch a data release with tools/fetch_data.py; set OPENSPS_DATA_HOME "
           "or configure with -Ddata_home)\n");
    exit(77);
}

#endif
